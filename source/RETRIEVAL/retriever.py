"""
Hybrid retriever: dense (FAISS) + keyword (BM25), fused with RRF, with
optional overlap de-duplication and cross-encoder re-ranking.

The optional stages are off by default so a baseline can be measured first and
each addition attributed separately:
    metadata_filter=False -- restrict the search to filings matching the
                             company and fiscal year named in the question
    dedup=False           -- drop adjacent chunks that overlap the same text
    rerank=False          -- re-score candidates with a cross-encoder

Position N means the same chunk in every index (FAISS vector N, BM25
document N, line N of chunks.jsonl and chunk_meta.jsonl). That invariant is
what makes fusion possible; it holds because every builder streamed
chunks.jsonl in file order.
"""
import json
import sys
from pathlib import Path

import bm25s
import faiss
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so this module also works when run directly

from source.RETRIEVAL.metadata_filter import MetadataFilter  # noqa: E402
PROCESSED = ROOT / "data" / "PROCESSED"
CHUNKS_PATH = PROCESSED / "chunks.jsonl"
META_PATH = PROCESSED / "chunk_meta.jsonl"
FAISS_PATH = PROCESSED / "faiss.index"
BM25_DIR = PROCESSED / "bm25_index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Smaller/faster than BAAI/bge-reranker-base, which matters on CPU. Swap if
# accuracy matters more than latency.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60  # standard RRF damping constant
CANDIDATES_K = 20  # how many each retriever contributes before fusion

# Many questions name the statement(s) they need ("basing your answers off of
# the cash flow statement and the income statement"). When that hint is
# present, searching per-statement covers several statements deliberately
# instead of hoping one large top-k happens to span them. Each key maps the
# phrases a question might use to the heading text the filing actually prints.
STATEMENT_HINTS = {
    "cash_flow": (
        ("cash flow statement", "statement of cash flows", "cash flows", "statements of cash flow"),
        "consolidated statements of cash flows operating activities",
    ),
    "income": (
        ("income statement", "statement of income", "statements of income", "p&l",
         "profit and loss", "statement of operations", "statements of operations"),
        "consolidated statements of income net revenue operating income",
    ),
    "balance_sheet": (
        ("balance sheet", "statement of financial position", "balance sheets"),
        "consolidated balance sheets total current assets liabilities",
    ),
}
# Fusion can yield up to 2x CANDIDATES_K unique chunks; cap what the
# cross-encoder scores, since it runs on CPU and cannot be precomputed.
RERANK_CANDIDATES = 20


class Retriever:
    def __init__(self, use_reranker: bool = False):
        print("Loading indexes ...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.stemmer = Stemmer.Stemmer("english")
        self.faiss_index = faiss.read_index(str(FAISS_PATH))
        self.bm25 = bm25s.BM25.load(str(BM25_DIR), mmap=True)
        self.meta = self._load_meta()

        self._chunks_file = open(CHUNKS_PATH, "rb")  # kept open for seek-reads

        self.positions_by_doc: dict[str, list[int]] = {}
        for position, meta in enumerate(self.meta):
            self.positions_by_doc.setdefault(meta["doc_name"], []).append(position)
        self.metadata_filter = MetadataFilter(list(self.positions_by_doc))
        # Reused across queries so we don't allocate a 107K-element array per search.
        self._mask_buffer = np.zeros(len(self.meta), dtype=np.float32)

        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder

            print(f"Loading reranker: {RERANK_MODEL}")
            self.reranker = CrossEncoder(RERANK_MODEL)

        if self.faiss_index.ntotal != len(self.meta):
            raise RuntimeError(
                f"Index mismatch: FAISS has {self.faiss_index.ntotal} vectors "
                f"but metadata has {len(self.meta)} rows. Rebuild the indexes."
            )
        print(f"Ready: {len(self.meta)} chunks indexed.")

    @staticmethod
    def _load_meta() -> list[dict]:
        with open(META_PATH, "rb") as f:
            return [json.loads(line) for line in f]

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: bool = False,
        dedup: bool = False,
        rerank: bool = False,
        oracle_doc: str | None = None,
        targeted: bool = False,
    ) -> list[dict]:
        allowed_positions = None
        if oracle_doc is not None:
            # Diagnostic only: uses the benchmark's ground-truth document, which
            # a real system would not know. Measures the ceiling that perfect
            # company/year extraction could reach.
            allowed_positions = self._positions_for_docs({oracle_doc})
        elif metadata_filter:
            allowed_positions = self._allowed_positions(query)

        queries = [query]
        if targeted:
            # Only splits when the question actually names statements; otherwise
            # this is exactly the single-query path.
            queries = self._statement_queries(query) or [query]

        ranked_lists = []
        for sub_query in queries:
            ranked_lists.append(self._dense_search(sub_query, CANDIDATES_K, allowed_positions))
            ranked_lists.append(self._bm25_search(sub_query, CANDIDATES_K, allowed_positions))
        fused = self._rrf_fuse(ranked_lists)

        if dedup:
            fused = self._drop_overlapping(fused)

        if rerank:
            if self.reranker is None:
                raise RuntimeError("Retriever was built with use_reranker=False.")
            fused = self._rerank(query, fused, top_k)

        return [self._build_result(pos, rank) for rank, pos in enumerate(fused[:top_k], 1)]

    @staticmethod
    def _statement_queries(query: str) -> list[str]:
        """
        One sub-query per financial statement the question names, or [] if it
        names none -- in which case the caller keeps the original single query.

        Each sub-query appends the heading wording filings actually print, which
        steers BM25 toward the statement itself rather than passing mentions.
        """
        lowered = query.lower()
        sub_queries = []
        for phrases, heading in STATEMENT_HINTS.values():
            if any(phrase in lowered for phrase in phrases):
                sub_queries.append(f"{query} {heading}")
        # A single hint adds nothing over the plain query, so only split when
        # the question genuinely spans more than one statement.
        return sub_queries if len(sub_queries) > 1 else []

    def _allowed_positions(self, query: str) -> np.ndarray | None:
        """Chunk positions belonging to filings that match the question's company/year."""
        allowed_docs = self.metadata_filter.allowed_docs(query)
        if allowed_docs is None:
            return None  # company not identified -- search everything
        return self._positions_for_docs(allowed_docs)

    def _positions_for_docs(self, doc_names: set[str]) -> np.ndarray | None:
        positions = [p for doc in doc_names for p in self.positions_by_doc.get(doc, [])]
        return np.array(sorted(positions), dtype=np.int64) if positions else None

    def _dense_search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        # normalize_embeddings must match how chunks were embedded, or the
        # inner-product index returns meaningless distances.
        vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        params = None
        if allowed is not None:
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
        _scores, positions = self.faiss_index.search(vec, k, params=params)
        return [int(p) for p in positions[0] if p >= 0]

    def _bm25_search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        # Same stopwords + stemmer as bm25_builder.py, so query terms map onto
        # the index's vocabulary.
        tokens = bm25s.tokenize(
            query, stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False
        )

        mask = None
        if allowed is not None:
            self._mask_buffer.fill(0.0)
            self._mask_buffer[allowed] = 1.0
            mask = self._mask_buffer
            k = min(k, len(allowed))

        positions, _scores = self.bm25.retrieve(
            tokens, k=k, weight_mask=mask, show_progress=False
        )
        return [int(p) for p in positions[0]]

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
        """Reciprocal Rank Fusion: score by 1/(k + rank), summed across lists."""
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, position in enumerate(ranked, 1):
                scores[position] = scores.get(position, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=scores.get, reverse=True)

    def _drop_overlapping(self, positions: list[int]) -> list[int]:
        """
        Chunks step 1700 chars but span 2000, so only *adjacent* chunks of the
        same document share text. Keep whichever ranked higher.
        """
        kept: list[int] = []
        for pos in positions:
            meta = self.meta[pos]
            doc, idx = meta["doc_name"], self._chunk_index(meta["chunk_id"])
            if any(
                self.meta[k]["doc_name"] == doc
                and abs(self._chunk_index(self.meta[k]["chunk_id"]) - idx) <= 1
                for k in kept
            ):
                continue
            kept.append(pos)
        return kept

    @staticmethod
    def _chunk_index(chunk_id: str) -> int:
        return int(chunk_id.rsplit("__", 1)[1])

    def _rerank(self, query: str, positions: list[int], top_k: int) -> list[int]:
        """
        Re-score candidates by reading query and chunk text together.

        Unlike the bi-encoder, this sees the query while reading the chunk, and
        reads up to 512 tokens -- so it is not subject to the 256-token
        truncation that limits our stored vectors.
        """
        candidates = positions[:RERANK_CANDIDATES]
        rest = positions[RERANK_CANDIDATES:]

        pairs = [(query, self._rerank_document(p)) for p in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order] + rest

    def _rerank_document(self, position: int) -> str:
        """
        Chunk text prefixed with a context header, so the cross-encoder can
        distinguish near-identical filings (3M's 2018 vs 2021 10-K read almost
        the same without knowing which is which).

        Only the reranker sees this; FAISS and BM25 scores come from indexes
        built at ingest time on raw text.
        """
        meta = self.meta[position]
        header = (
            f"[Document: {meta['doc_name']} | "
            f"Page: {meta['page_start']}-{meta['page_end']}]"
        )
        return f"{header}\n{self._chunk_text(position)}"

    def _chunk_text(self, position: int) -> str:
        """Seek straight to this chunk's line instead of holding 230 MB in RAM."""
        self._chunks_file.seek(self.meta[position]["byte_offset"])
        return json.loads(self._chunks_file.readline().decode("utf-8"))["text"]

    def _build_result(self, position: int, rank: int) -> dict:
        meta = self.meta[position]
        return {
            "rank": rank,
            "position": position,
            "chunk_id": meta["chunk_id"],
            "doc_name": meta["doc_name"],
            "page_start": meta["page_start"],
            "page_end": meta["page_end"],
            "text": self._chunk_text(position),
        }

    def close(self):
        self._chunks_file.close()


def main():
    """Quick manual check that retrieval returns something sensible."""
    retriever = Retriever(use_reranker=False)
    query = "What was 3M's FY2018 net income including noncontrolling interest?"

    print(f"\nQuery: {query}\n")
    for hit in retriever.search(query, top_k=5):
        preview = " ".join(hit["text"].split())[:160]
        print(f"{hit['rank']}. {hit['doc_name']} p.{hit['page_start']}-{hit['page_end']} "
              f"[{hit['chunk_id']}]")
        print(f"   {preview.encode('ascii', 'replace').decode('ascii')}\n")

    retriever.close()


if __name__ == "__main__":
    main()
