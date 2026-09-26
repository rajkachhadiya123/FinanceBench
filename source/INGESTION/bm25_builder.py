"""
BM25 keyword index builder.

Reads data/PROCESSED/chunks.jsonl and builds a BM25 index over the chunk
texts, for the keyword half of hybrid retrieval.

Uses bm25s rather than rank_bm25: same BM25 algorithm, but backed by scipy
sparse matrices instead of Python lists, which keeps memory usage viable for
107K documents on a machine with limited free RAM.

Streams chunks.jsonl in file order, matching embedder.py, so BM25 document N
and FAISS vector N refer to the same chunk.
"""
import json
import time
from pathlib import Path

import bm25s
import Stemmer

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "PROCESSED" / "chunks.jsonl"
BM25_DIR = ROOT / "data" / "PROCESSED" / "bm25_index"


def load_chunk_texts(path: Path) -> list[str]:
    texts = []
    with open(path, "rb") as f:
        for raw_line in f:
            texts.append(json.loads(raw_line.decode("utf-8"))["text"])
    return texts


def main():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}. Run chunker.py first.")

    start_time = time.time()

    print(f"Reading chunk texts from {CHUNKS_PATH.name} ...")
    texts = load_chunk_texts(CHUNKS_PATH)
    print(f"  {len(texts)} chunks loaded ({time.time() - start_time:.0f}s)")

    # Stemming collapses "reported"/"reporting"/"reports" to one term, so a
    # question's wording doesn't have to match the filing's wording exactly.
    stemmer = Stemmer.Stemmer("english")

    print("Tokenizing (lowercase, stopword removal, stemming) ...")
    token_ids = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=True)
    del texts  # free the raw text before building the index

    print("Building BM25 index ...")
    retriever = bm25s.BM25()
    retriever.index(token_ids, show_progress=True)

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(BM25_DIR))

    print(f"\nBM25 index written to {BM25_DIR}")
    print(f"Total time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
