"""
The final FilingLens pipeline (119/150 = 79.3% on FinanceBench), loaded once and
shared by every API request.

    question -> metadata filter (company + fiscal year)
             -> Qwen3 dense top 20 + BM25 top 20 on headed chunks -> RRF
             -> Cohere rerank -> top 5 -> Cerebras gpt-oss-120b (prompt v2)
             -> on NOT FOUND: answer again with the top 10 of the same reranked list

Same steps as tools/test_answers.py with --embedding qwen_headed, except that the
reranked list is fetched once at the larger k and sliced: the fallback then costs
no second Cohere call, and its top 10 is exactly the list the top 5 came from.

The retriever is not thread-safe (it seeks in a shared file handle and reuses a
mask buffer), and both APIs are rate-limited anyway, so calls are serialized.
"""
import re
import threading
from dataclasses import dataclass

from source.GENERATION.answer_generator import (
    PROMPT_VERSION,
    REASONING_EFFORT,
    AnswerGenerator,
)
from source.GUARDRAILS.guards import GUARDRAILS_VERSION, check_output
from source.RETRIEVAL.metadata_filter import FILENAME_TYPOS, YEAR_IN_NAME_RE, parse_doc_name
from source.RETRIEVAL.retriever import Retriever
from source.TRACING.tracer import current_trace, span


DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class GenerationFailed(RuntimeError):
    """The LLM call failed after retries (network or provider error)."""


@dataclass(frozen=True)
class PipelineConfig:
    embedding: str = "qwen_headed"
    reranker: str = "cohere"
    top_k: int = 5          # chunks the LLM sees first
    retry_top_k: int = 10   # chunks on the single retry after NOT FOUND


def is_not_found(answer: str) -> bool:
    return answer.strip().upper().startswith("NOT FOUND")


def split_header(text: str) -> tuple[str | None, str]:
    """A headed chunk starts with '[COMPANY | TYPE | PERIOD | page N | section]'."""
    first, _, rest = text.partition("\n")
    if first.startswith("[") and first.endswith("]") and rest:
        return first, rest
    return None, text


class QAPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.retriever = Retriever(use_reranker=config.reranker, embedding=config.embedding)
        self.generator = AnswerGenerator()
        self._lock = threading.Lock()
        reranker = getattr(self.retriever.reranker, "model", config.reranker)
        self.version = (f"{config.embedding}+bm25+{reranker}|{self.generator.model}"
                        f"|prompt-{PROMPT_VERSION}|{REASONING_EFFORT}"
                        f"|k{config.top_k}/{config.retry_top_k}|guards-{GUARDRAILS_VERSION}")

    @property
    def chunks_indexed(self) -> int:
        return len(self.retriever.meta)

    def documents(self) -> list[dict]:
        """Every indexed filing, grouped by company (built from the loaded indexes)."""
        return catalogue(self.retriever.positions_by_doc)

    def embedding_cache_stats(self) -> dict:
        return self.retriever.embedding_cache.stats()

    def filings_for(self, question: str) -> list[str] | None:
        """Filings the metadata filter keeps; None means no company was recognised."""
        docs = self.retriever.metadata_filter.allowed_docs(question)
        return sorted(docs) if docs is not None else None

    def retrieve(self, question: str, top_k: int, rerank: bool = True) -> list[dict]:
        with self._lock:
            return [public_chunk(c) for c in self._search(question, top_k, rerank)]

    def answer(self, question: str) -> dict:
        cfg = self.config
        with self._lock:
            chunks = self._search(question, max(cfg.top_k, cfg.retry_top_k), rerank=True)
            used = chunks[:cfg.top_k]
            result = self._generate(question, used, "llm.answer")
            retried = False
            if (result["answer"] != "ERROR" and is_not_found(result["answer"])
                    and cfg.retry_top_k > cfg.top_k and len(chunks) > cfg.top_k):
                used, retried = chunks, True
                result = self._generate(question, used, "llm.retry_top10")
        if result["answer"] == "ERROR":
            raise GenerationFailed(result.get("error") or "LLM call failed")
        chunks_out = [public_chunk(c) for c in used]
        not_found = is_not_found(result["answer"])
        return {
            "answer": result["answer"],
            "not_found": not_found,
            "reasoning": result.get("reasoning", ""),
            "citations": result.get("citations", ""),
            "retried_with_more_chunks": retried,
            "chunks": chunks_out,
            # Signals only -- computed on chunk text without its header, so a header's
            # page number can never count as evidence for a figure.
            "output_checks": check_output(result["answer"], result.get("reasoning", ""),
                                          chunks_out, not_found),
            "tokens": {"input": result.get("input_tokens", 0), "output": result.get("output_tokens", 0)},
        }

    def _search(self, question: str, top_k: int, rerank: bool) -> list[dict]:
        trace = current_trace()
        if trace is not None and rerank:
            trace.update(rerank_calls=trace.fields.get("rerank_calls", 0) + 1)
        return self.retriever.search(question, top_k=top_k, metadata_filter=True, rerank=rerank)

    def _generate(self, question: str, chunks: list[dict], step: str) -> dict:
        """One LLM call, traced as its own span with tokens and the answer's start."""
        with span(step, chunks=len(chunks)) as s:
            result = self.generator.generate(question, chunks)
            s.set(input_tokens=result.get("input_tokens", 0),
                  output_tokens=result.get("output_tokens", 0),
                  answer_start=result["answer"][:60], error=result.get("error"))
        trace = current_trace()
        if trace is not None:
            trace.add_tokens(result.get("input_tokens", 0), result.get("output_tokens", 0))
        return result

    def close(self) -> None:
        self.retriever.close()


def catalogue(positions_by_doc: dict[str, list[int]]) -> list[dict]:
    """
    [{"company": "AMERICAN EXPRESS", "key": "americanexpress", "filings": [...]}, ...]

    "key" is the normalized name the metadata filter matches against; a filing's
    "type" is what follows the period in its name (10K, 10Q, 8K, EARNINGS, ...).
    """
    companies: dict[str, dict] = {}
    for doc_name, positions in positions_by_doc.items():
        key, year, quarter, doc_type = parse_doc_name(doc_name)
        key = FILENAME_TYPOS.get(key, key)  # same mapping the metadata filter uses
        parts = doc_name.split("_")
        period_at = next((i for i, p in enumerate(parts) if YEAR_IN_NAME_RE.match(p)), 1)
        filed = DATE_RE.search(doc_type)    # 8-Ks carry their filing date in the name
        entry = companies.setdefault(key, {"company": " ".join(parts[:period_at]).upper(),
                                           "key": key, "filings": []})
        entry["filings"].append({"doc_name": doc_name,
                                 "type": re.split(r"[_-]", doc_type)[0].upper() or "FILING",
                                 "fiscal_year": year, "quarter": quarter,
                                 "filed": filed.group() if filed else None,
                                 "chunks": len(positions)})
    for entry in companies.values():
        entry["filings"].sort(key=lambda f: (f["fiscal_year"] or 0, f["quarter"] or "", f["doc_name"]))
    return sorted(companies.values(), key=lambda e: e["key"])


def public_chunk(chunk: dict) -> dict:
    header, text = split_header(chunk["text"])
    return {
        "rank": chunk["rank"],
        "doc_name": chunk["doc_name"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "header": header,
        "text": text,
        "rerank_score": chunk.get("rerank_score"),
    }
