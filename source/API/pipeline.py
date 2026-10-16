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
import threading
from dataclasses import dataclass

from source.GENERATION.answer_generator import (
    PROMPT_VERSION,
    REASONING_EFFORT,
    AnswerGenerator,
)
from source.RETRIEVAL.retriever import Retriever


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
                        f"|k{config.top_k}/{config.retry_top_k}")

    @property
    def chunks_indexed(self) -> int:
        return len(self.retriever.meta)

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
            result = self.generator.generate(question, used)
            retried = False
            if (result["answer"] != "ERROR" and is_not_found(result["answer"])
                    and cfg.retry_top_k > cfg.top_k and len(chunks) > cfg.top_k):
                used, retried = chunks, True
                result = self.generator.generate(question, used)
        if result["answer"] == "ERROR":
            raise GenerationFailed(result.get("error") or "LLM call failed")
        return {
            "answer": result["answer"],
            "not_found": is_not_found(result["answer"]),
            "reasoning": result.get("reasoning", ""),
            "citations": result.get("citations", ""),
            "retried_with_more_chunks": retried,
            "chunks": [public_chunk(c) for c in used],
            "tokens": {"input": result.get("input_tokens", 0), "output": result.get("output_tokens", 0)},
        }

    def _search(self, question: str, top_k: int, rerank: bool) -> list[dict]:
        return self.retriever.search(question, top_k=top_k, metadata_filter=True, rerank=rerank)

    def close(self) -> None:
        self.retriever.close()


def public_chunk(chunk: dict) -> dict:
    header, text = split_header(chunk["text"])
    return {
        "rank": chunk["rank"],
        "doc_name": chunk["doc_name"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "header": header,
        "text": text,
    }
