"""
FilingLens HTTP API.

    POST   /query     question -> answer, citations, the chunks the answer used
    POST   /retrieve  question -> top chunks only (no LLM call; rerank optional)
    GET    /health    status, index size, pipeline version, cache stats
    DELETE /cache     empty the answer cache

Run from the repository root:
    python -m uvicorn source.API.app:app --port 8000
Then open http://127.0.0.1:8000/docs for interactive documentation.

Startup loads the Qwen model, the FAISS and BM25 indexes and the API clients once
(~30-60 s on CPU); the server accepts requests only after that. Answers are cached
in SQLite (data/PROCESSED/cache.db, or FILINGLENS_CACHE_PATH), keyed by the pipeline
version, so changing the model, prompt or index never serves a stale answer.
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from source.API.pipeline import GenerationFailed, QAPipeline
from source.GENERATION.answer_generator import PROMPT_VERSION, DailyLimitReached
from source.SHARED.cache import AnswerCache

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = Path(os.getenv("FILINGLENS_CACHE_PATH", ROOT / "data" / "PROCESSED" / "cache.db"))

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["cache"] = AnswerCache(CACHE_PATH)
    state["pipeline"] = QAPipeline()
    yield
    state["pipeline"].close()
    state["cache"].close()


app = FastAPI(
    title="FilingLens API",
    version="1.0.0",
    description="Question answering over SEC filings (FinanceBench: 79.3% end-to-end).",
    lifespan=lifespan,
)


# --- request / response models ------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000,
                          examples=["What was 3M's FY2018 net income?"])
    use_cache: bool = True


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = Field(True, description="Cohere rerank (costs one API call); false = RRF order")
    use_cache: bool = True


class Chunk(BaseModel):
    rank: int
    doc_name: str
    page_start: int
    page_end: int
    header: str | None = Field(None, description="[COMPANY | TYPE | PERIOD | page | section]")
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    not_found: bool
    reasoning: str
    citations: str
    retried_with_more_chunks: bool
    filings_searched: list[str] | None = Field(
        None, description="Filings kept by the metadata filter; null = no company recognised, all searched")
    chunks: list[Chunk]
    cached: bool
    latency_ms: int
    pipeline_version: str


class RetrieveResponse(BaseModel):
    question: str
    filings_searched: list[str] | None
    chunks: list[Chunk]
    cached: bool
    latency_ms: int
    pipeline_version: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    pipeline_version: str
    prompt_version: str
    cache_path: str
    cache: dict


# --- endpoints ------------------------------------------------------------------
# Plain "def" endpoints run in FastAPI's thread pool, so a slow LLM call does not
# block the server's event loop.

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline: QAPipeline = state["pipeline"]
    cache: AnswerCache = state["cache"]
    start = time.perf_counter()

    hit = cache.get("query", req.question, pipeline.version) if req.use_cache else None
    if hit is None:
        try:
            body = pipeline.answer(req.question)
        except DailyLimitReached as exc:
            raise HTTPException(503, f"LLM quota exhausted, try later: {exc}") from exc
        except GenerationFailed as exc:
            raise HTTPException(502, f"LLM call failed: {exc}") from exc
        body["filings_searched"] = pipeline.filings_for(req.question)
        cache.set("query", req.question, pipeline.version, body)
    else:
        body = hit

    return QueryResponse(
        question=req.question, cached=hit is not None, pipeline_version=pipeline.version,
        latency_ms=round((time.perf_counter() - start) * 1000),
        **{k: v for k, v in body.items() if k != "tokens"},
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    pipeline: QAPipeline = state["pipeline"]
    cache: AnswerCache = state["cache"]
    params = f"k={req.top_k}|rerank={req.rerank}"
    start = time.perf_counter()

    hit = cache.get("retrieve", req.question, pipeline.version, params) if req.use_cache else None
    if hit is None:
        body = {"chunks": pipeline.retrieve(req.question, req.top_k, req.rerank),
                "filings_searched": pipeline.filings_for(req.question)}
        cache.set("retrieve", req.question, pipeline.version, body, params)
    else:
        body = hit

    return RetrieveResponse(
        question=req.question, cached=hit is not None, pipeline_version=pipeline.version,
        latency_ms=round((time.perf_counter() - start) * 1000), **body,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline: QAPipeline = state["pipeline"]
    return HealthResponse(
        status="ok", chunks_indexed=pipeline.chunks_indexed, pipeline_version=pipeline.version,
        prompt_version=PROMPT_VERSION, cache_path=str(CACHE_PATH), cache=state["cache"].stats(),
    )


@app.delete("/cache")
def clear_cache() -> dict:
    return {"removed": state["cache"].clear()}
