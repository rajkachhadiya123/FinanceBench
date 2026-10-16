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
version, so changing the model, prompt, index or guardrails never serves a stale answer.

Guardrails (source/GUARDRAILS/), in request order:
    1. rate limit per client      -> 429 with Retry-After
    2. input check                -> 400 only for clear abuse (hidden characters,
                                     unreadable text, prompt injection)
    3. cache / pipeline
    4. "guardrails" in the response: warnings, output checks, disclaimer --
       signals only; the answer is never changed
"""
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from source.API.pipeline import GenerationFailed, QAPipeline
from source.GENERATION.answer_generator import PROMPT_VERSION, DailyLimitReached
from source.GUARDRAILS.guards import DISCLAIMER, check_input, needs_disclaimer
from source.GUARDRAILS.rate_limit import RateLimiter
from source.SHARED.cache import AnswerCache

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = Path(os.getenv("FILINGLENS_CACHE_PATH", ROOT / "data" / "PROCESSED" / "cache.db"))
NO_COMPANY_WARNING = ("No company was recognised in the question, so all filings were "
                      "searched; naming the company and fiscal year gives more reliable answers.")

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["cache"] = AnswerCache(CACHE_PATH)
    state["limiter"] = RateLimiter()
    state["pipeline"] = QAPipeline()
    yield
    state["pipeline"].close()
    state["cache"].close()


app = FastAPI(
    title="FilingLens API",
    version="1.1.0",
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
    rerank_score: float | None = Field(None, description="Cohere relevance 0-1; null if not reranked")


class OutputChecks(BaseModel):
    grounded: bool | None = Field(
        None, description="At least half of the answer's figures appear in the retrieved text; "
                          "null if the answer has no figures or is NOT FOUND")
    grounding_ratio: float | None
    figures_checked: int
    ungrounded_figures: list[float]
    retrieval_confidence: float | None = Field(None, description="Best rerank score among chunks used")
    low_confidence: bool


class Guardrails(BaseModel):
    warnings: list[str]
    output: OutputChecks | None = None
    disclaimer: str | None = None


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
    guardrails: Guardrails
    cached: bool
    latency_ms: int
    pipeline_version: str


class RetrieveResponse(BaseModel):
    question: str
    filings_searched: list[str] | None
    chunks: list[Chunk]
    guardrails: Guardrails
    cached: bool
    latency_ms: int
    pipeline_version: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    pipeline_version: str
    prompt_version: str
    rate_limits_per_minute: dict
    cache_path: str
    cache: dict


# --- guard helpers ----------------------------------------------------------------
def guard_request(request: Request, group: str, question: str) -> None:
    """Rate limit, then input check; raises before any cache lookup or API call."""
    client = request.client.host if request.client else "unknown"
    wait = state["limiter"].check(client, group)
    if wait:
        raise HTTPException(429, f"Rate limit: {state['limiter'].limits[group]} {group} "
                                 f"requests per minute.",
                            headers={"Retry-After": str(math.ceil(wait))})
    verdict = check_input(question)
    if verdict["blocked"]:
        raise HTTPException(400, {"guardrail": "input", "reason": verdict["reason"]})


def warnings_for(filings: list[str] | None) -> list[str]:
    return [NO_COMPANY_WARNING] if filings is None else []


# --- endpoints ------------------------------------------------------------------
# Plain "def" endpoints run in FastAPI's thread pool, so a slow LLM call does not
# block the server's event loop.

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    guard_request(request, "query", req.question)
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

    guardrails = Guardrails(
        warnings=warnings_for(body["filings_searched"]),
        output=OutputChecks(**body["output_checks"]),
        disclaimer=DISCLAIMER if needs_disclaimer(req.question) else None,
    )
    return QueryResponse(
        question=req.question, cached=hit is not None, pipeline_version=pipeline.version,
        latency_ms=round((time.perf_counter() - start) * 1000), guardrails=guardrails,
        **{k: v for k, v in body.items() if k not in ("tokens", "output_checks")},
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
    guard_request(request, "retrieve", req.question)
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
        latency_ms=round((time.perf_counter() - start) * 1000),
        guardrails=Guardrails(warnings=warnings_for(body["filings_searched"])), **body,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline: QAPipeline = state["pipeline"]
    return HealthResponse(
        status="ok", chunks_indexed=pipeline.chunks_indexed, pipeline_version=pipeline.version,
        prompt_version=PROMPT_VERSION, rate_limits_per_minute=state["limiter"].limits,
        cache_path=str(CACHE_PATH), cache=state["cache"].stats(),
    )


@app.delete("/cache")
def clear_cache() -> dict:
    return {"removed": state["cache"].clear()}
