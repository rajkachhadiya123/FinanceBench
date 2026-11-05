"""
FilingLens HTTP API.

    POST   /query           question -> answer, citations, the chunks the answer used
    POST   /retrieve        question -> top chunks only (no LLM call; rerank optional)
    GET    /health          status, index size, pipeline version, cache stats
    DELETE /cache           empty the answer cache
    GET    /traces          latest traced requests
    GET    /traces/{id}     one request step by step (id = X-Request-ID header / trace_id)
    GET    /stats?hours=24  volume, errors, cache hit rate, guardrails, tokens, cost, p50/p95 per step
    GET    /documents       companies and filings the API can answer about (?company= filter)

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

Tracing (source/TRACING/): every /query and /retrieve request gets a trace id, returned
in the X-Request-ID header and as trace_id in the body, and is stored with its timed
steps in data/PROCESSED/traces.db (FILINGLENS_TRACE_PATH) for
FILINGLENS_TRACE_RETENTION_DAYS days (default 30). Cached answers have no expiry.
"""
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from source.API.pipeline import GenerationFailed, QAPipeline
from source.GENERATION.answer_generator import PROMPT_VERSION, DailyLimitReached
from source.GUARDRAILS.guards import DISCLAIMER, check_input, needs_disclaimer
from source.GUARDRAILS.rate_limit import RateLimiter
from source.RETRIEVAL.metadata_filter import normalize as normalize_name
from source.SHARED.cache import AnswerCache, make_key
from source.TRACING.costs import PRICES, estimate_cost
from source.TRACING.tracer import TraceStore, current_trace, span, start_trace

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = Path(os.getenv("FILINGLENS_CACHE_PATH", ROOT / "data" / "PROCESSED" / "cache.db"))
TRACE_PATH = Path(os.getenv("FILINGLENS_TRACE_PATH", ROOT / "data" / "PROCESSED" / "traces.db"))
TRACED_PATHS = {"/query", "/retrieve"}  # /health, /docs, /traces and /stats are not traced
NO_COMPANY_WARNING = ("No company was recognised in the question, so all filings were "
                      "searched; naming the company and fiscal year gives more reliable answers.")

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["cache"] = AnswerCache(CACHE_PATH)
    state["traces"] = TraceStore(TRACE_PATH)
    state["traces"].purge_expired()  # traces have a lifetime; cached answers do not
    state["limiter"] = RateLimiter()
    state["pipeline"] = QAPipeline()
    yield
    state["pipeline"].close()
    state["cache"].close()
    state["traces"].close()


app = FastAPI(
    title="FilingLens API",
    version="1.2.0",
    description="Question answering over SEC filings (FinanceBench: 79.3% end-to-end).",
    lifespan=lifespan,
)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """One trace per /query or /retrieve request; its id is returned as X-Request-ID."""
    if request.url.path not in TRACED_PATHS:
        return await call_next(request)
    with start_trace(request.url.path, state.get("traces")) as trace:
        trace.update(client=request.client.host if request.client else "unknown")
        try:
            response = await call_next(request)
        except Exception as exc:  # unexpected failure: record it and answer 500
            trace.update(error=f"{type(exc).__name__}: {exc}"[:500])
            response = JSONResponse({"detail": "Internal server error"}, status_code=500)
        trace.update(status_code=response.status_code, cost_usd=estimate_cost(trace.fields))
        response.headers["X-Request-ID"] = trace.trace_id
    return response


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
        None, description="Enough of the answer's figures appear in the retrieved text; "
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
    trace_id: str | None = Field(None, description="Look this request up at GET /traces/{trace_id}")


class RetrieveResponse(BaseModel):
    question: str
    filings_searched: list[str] | None
    chunks: list[Chunk]
    guardrails: Guardrails
    cached: bool
    latency_ms: int
    pipeline_version: str
    trace_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    pipeline_version: str
    prompt_version: str
    rate_limits_per_minute: dict
    cache_path: str
    cache: dict
    trace_path: str
    embedding_cache: dict
    prices_usd: dict


# --- guard helpers ----------------------------------------------------------------
def guard_request(request: Request, group: str, question: str) -> None:
    """Rate limit, then input check; raises before any cache lookup or API call."""
    client = request.client.host if request.client else "unknown"
    trace = current_trace()
    if trace is not None:
        trace.update(question=question)
    with span("guard.rate_limit") as s:
        wait = state["limiter"].check(client, group)
        s.set(allowed=not wait)
    if wait:
        if trace is not None:
            trace.update(blocked_reason="rate_limit")
        raise HTTPException(429, f"Rate limit: {state['limiter'].limits[group]} {group} "
                                 f"requests per minute.",
                            headers={"Retry-After": str(math.ceil(wait))})
    with span("guard.input") as s:
        verdict = check_input(question)
        s.set(blocked=verdict["blocked"])
    if verdict["blocked"]:
        if trace is not None:
            trace.update(blocked_reason=verdict["reason"])
        raise HTTPException(400, {"guardrail": "input", "reason": verdict["reason"]})


def warnings_for(filings: list[str] | None) -> list[str]:
    return [NO_COMPANY_WARNING] if filings is None else []


# --- endpoints ------------------------------------------------------------------
# Plain "def" endpoints run in FastAPI's thread pool, so a slow LLM call does not
# block the server's event loop. The thread inherits the request's trace context.

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    guard_request(request, "query", req.question)
    pipeline: QAPipeline = state["pipeline"]
    cache: AnswerCache = state["cache"]
    trace = current_trace()
    start = time.perf_counter()

    with span("cache.lookup") as s:
        hit = cache.get("query", req.question, pipeline.version) if req.use_cache else None
        s.set(hit=hit is not None, enabled=req.use_cache)
    if hit is None:
        try:
            body = pipeline.answer(req.question)
        except DailyLimitReached as exc:
            raise HTTPException(503, f"LLM quota exhausted, try later: {exc}") from exc
        except GenerationFailed as exc:
            raise HTTPException(502, f"LLM call failed: {exc}") from exc
        body["filings_searched"] = pipeline.filings_for(req.question)
        with span("cache.store"):
            cache.set("query", req.question, pipeline.version, body)
    else:
        body = hit

    checks = body["output_checks"]
    with span("guard.output_summary") as s:
        guardrails = Guardrails(
            warnings=warnings_for(body["filings_searched"]),
            output=OutputChecks(**checks),
            disclaimer=DISCLAIMER if needs_disclaimer(req.question) else None,
        )
        s.set(grounded=checks["grounded"], low_confidence=checks["low_confidence"],
              warnings=len(guardrails.warnings), disclaimer=guardrails.disclaimer is not None)
    if trace is not None:
        trace.update(cache_hit=hit is not None, answer_preview=body["answer"],
                     not_found=body["not_found"], grounded=checks["grounded"],
                     low_confidence=checks["low_confidence"], pipeline_version=pipeline.version,
                     cache_key=make_key("query", req.question, pipeline.version))
    return QueryResponse(
        question=req.question, cached=hit is not None, pipeline_version=pipeline.version,
        latency_ms=round((time.perf_counter() - start) * 1000), guardrails=guardrails,
        trace_id=trace.trace_id if trace else None,
        **{k: v for k, v in body.items() if k not in ("tokens", "output_checks")},
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
    guard_request(request, "retrieve", req.question)
    pipeline: QAPipeline = state["pipeline"]
    cache: AnswerCache = state["cache"]
    trace = current_trace()
    params = f"k={req.top_k}|rerank={req.rerank}"
    start = time.perf_counter()

    with span("cache.lookup") as s:
        hit = cache.get("retrieve", req.question, pipeline.version, params) if req.use_cache else None
        s.set(hit=hit is not None, enabled=req.use_cache)
    if hit is None:
        body = {"chunks": pipeline.retrieve(req.question, req.top_k, req.rerank),
                "filings_searched": pipeline.filings_for(req.question)}
        with span("cache.store"):
            cache.set("retrieve", req.question, pipeline.version, body, params)
    else:
        body = hit

    if trace is not None:
        trace.update(cache_hit=hit is not None, pipeline_version=pipeline.version,
                     cache_key=make_key("retrieve", req.question, pipeline.version, params))
    return RetrieveResponse(
        question=req.question, cached=hit is not None, pipeline_version=pipeline.version,
        latency_ms=round((time.perf_counter() - start) * 1000),
        guardrails=Guardrails(warnings=warnings_for(body["filings_searched"])),
        trace_id=trace.trace_id if trace else None, **body,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline: QAPipeline = state["pipeline"]
    return HealthResponse(
        status="ok", chunks_indexed=pipeline.chunks_indexed, pipeline_version=pipeline.version,
        prompt_version=PROMPT_VERSION, rate_limits_per_minute=state["limiter"].limits,
        cache_path=str(CACHE_PATH), cache=state["cache"].stats(), trace_path=str(TRACE_PATH),
        embedding_cache=pipeline.embedding_cache_stats(), prices_usd=PRICES,
    )


@app.get("/documents")
def documents(company: str | None = None) -> dict:
    """
    What the API can answer about: every indexed filing grouped by company.
    ?company= filters by name (case and spacing ignored), e.g. "american express".
    """
    companies = state["pipeline"].documents()
    if company:
        wanted = normalize_name(company)
        companies = [c for c in companies if wanted in c["key"]]
    return {"companies": len(companies),
            "filings": sum(len(c["filings"]) for c in companies),
            "items": companies}


@app.delete("/cache")
def clear_cache() -> dict:
    return {"removed": state["cache"].clear()}


@app.get("/traces")
def list_traces(limit: int = 20) -> list[dict]:
    """Latest traced requests, newest first."""
    return state["traces"].recent(min(max(limit, 1), 200))


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    """One request step by step: every span with its duration and attributes."""
    trace = state["traces"].get(trace_id)
    if trace is None:
        raise HTTPException(404, f"No trace {trace_id!r} (unknown, or past the retention period)")
    return trace


@app.get("/stats")
def stats(hours: float = 24) -> dict:
    """Aggregates over the last `hours`: volume, errors, cache, guardrails, tokens, latency."""
    return state["traces"].stats(since_hours=hours)
