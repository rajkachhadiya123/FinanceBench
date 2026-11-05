"""
Request tracing for FilingLens: one trace per API request, made of timed spans.

    trace a1b2c3   POST /query   200   3,712 ms   cache: miss
     ├─ guard.input          1 ms
     ├─ retrieval.embed  1,690 ms
     ├─ rerank.cohere      880 ms   best=0.91
     └─ llm.cerebras     1,080 ms   tokens 3,912 / 287

Usage
-----
    store = TraceStore(Path("data/PROCESSED/traces.db"))     # once, at startup
    store.purge_expired()                                    # drop traces past retention

    with start_trace("/query", store) as trace:              # once per request
        trace.update(question=q)
        with span("rerank.cohere", docs=40) as s:            # anywhere below, any depth
            scores = reranker.score(...)
            s.set(best=max(scores))
        trace.update(status_code=200, cache_hit=False)

`span()` finds the active trace through a context variable, so deep code (retriever,
generator) can add spans without a trace being passed around; with no active trace it
does nothing, so scripts and tests that call the pipeline directly are unaffected.

Storage and lifetime
--------------------
Traces live in their own SQLite file, separate from the answer cache, because their
lifetimes differ: a cached answer has no expiry (it is removed only by DELETE /cache or
a pipeline-version change), while traces are kept TRACE_RETENTION_DAYS days
(FILINGLENS_TRACE_RETENTION_DAYS, default 30) and deleted by purge_expired().

A trace holds metadata only -- timings, tokens, guardrail results, a short answer
preview and the cache key of the full answer -- never chunk texts or full reasoning,
which the cache already stores. Roughly 1-2 KB per request.

Saving a trace never raises: if the database cannot be written, the request still
succeeds and a warning is logged. Tracing must never break an answer.
"""
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("filinglens.tracing")

TRACE_RETENTION_DAYS = int(os.getenv("FILINGLENS_TRACE_RETENTION_DAYS", "30"))
ANSWER_PREVIEW_CHARS = 200

# Trace-level fields that update() accepts; each is a column in the traces table.
TRACE_FIELDS = (
    "status_code", "cache_hit", "question", "answer_preview", "not_found", "error",
    "blocked_reason", "low_confidence", "grounded", "input_tokens", "output_tokens",
    "llm_calls", "rerank_calls", "cost_usd", "pipeline_version", "cache_key", "client",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id         TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,      -- ISO 8601, UTC
    started_epoch    REAL NOT NULL,      -- for retention and time-window queries
    endpoint         TEXT NOT NULL,
    duration_ms      REAL,
    status_code      INTEGER,
    cache_hit        INTEGER,
    question         TEXT,
    answer_preview   TEXT,
    not_found        INTEGER,
    error            TEXT,
    blocked_reason   TEXT,
    low_confidence   INTEGER,
    grounded         INTEGER,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    llm_calls        INTEGER,
    rerank_calls     INTEGER,
    cost_usd         REAL,
    pipeline_version TEXT,
    cache_key        TEXT,
    client           TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id     TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,       -- order in which spans started
    name         TEXT NOT NULL,          -- e.g. "rerank.cohere"
    offset_ms    REAL NOT NULL,          -- start, relative to the trace start
    duration_ms  REAL NOT NULL,
    attributes   TEXT,                   -- JSON: hits, scores, tokens, ...
    error        TEXT,
    PRIMARY KEY (trace_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_epoch);
CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name);
"""


# --- in-memory records ------------------------------------------------------------
@dataclass
class Span:
    name: str
    offset_ms: float
    attributes: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None

    def set(self, **attributes) -> None:
        """Attach facts learned during the step (e.g. hits, best score, tokens)."""
        self.attributes.update(attributes)


@dataclass
class Trace:
    endpoint: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fields: dict = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    duration_ms: float = 0.0
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **values) -> None:
        unknown = set(values) - set(TRACE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown trace field(s): {sorted(unknown)}")
        if "answer_preview" in values and values["answer_preview"]:
            values["answer_preview"] = str(values["answer_preview"])[:ANSWER_PREVIEW_CHARS]
        self.fields.update(values)

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Accumulate across several LLM calls (e.g. the NOT FOUND retry)."""
        with self._lock:
            self.fields["input_tokens"] = self.fields.get("input_tokens", 0) + input_tokens
            self.fields["output_tokens"] = self.fields.get("output_tokens", 0) + output_tokens
            self.fields["llm_calls"] = self.fields.get("llm_calls", 0) + 1

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000


_current: ContextVar[Trace | None] = ContextVar("filinglens_trace", default=None)


def current_trace() -> Trace | None:
    return _current.get()


# --- context managers --------------------------------------------------------------
@contextmanager
def start_trace(endpoint: str, store: "TraceStore | None" = None):
    """Open a trace for one request; it is saved to `store` when the block ends."""
    trace = Trace(endpoint=endpoint)
    token = _current.set(trace)
    try:
        yield trace
    except Exception as exc:
        trace.fields.setdefault("error", f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        trace.duration_ms = trace.elapsed_ms()
        _current.reset(token)
        if store is not None:
            store.save(trace)


class _NoSpan:
    """Stand-in when no trace is active: set() is accepted and ignored."""

    def set(self, **_attributes) -> None:
        pass


@contextmanager
def span(name: str, **attributes):
    """Time a step of the active trace. A no-op when no trace is active."""
    trace = _current.get()
    if trace is None:
        yield _NoSpan()
        return
    record = Span(name=name, offset_ms=trace.elapsed_ms(), attributes=dict(attributes))
    with trace._lock:
        trace.spans.append(record)
    start = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        record.duration_ms = (time.perf_counter() - start) * 1000


# --- storage -----------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; SQLite has no built-in percentile function."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered) + 0.5 - 1e-9))
    return round(ordered[min(rank, len(ordered)) - 1], 1)


def _to_db(value):
    return int(value) if isinstance(value, bool) else value


class TraceStore:
    def __init__(self, path: Path, retention_days: int = TRACE_RETENTION_DAYS):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()
        self.path = path
        self.retention_days = retention_days

    def save(self, trace: Trace) -> bool:
        """Write a finished trace. Never raises: returns False and logs on failure."""
        columns = ["trace_id", "started_at", "started_epoch", "endpoint", "duration_ms",
                   *trace.fields]
        values = [trace.trace_id, trace.started_at.isoformat(timespec="milliseconds"),
                  trace.started_at.timestamp(), trace.endpoint, round(trace.duration_ms, 1),
                  *(_to_db(v) for v in trace.fields.values())]
        spans = [(trace.trace_id, seq, s.name, round(s.offset_ms, 1), round(s.duration_ms, 1),
                  json.dumps(s.attributes, default=str) if s.attributes else None, s.error)
                 for seq, s in enumerate(trace.spans)]
        try:
            with self._lock:
                self._db.execute(
                    f"INSERT OR REPLACE INTO traces ({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' * len(columns))})", values)
                self._db.executemany("INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?)", spans)
                self._db.commit()
            return True
        except Exception as exc:  # tracing must never break a request
            log.warning("Could not save trace %s: %s", trace.trace_id, exc)
            return False

    def purge_expired(self) -> int:
        """Delete traces (and their spans) older than the retention period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).timestamp()
        with self._lock:
            removed = self._db.execute("DELETE FROM traces WHERE started_epoch < ?", (cutoff,)).rowcount
            self._db.commit()
        return removed

    def get(self, trace_id: str) -> dict | None:
        """One trace with all of its spans, in start order."""
        with self._lock:
            self._db.row_factory = sqlite3.Row
            try:
                row = self._db.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
                spans = self._db.execute(
                    "SELECT name, offset_ms, duration_ms, attributes, error FROM spans "
                    "WHERE trace_id = ? ORDER BY seq", (trace_id,)).fetchall()
            finally:
                self._db.row_factory = None
        if row is None:
            return None
        return {**dict(row), "spans": [
            {**dict(s), "attributes": json.loads(s["attributes"]) if s["attributes"] else {}}
            for s in spans]}

    def recent(self, limit: int = 20) -> list[dict]:
        """Newest traces first, summary columns only."""
        with self._lock:
            rows = self._db.execute(
                "SELECT trace_id, started_at, endpoint, status_code, duration_ms, cache_hit, "
                "not_found, blocked_reason, question FROM traces "
                "ORDER BY started_epoch DESC LIMIT ?", (limit,)).fetchall()
        keys = ("trace_id", "started_at", "endpoint", "status_code", "duration_ms", "cache_hit",
                "not_found", "blocked_reason", "question")
        return [dict(zip(keys, r)) for r in rows]

    def stats(self, since_hours: float = 24) -> dict:
        """Aggregates over a time window: volume, errors, cache, guardrails, cost, latency."""
        since = time.time() - since_hours * 3600
        with self._lock:
            rows = self._db.execute(
                "SELECT endpoint, status_code, duration_ms, cache_hit, not_found, "
                "blocked_reason, low_confidence, grounded, input_tokens, output_tokens, "
                "llm_calls, rerank_calls, cost_usd FROM traces WHERE started_epoch >= ?",
                (since,)).fetchall()
            span_rows = self._db.execute(
                "SELECT s.name, s.duration_ms FROM spans s JOIN traces t USING (trace_id) "
                "WHERE t.started_epoch >= ?", (since,)).fetchall()

        def count(pred):
            return sum(1 for r in rows if pred(r))

        by_endpoint: dict[str, list[float]] = {}
        for r in rows:
            by_endpoint.setdefault(r[0], []).append(r[2] or 0)
        by_span: dict[str, list[float]] = {}
        for name, ms in span_rows:
            by_span.setdefault(name, []).append(ms)
        answered = [r for r in rows if r[1] == 200 and r[0] == "/query"]

        return {
            "window_hours": since_hours,
            "requests": len(rows),
            "status_codes": {str(code): count(lambda r, c=code: r[1] == c)
                             for code in sorted({r[1] for r in rows if r[1] is not None})},
            "errors_5xx": count(lambda r: (r[1] or 0) >= 500),
            "blocked_by_guardrails": count(lambda r: r[5] is not None),
            "cache_hit_rate": (round(count(lambda r: r[3] == 1) / len(answered), 3)
                               if answered else None),
            "query_answers": {
                "total": len(answered),
                "not_found": sum(1 for r in answered if r[4] == 1),
                "low_confidence": sum(1 for r in answered if r[6] == 1),
                "ungrounded": sum(1 for r in answered if r[7] == 0),
            },
            "usage": {
                "llm_calls": sum(r[10] or 0 for r in rows),
                "rerank_calls": sum(r[11] or 0 for r in rows),
                "input_tokens": sum(r[8] or 0 for r in rows),
                "output_tokens": sum(r[9] or 0 for r in rows),
                "cost_usd": round(sum(r[12] or 0 for r in rows), 4),
            },
            "latency_ms_by_endpoint": {
                ep: {"count": len(v), "p50": _percentile(v, 50), "p95": _percentile(v, 95),
                     "max": round(max(v), 1)}
                for ep, v in sorted(by_endpoint.items())},
            "latency_ms_by_step": {
                name: {"count": len(v), "p50": _percentile(v, 50), "p95": _percentile(v, 95),
                       "avg": round(sum(v) / len(v), 1)}
                for name, v in sorted(by_span.items())},
        }

    def close(self) -> None:
        with self._lock:
            self._db.close()
