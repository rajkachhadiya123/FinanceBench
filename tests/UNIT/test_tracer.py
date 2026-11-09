"""Request tracing: spans, context propagation, storage, retention, stats, failure safety."""
import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

from source.TRACING.tracer import TraceStore, current_trace, span, start_trace


@pytest.fixture
def store(tmp_path):
    s = TraceStore(tmp_path / "traces.db", retention_days=30)
    yield s
    try:
        s.close()
    except Exception:
        pass


def test_trace_with_spans_is_saved(store):
    with start_trace("/query", store) as trace:
        trace.update(question="q?", status_code=200, cache_hit=False)
        with span("rerank.cohere", docs=40) as s:
            s.set(best=0.91)
        trace.add_tokens(100, 10)
        trace.add_tokens(50, 5)          # e.g. the NOT FOUND retry
    saved = store.get(trace.trace_id)
    assert saved["status_code"] == 200 and saved["cache_hit"] == 0
    assert (saved["input_tokens"], saved["output_tokens"], saved["llm_calls"]) == (150, 15, 2)
    assert [s["name"] for s in saved["spans"]] == ["rerank.cohere"]
    assert saved["spans"][0]["attributes"] == {"docs": 40, "best": 0.91}


def test_span_reaches_trace_from_a_worker_thread(store):
    # FastAPI runs plain-def endpoints in a thread pool with a copy of the context.
    with start_trace("/query", store) as trace:
        def work():
            with span("retrieval.embed"):
                pass
        ctx = contextvars.copy_context()
        ThreadPoolExecutor(1).submit(ctx.run, work).result()
    assert [s["name"] for s in store.get(trace.trace_id)["spans"]] == ["retrieval.embed"]


def test_span_without_trace_is_a_no_op():
    assert current_trace() is None
    with span("anything", x=1) as s:
        s.set(y=2)                      # accepted and ignored


def test_errors_are_recorded_and_reraised(store):
    with pytest.raises(RuntimeError):
        with start_trace("/query", store) as trace:
            with span("llm.answer"):
                raise RuntimeError("HTTP 502")
    saved = store.get(trace.trace_id)
    assert "HTTP 502" in saved["error"] and "HTTP 502" in saved["spans"][0]["error"]


def test_answer_preview_is_truncated_and_fields_validated(store):
    with start_trace("/query", store) as trace:
        trace.update(answer_preview="x" * 1000)
        with pytest.raises(ValueError):
            trace.update(not_a_field=1)
    assert len(store.get(trace.trace_id)["answer_preview"]) == 200


def test_retention_purges_old_traces_and_their_spans(store):
    with start_trace("/query", store) as old:
        with span("step"):
            pass
    with start_trace("/query", store) as new:
        pass
    store._db.execute("UPDATE traces SET started_epoch = started_epoch - ? WHERE trace_id = ?",
                      (31 * 86400, old.trace_id))
    store._db.commit()
    assert store.purge_expired() == 1
    assert store.get(old.trace_id) is None and store.get(new.trace_id) is not None
    assert store._db.execute("SELECT COUNT(*) FROM spans WHERE trace_id = ?", (old.trace_id,)).fetchone()[0] == 0


def test_stats_and_recent(store):
    for status, cached in ((200, False), (200, True), (400, None)):
        with start_trace("/query", store) as t:
            t.update(status_code=status)
            if cached is not None:
                t.update(cache_hit=cached, not_found=False)
            if status == 400:
                t.update(blocked_reason="injection")
    stats = store.stats(since_hours=1)
    assert stats["requests"] == 3
    assert stats["status_codes"] == {"200": 2, "400": 1}
    assert stats["blocked_by_guardrails"] == 1
    assert stats["cache_hit_rate"] == 0.5
    assert [r["status_code"] for r in store.recent(10)] == [400, 200, 200]


def test_a_failed_save_never_breaks_the_request(store):
    store.close()
    with start_trace("/query", store) as trace:      # must not raise
        trace.update(status_code=200)
    assert store.save(trace) is False
