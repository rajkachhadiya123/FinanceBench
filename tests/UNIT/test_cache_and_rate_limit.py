"""SQLite answer cache and the per-client rate limiter."""
import time

from source.GUARDRAILS.rate_limit import RateLimiter
from source.SHARED.cache import AnswerCache, make_key, normalize


def test_normalize_and_key():
    assert normalize("  What WAS  3M   revenue? ") == "what was 3m revenue?"
    assert make_key("query", "What was 3M revenue?", "v1") == make_key("query", " what was 3m REVENUE? ", "v1")
    assert make_key("query", "q?", "v1") != make_key("query", "q?", "v2")
    assert make_key("query", "q?", "v1") != make_key("retrieve", "q?", "v1")
    assert make_key("retrieve", "q?", "v1", "k=5") != make_key("retrieve", "q?", "v1", "k=10")


def test_cache_round_trip_and_stats(tmp_path):
    cache = AnswerCache(tmp_path / "cache.db")
    assert cache.get("query", "What was 3M revenue?", "v1") is None
    cache.set("query", "What was 3M revenue?", "v1", {"answer": "$32.8B"})
    assert cache.get("query", "  what was 3m revenue? ", "v1") == {"answer": "$32.8B"}
    assert cache.get("query", "What was 3M revenue?", "v2") is None      # other pipeline version
    assert cache.stats() == {"query": {"entries": 1, "hits": 1}}
    assert cache.clear() == 1 and cache.stats() == {}
    cache.close()


def test_cache_survives_reopen(tmp_path):
    path = tmp_path / "cache.db"
    cache = AnswerCache(path)
    cache.set("query", "q?", "v1", {"answer": "a"})
    cache.close()
    assert AnswerCache(path).get("query", "q?", "v1") == {"answer": "a"}  # no expiry


def test_rate_limit_per_client_and_group():
    limiter = RateLimiter({"query": 2, "retrieve": 5})
    assert limiter.check("1.1.1.1", "query") == 0
    assert limiter.check("1.1.1.1", "query") == 0
    assert limiter.check("1.1.1.1", "query") > 0          # third within the window
    assert limiter.check("2.2.2.2", "query") == 0         # other client unaffected
    assert limiter.check("1.1.1.1", "retrieve") == 0      # other group unaffected


def test_rate_limit_window_slides():
    limiter = RateLimiter({"query": 1, "retrieve": 1}, window=0.2)
    assert limiter.check("c", "query") == 0
    assert 0 < limiter.check("c", "query") <= 0.2
    time.sleep(0.25)
    assert limiter.check("c", "query") == 0
