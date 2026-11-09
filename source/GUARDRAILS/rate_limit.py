"""
Per-client rate limiting: a sliding 60-second window kept in memory.

Protects the shared quotas behind the API -- Cerebras (5 requests/min, 150/hour)
and the Cohere trial key (10/min, 1,000/month) -- from one client or a looping
agent. Limits come from the environment:

    FILINGLENS_QUERY_PER_MIN     /query requests per client per minute (default 10)
    FILINGLENS_RETRIEVE_PER_MIN  /retrieve requests per client per minute (default 30)

Cached answers count too: the limiter runs before the cache lookup and is about
request volume, not cost. In-memory state suits the single API process; several
workers would need a shared store (e.g. Redis) instead.
"""
import os
import threading
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60.0
LIMITS = {
    "query": int(os.getenv("FILINGLENS_QUERY_PER_MIN", "10")),
    "retrieve": int(os.getenv("FILINGLENS_RETRIEVE_PER_MIN", "30")),
}


class RateLimiter:
    def __init__(self, limits: dict[str, int] = LIMITS, window: float = WINDOW_SECONDS):
        self.limits = limits
        self.window = window
        self._hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str, group: str) -> float:
        """Record a request; return 0 if allowed, else seconds until the next slot frees."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(client, group)]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limits[group]:
                return self.window - (now - hits[0])
            hits.append(now)
            return 0.0
