"""
Persistent answer cache in a single SQLite file (data/cache.db by default).

Why: every uncached question costs ~1.7 s of CPU embedding, one Cohere rerank call
(trial key: 10/min, 1,000/month) and one or two Cerebras calls (5/min, 150/hour).
An agent calling the API often repeats a question, so a hit returns in ~1 ms.

Keys are exact, never "semantic": "revenue FY2021" and "revenue FY2022" look almost
identical but have different answers. A key combines
    kind      -- which endpoint ("query" or "retrieve")
    pipeline  -- a version string naming the model, prompt, index, reranker and k;
                 change any of them and old entries simply stop matching
    params    -- endpoint options that change the result (e.g. top_k)
    question  -- normalized: whitespace collapsed, lowercased
and is stored as a SHA-256 hash. The readable question and pipeline are kept in
their own columns so the file can be inspected with any SQLite viewer.

The interface is just get / set / stats / clear, so a Redis-backed class with the
same methods can replace this one when several API processes need a shared cache.
"""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    question   TEXT NOT NULL,
    pipeline   TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
)
"""


def normalize(question: str) -> str:
    return " ".join(question.split()).lower()


def make_key(kind: str, question: str, pipeline: str, params: str = "") -> str:
    raw = "\x1f".join((kind, pipeline, params, normalize(question)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the API's worker threads, serialized by a lock.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
        self._db.execute(SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()
        self.path = path

    def get(self, kind: str, question: str, pipeline: str, params: str = "") -> dict | None:
        key = make_key(kind, question, pipeline, params)
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            self._db.commit()
        return json.loads(row[0])

    def set(self, kind: str, question: str, pipeline: str, value: dict, params: str = "") -> None:
        key = make_key(kind, question, pipeline, params)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, kind, question, pipeline, value, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, kind, normalize(question), pipeline, json.dumps(value, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind, COUNT(*), COALESCE(SUM(hits), 0) FROM cache GROUP BY kind").fetchall()
        return {kind: {"entries": n, "hits": hits} for kind, n, hits in rows}

    def clear(self) -> int:
        with self._lock:
            removed = self._db.execute("DELETE FROM cache").rowcount
            self._db.commit()
        return removed

    def close(self) -> None:
        with self._lock:
            self._db.close()
