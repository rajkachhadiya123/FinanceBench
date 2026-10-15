"""
Hosted reranking through Cohere's Rerank API.

A cross-encoder reads the question and each candidate chunk together, which
fixes rankings the first-stage searches get wrong. Running a strong one
locally (bge-reranker-v2-m3) would take ~1-1.5 min per question on our CPU;
the API answers in well under a second.

Free trial keys allow 10 rerank calls per minute and 1,000 per month (read
from the response headers). One question = one call, however many candidates.
"""
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.cohere.com/v2/rerank"
# Models available to this key (v1/models?endpoint=rerank): rerank-v4.0-pro,
# rerank-v4.0-fast (32K context), rerank-v3.5, rerank-english/multilingual-v3.0.
DEFAULT_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro")
RATE_LIMIT_WAIT = 61  # seconds; the trial limit is per minute


class CohereReranker:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise SystemExit("COHERE_API_KEY not set. Add it to .env")
        self.model = model or DEFAULT_MODEL
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document, in the input order."""
        for attempt in range(5):
            response = self._client.post(API_URL, json={
                "model": self.model, "query": query, "documents": documents,
            })
            if response.status_code == 429:  # per-minute trial limit
                print(f"    [Cohere rate limit; waiting {RATE_LIMIT_WAIT}s]", flush=True)
                time.sleep(RATE_LIMIT_WAIT)
                continue
            if response.status_code >= 500 and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            scores = [0.0] * len(documents)
            for item in response.json()["results"]:
                scores[item["index"]] = item["relevance_score"]
            return scores
        raise RuntimeError("Cohere rerank failed after retries")

    def close(self):
        self._client.close()
