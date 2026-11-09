"""Query-embedding cache, the document catalogue and cost estimates."""
import numpy as np

from source.API.pipeline import catalogue
from source.RETRIEVAL.retriever import EmbeddingCache, Retriever
from source.TRACING.costs import estimate_cost


def test_embedding_cache_is_lru():
    cache = EmbeddingCache(max_size=2)
    a, b, c = (np.full((1, 2), v, dtype=np.float32) for v in (1, 2, 3))
    cache.put("a", a)
    cache.put("b", b)
    assert cache.get("a") is a          # "a" becomes most recent
    cache.put("c", c)                   # evicts "b", the least recently used
    assert cache.get("b") is None and cache.get("c") is c
    assert cache.stats() == {"entries": 2, "hits": 2, "misses": 1}


class _CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        return np.ones((len(texts), 4), dtype=np.float32)


class _FakeIndex:
    def search(self, vec, k, params=None):
        return np.zeros((1, k)), np.arange(k).reshape(1, k)


def test_dense_search_embeds_each_text_once():
    retriever = Retriever.__new__(Retriever)          # skip loading models and indexes
    retriever.embedder = _CountingEmbedder()
    retriever.embedding_cache = EmbeddingCache()
    retriever.query_prefix = "Instruct: x\nQuery:"
    retriever.faiss_index = _FakeIndex()
    assert retriever._dense_search("Nike revenue FY2023", 3) == [0, 1, 2]
    assert retriever._dense_search("Nike revenue FY2023", 5) == [0, 1, 2, 3, 4]
    assert retriever.embedder.calls == 1                # second search reused the vector
    retriever._dense_search("Nike revenue FY2022", 3)
    assert retriever.embedder.calls == 2                # different text, new embedding


def test_catalogue_groups_and_normalizes_filings():
    items = catalogue({
        "MCDONALDS_8K_dated-2023-01-06": [0], "MCDONALDS_2022_10K": [1, 2],
        "JOHNSON_JOHNSON_2022Q4_EARNINGS": [3], "ACTIVISIONBLIZZARD_2019_10K": [4],
        "ACTIVSIONBLIZZARD_2023Q2_10Q": [5],           # filename typo in the dataset
        "AMCOR_2023Q4_EARNINGS-DATED-2023-07-27": [6],
    })
    by_key = {c["key"]: c for c in items}
    assert set(by_key) == {"mcdonalds", "johnsonjohnson", "activisionblizzard", "amcor"}
    assert len(by_key["activisionblizzard"]["filings"]) == 2
    mcd = by_key["mcdonalds"]["filings"]
    assert [(f["type"], f["fiscal_year"], f["filed"], f["chunks"]) for f in mcd] == \
        [("10K", 2022, None, 2), ("8K", 2023, "2023-01-06", 1)]
    assert by_key["johnsonjohnson"]["company"] == "JOHNSON JOHNSON"
    assert by_key["johnsonjohnson"]["filings"][0]["quarter"] == "Q4"
    assert by_key["amcor"]["filings"][0]["type"] == "EARNINGS"


def test_cost_estimate():
    prices = {"llm_input_per_million_tokens": 1.0, "llm_output_per_million_tokens": 2.0,
              "rerank_per_call": 0.01}
    assert estimate_cost({"input_tokens": 1_000_000, "output_tokens": 500_000, "rerank_calls": 2},
                         prices) == 2.02
    assert estimate_cost({}, prices) == 0               # cache hit / blocked: no API calls
