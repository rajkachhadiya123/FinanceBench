"""
Estimated cost of a request, from the tokens and API calls its trace recorded.

Prices are ESTIMATES for planning, not billing. The defaults are approximate public
list prices at the time of writing -- check your provider's pricing page and set
the real values in .env. Trial keys (as used during development) cost nothing; set
the prices to 0 to report that instead.

    FILINGLENS_PRICE_LLM_INPUT_PER_M     USD per 1M input tokens  (DeepSeek deepseek-flash)
    FILINGLENS_PRICE_LLM_OUTPUT_PER_M    USD per 1M output tokens
    FILINGLENS_PRICE_RERANK_PER_CALL     USD per Cohere rerank call (one search)
"""
import os


def _price(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


PRICES = {
    # deepseek-flash peak list prices (cache miss), 2026-11; off-peak is half.
    # Was 0.35 / 0.75 for Cerebras gpt-oss-120b.
    "llm_input_per_million_tokens": _price("FILINGLENS_PRICE_LLM_INPUT_PER_M", 0.30),
    "llm_output_per_million_tokens": _price("FILINGLENS_PRICE_LLM_OUTPUT_PER_M", 1.20),
    "rerank_per_call": _price("FILINGLENS_PRICE_RERANK_PER_CALL", 0.002),  # ~$2 per 1,000 searches
}


def estimate_cost(fields: dict, prices: dict = PRICES) -> float:
    """USD estimate for one request; 0 for cache hits and blocked requests (no API calls)."""
    cost = (fields.get("input_tokens") or 0) / 1e6 * prices["llm_input_per_million_tokens"]
    cost += (fields.get("output_tokens") or 0) / 1e6 * prices["llm_output_per_million_tokens"]
    cost += (fields.get("rerank_calls") or 0) * prices["rerank_per_call"]
    return round(cost, 6)
