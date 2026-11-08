# FilingLens

**Question answering over SEC filings, evaluated on the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark.**

Ask a question such as *"What was Nike's total revenue in FY2023?"* and FilingLens finds the
right filing among 368 documents (10-Ks, 10-Qs, 8-Ks, earnings releases of 40 US companies,
2015–2023), retrieves the relevant passages and answers with citations — through a FastAPI
service with caching, guardrails and request tracing.

[![tests](https://github.com/rajkachhadiya123/FinanceBench/actions/workflows/tests.yml/badge.svg)](https://github.com/rajkachhadiya123/FinanceBench/actions/workflows/tests.yml)

## Results

| Measure | Result |
|---|---|
| **End-to-end accuracy, all 150 FinanceBench questions** | **119 / 150 (79.3%)** — 3 partial, 20 wrong, 8 NOT FOUND |
| Answer passage in the top 5 (71 questions whose answer is written in the filing) | 60 / 71 (85%) — holdout 23 / 28 |
| Typical latency (uncached / cached) | ~2.5–4 s / ~5 ms |

Numeric answers are graded automatically; free-text answers were judged by hand against the
expected answer with fixed rules. How every number was measured, and what was tried and
rejected, is in the [experiment log](documentation/experiment_log/README.md); known weaknesses
are in [LIMITATIONS.md](documentation/LIMITATIONS.md).

## How it works

```
Question
 → guardrails: rate limit · input check (prompt injection, hidden characters, gibberish)
 → cache lookup (SQLite) ─────────────────────────────── hit → answer in ~5 ms
 → metadata filter: company + fiscal year from the question → ~3 of 368 filings
 → Qwen3-Embedding-0.6B dense search (top 20)  +  BM25 keyword search (top 20)
 → Reciprocal Rank Fusion → up to 40 candidates
 → Cohere rerank-v4.0-pro → top 5 chunks
 → Cerebras gpt-oss-120b → answer with reasoning and citations
     (if "NOT FOUND": retry once with the top 10 of the same reranked list)
 → output checks: figures traceable to the sources · retrieval confidence · disclaimer
 → response + trace id (every step timed and stored)
```

Each of the 107,049 chunks (2,000 characters, 300 overlap) carries a context header such as
`[VERIZON | 10-K | FY2022 | page 55 | Consolidated Balance Sheets]`, so search, reranker and
LLM can tell which statement a table comes from.

## Features

| Feature | Details |
|---|---|
| **REST API** | `POST /query`, `POST /retrieve` (no LLM), `GET /documents`, `GET /health` — interactive docs at `/docs` |
| **Answer cache** | SQLite, exact-match keys that include the pipeline version, no expiry |
| **Embedding cache** | In-memory LRU of query vectors (the slowest step on CPU: ~1 s → 0 ms when repeated) |
| **Guardrails** | Rule-based, no LLM. Block only clear abuse (HTTP 400); everything else is a signal next to the answer, which is never changed. None of the 150 benchmark questions is blocked |
| **Rate limiting** | Per client: 10 `/query` and 30 `/retrieve` per minute (HTTP 429) |
| **Tracing** | Every request gets an id (`X-Request-ID`) and a stored trace of timed steps, tokens and estimated cost; `GET /traces/{id}`, `GET /stats` |
| **Tests + CI** | 80 tests, no API keys or data needed; GitHub Actions on every push |

## Quick start

**Requirements:** Python 3.10, ~3 GB of disk, API keys for [Cerebras](https://cloud.cerebras.ai)
(LLM) and [Cohere](https://dashboard.cohere.com) (reranker), and the built indexes in
`data/PROCESSED/` — see [data/README.md](data/README.md) for how to get the PDFs and build them.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

copy .env.example .env            # then fill in CEREBRAS_API_KEY and COHERE_API_KEY

python -m uvicorn source.API.app:app --port 8000
```

Startup takes ~30–60 s (the embedding model and indexes load once). Then open
**http://127.0.0.1:8000/docs**, or:

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
     -d "{\"question\": \"What was Nike's total revenue in FY2023?\"}"
```

```json
{
  "answer": "Nike's total revenue in FY2023 was $51,217 million.",
  "not_found": false,
  "filings_searched": ["NIKE_2023_10K"],
  "chunks": [{"rank": 1, "header": "[NIKE | 10-K | FY2023 | page 54 | ...]", "rerank_score": 0.91, "...": "..."}],
  "guardrails": {"warnings": [], "output": {"grounded": true, "retrieval_confidence": 0.91, "low_confidence": false}, "disclaimer": null},
  "cached": false,
  "trace_id": "fd5e4e55a4b64c15"
}
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /query` | Question → answer, reasoning, citations, the chunks used, guardrail signals, trace id |
| `POST /retrieve` | Question → top chunks only, no LLM (`top_k` 1–20, `rerank` true/false) |
| `GET /documents` | Companies and filings the API can answer about (`?company=` filter) |
| `GET /health` | Status, index size, pipeline version, cache and embedding-cache stats, prices |
| `GET /traces`, `GET /traces/{id}` | Latest requests; one request step by step |
| `GET /stats?hours=24` | Volume, status codes, cache hit rate, guardrail counts, tokens, cost, p50/p95 per step |
| `DELETE /cache` | Empty the answer cache |

Errors: `400` blocked by an input guardrail · `422` invalid request · `429` rate limit
(with `Retry-After`) · `502` LLM call failed · `503` LLM quota exhausted.

## Configuration

Set in `.env` (see [.env.example](.env.example)); only the two API keys are required.

| Variable | Default | Purpose |
|---|---|---|
| `CEREBRAS_API_KEY`, `COHERE_API_KEY` | — | LLM and reranker |
| `CEREBRAS_MODEL`, `CEREBRAS_REASONING_EFFORT` | `gpt-oss-120b`, `medium` | Answering model |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-pro` | Reranker |
| `FILINGLENS_QUERY_PER_MIN`, `FILINGLENS_RETRIEVE_PER_MIN` | `10`, `30` | Rate limits per client |
| `FILINGLENS_CACHE_PATH`, `FILINGLENS_TRACE_PATH` | `data/PROCESSED/cache.db`, `traces.db` | Storage |
| `FILINGLENS_TRACE_RETENTION_DAYS` | `30` | Traces are deleted after this; cached answers never expire |
| `FILINGLENS_PRICE_LLM_INPUT_PER_M`, `…_OUTPUT_PER_M`, `FILINGLENS_PRICE_RERANK_PER_CALL` | list-price estimates | Cost estimates in traces (trial keys cost nothing) |

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -rx
```

80 tests in ~11 s, with no API keys, network or `data/PROCESSED` — external services are faked.
They include a check that none of the 150 benchmark questions is blocked by the guardrails.
One test is an expected failure documenting a known metadata-filter bug. See [tests/README.md](tests/README.md).

## Reproducing the evaluation

```bash
# end-to-end answers on all 150 questions (~45–60 min; ~165 Cerebras + ~150 Cohere calls; answers are cached)
python tools/test_answers.py --include-holdout --n 200 --top-k 5 --retry-top-k 10 --no-targeted --embedding qwen_headed

# retrieval only, 71 questions whose answer is written in the filing
python tools/evaluate_reranker.py --embedding qwen_headed
```

Details and the other evaluation scripts: [tools/README.md](tools/README.md).

## Project layout

```
source/
  INGESTION/    PDF → chunks, embedding and BM25 index builders
  RETRIEVAL/    metadata filter, hybrid retriever, Cohere reranker client
  GENERATION/   Cerebras answer generator and prompts
  GUARDRAILS/   input/output checks, rate limiter
  TRACING/      request traces, cost estimates
  SHARED/       SQLite answer cache
  API/          FastAPI app and the pipeline it serves
tools/          evaluation and data-preparation scripts
tests/          unit and API tests (pytest)
data/           EVAL (questions, in git) · RAW (PDFs) · PROCESSED (indexes, gitignored)
documentation/  experiment log, limitations, early retrieval reports
analysis/       early exploration scripts
experiments/    reserved for evaluation-harness configs (not built yet)
deployment/     reserved for Docker (not built yet)
```

## Documentation

| Document | Contents |
|---|---|
| [Experiment log](documentation/experiment_log/README.md) | Every version (V1 → V3), measurements, what was tried and rejected |
| [LIMITATIONS.md](documentation/LIMITATIONS.md) | Where the remaining 31 failures come from; what is not built |
| [HANDOFF.md](HANDOFF.md) | Working notes for continuing development |
| [data/README.md](data/README.md) | Data files and how to rebuild the indexes |

## Status

Accuracy work is closed at 79.3%. Not built yet: an evaluation harness (config-driven runs with
an automatic judge), Docker packaging, and an AI-agent layer on top of the API.
