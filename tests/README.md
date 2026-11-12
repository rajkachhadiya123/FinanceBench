# tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -rx
```

156 tests in about 16 seconds (incl. `tests/MCP`). They need **no API keys, no network and no `data/PROCESSED/`**:
external services are replaced by fakes and every database is a temporary file. Only
`data/EVAL/` (the 150 benchmark questions, tracked in git) is read. GitHub Actions runs the
same command on every push and pull request (`.github/workflows/tests.yml`).

## UNIT/

| File | Covers |
|---|---|
| `test_metadata_filter.py` | Company names, short names, abbreviations, finance jargon, renamed companies, multi-year questions, fallbacks |
| `test_pipeline_helpers.py` | Reciprocal Rank Fusion, chunk-header splitting, public chunk shape, NOT FOUND detection |
| `test_guards.py` | **None of the 150 benchmark questions is blocked**; attacks are blocked; tricky legitimate questions pass; figure grounding; confidence threshold; disclaimer |
| `test_cache_and_rate_limit.py` | Cache keys and normalization, round trip, no expiry; rate limiter per client and window |
| `test_tracer.py` | Spans across worker threads, error recording, retention purge, stats, a failed save never breaking a request |
| `test_answer_parsing.py` | Parsing of the LLM reply, context labels, formula check |
| `test_chunk_headers.py` | Heading detection for items, statements, notes, parent-company schedules; table-of-contents pages ignored |
| `test_api_extras.py` | Query-embedding LRU cache, document catalogue, cost estimate |

## INTEGRATION/

| File | Covers |
|---|---|
| `test_api.py` | The real FastAPI app with a fake pipeline: cache hit, `use_cache=false`, HTTP 400 / 422 / 429 / 502 / 503, `X-Request-ID`, traces, `/stats`, `/documents`, cost per request, `DELETE /cache` |

## Expected failure

`test_var_does_not_resolve_to_activision` is marked `xfail(strict=True)`: it documents a known
metadata-filter bug ("VaR" in a JPMorgan question matches Activision). If the bug is fixed the
test starts passing, pytest reports it, and the marker should be removed.
