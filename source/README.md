# source

Application code, one folder per pipeline stage. The API in `API/` wires them together.

```
INGESTION → RETRIEVAL → GENERATION        (the question-answering pipeline)
GUARDRAILS · TRACING · SHARED             (around every API request)
API                                       (FastAPI service)
```

## INGESTION/ — PDFs to indexes (offline)

| File | Role |
|---|---|
| `chunker.py` | PDF text (PyMuPDF) → fixed 2,000-character chunks with 300 overlap, page-tracked → `chunks.jsonl` |
| `embedder.py` | MiniLM embeddings + `faiss.index`; also writes `chunk_meta.jsonl` (byte offsets into `chunks.jsonl`) |
| `embedder2.py` | BGE-base constants and index builder (retired) |
| `embedder3.py` | **Qwen3-Embedding-0.6B** constants (incl. the query instruction) and index builder; `--headed` builds the index for the headed chunks |
| `bm25_builder.py` | BM25 keyword index (bm25s, English stopwords + Snowball stemmer); `--headed` for the headed chunks |

Contract: position N means the same chunk in every index and metadata file.

## RETRIEVAL/

| File | Role |
|---|---|
| `metadata_filter.py` | Company and fiscal year(s) from the question text → the filings to search (or none = search all) |
| `retriever.py` | Hybrid search: dense (FAISS) + BM25, Reciprocal Rank Fusion, optional rerank; query-embedding LRU cache; per-embedding settings in `EMBEDDINGS` (`qwen_headed` is the one the API uses) |
| `cohere_reranker.py` | Cohere Rerank API client (handles the trial key's per-minute limit) |

## GENERATION/

| File | Role |
|---|---|
| `answer_generator.py` | Cerebras `gpt-oss-120b` over HTTP; `PROMPT_TEMPLATE` (v2, active), v1 and v4 kept for reference; parses REASONING / FINAL ANSWER / CITATIONS; waits out rate limits |

## GUARDRAILS/

| File | Role |
|---|---|
| `guards.py` | `check_input` (blocks hidden characters, unreadable text, repeated characters/words, prompt-injection patterns), `check_output` (share of figures traceable to the sources, retrieval confidence), investment-advice disclaimer |
| `rate_limit.py` | Per-client sliding-window rate limiter |

## TRACING/

| File | Role |
|---|---|
| `tracer.py` | `start_trace()` / `span()` and `TraceStore` (SQLite `traces.db`, 30-day retention, stats) |
| `costs.py` | Estimated USD per request from tokens and rerank calls (prices via env) |

## SHARED/

| File | Role |
|---|---|
| `cache.py` | SQLite answer cache with exact, version-aware keys |

## API/

| File | Role |
|---|---|
| `pipeline.py` | `QAPipeline`: the final pipeline loaded once (filter → hybrid search → rerank → LLM → top-10 retry on NOT FOUND → output checks); document catalogue |
| `app.py` | FastAPI app: endpoints, guardrail order, cache, tracing middleware |

Run the API from the repository root: `python -m uvicorn source.API.app:app --port 8000`.
