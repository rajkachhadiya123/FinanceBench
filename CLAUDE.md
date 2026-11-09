# FilingLens — notes for Claude

RAG question answering over SEC filings, evaluated on FinanceBench (150 questions, 368 filings).
**Status: complete** — 119/150 = 79.3% end-to-end, FastAPI service with cache, guardrails,
tracing, 80 tests + CI, Docker. Full history: `HANDOFF.md` (human notes, read on demand),
`documentation/experiment_log/README.md` (every measurement), `documentation/LIMITATIONS.md`.

## Where things are
| Path | What |
|---|---|
| `source/API/app.py`, `pipeline.py` | FastAPI endpoints, middleware; the final pipeline loaded once (`QAPipeline`) |
| `source/RETRIEVAL/` | `metadata_filter.py` (company/year → filings), `retriever.py` (FAISS + BM25 + RRF + rerank, `EMBEDDINGS`, embedding LRU), `cohere_reranker.py` |
| `source/GENERATION/answer_generator.py` | Cerebras gpt-oss-120b; `PROMPT_TEMPLATE` = v2 (active), v1/v4 kept for reference |
| `source/GUARDRAILS/` | `guards.py` (input block / output signals), `rate_limit.py` |
| `source/TRACING/` | `tracer.py` (traces + spans, `traces.db`, 30-day retention), `costs.py` |
| `source/SHARED/cache.py` | SQLite answer cache (version-aware keys, no expiry) |
| `source/INGESTION/` | `chunker.py`, `embedder*.py`, `bm25_builder.py` (offline index builders) |
| `tools/` | Evaluation (`test_answers.py`, `evaluate_reranker.py`, `compare_answer_runs.py`, `evaluate_guardrails.py`) and data prep (`add_chunk_headers.py`, `colab_embed_qwen.py`) |
| `tests/UNIT`, `tests/INTEGRATION` | pytest; API tests use a fake pipeline |
| `data/EVAL/` (in git) · `data/RAW/` PDFs · `data/PROCESSED/` indexes, caches, results (gitignored, ~2 GB) | see `data/README.md` |
| `Dockerfile`, `docker-compose.yml`, `deployment/README.md` | Local Docker hosting |
| `graphify-out/` | Knowledge graph of the code + docs (see the graphify section below) |

Final pipeline: metadata filter → Qwen3-Embedding-0.6B on **headed** chunks (`embedding="qwen_headed"`;
`DEFAULT_EMBEDDING` in retriever.py is still `"qwen"`) + BM25 → RRF → Cohere `rerank-v4.0-pro` → top 5 →
Cerebras → on NOT FOUND retry with top 10 of the same reranked list.

## Commands (run from the repo root; Windows, venv `.venv`, Python 3.10)
```
.venv\Scripts\python.exe -m pytest -rx                        # 80 passed + 1 xfail (known VaR bug), no keys needed
.venv\Scripts\python.exe -m uvicorn source.API.app:app --port 8000   # API, docs at /docs
docker compose up --build -d                                  # Docker (docker.exe lives in %LOCALAPPDATA%\Programs\DockerDesktop\resources\bin)
.venv\Scripts\python.exe tools/test_answers.py --include-holdout --n 200 --top-k 5 --retry-top-k 10 --no-targeted --embedding qwen_headed
```

## Rules (from the user — follow them)
- **Ask before anything that spends API quota** (Cerebras: 5 req/min, 150/hour; Cohere trial: 10/min, 1,000/month)
  and before changing a script the user is about to run.
- **Accuracy work is closed at 79.3%.** Don't propose accuracy experiments unless asked; keep prompt v2 /
  reasoning "medium" (v3 and v4+high were measured and rejected).
- **No overfitting:** no rules for specific questions or companies; report holdout separately.
- **Git is the user's job:** commit locally on a feature branch, never to `main`; don't remind about push/PR/merge.
- **Never print or commit API keys** (`.env`, gitignored). Don't edit `.env` with PowerShell `Set-Content` (adds a BOM).
- Measure, don't guess: quote only measured numbers; the old `tools/evaluate_retrieval.py` matcher is lenient.
- Keep answers short; the user likes tables and diagrams. Keep durable code in the repo, not the scratchpad.

## graphify

This project has a knowledge graph at `graphify-out/` (code + docs; `data/` excluded via `.graphifyignore`).
- For questions about the codebase (where is X, what calls Y, how does Z flow), read
  `graphify-out/GRAPH_REPORT.md` or run `graphify query "<question>"` before reading files one by one.
- After changing code, refresh it with `/graphify . --update` (code-only changes are cheap: AST, no LLM).
