# HANDOFF — full project notes

**Purpose:** the complete working record of this project — decisions, results, what was
built and why — for the owner, and for Claude when it needs detail. Last updated:
2026-11-08. Keep it current at the end of each session.

> A new Claude chat automatically reads **`CLAUDE.md`** (short: map, commands, rules) and
> opens this file only when it needs the details. How a new chat finds things: **§10**.

---

## 1. Project in one paragraph
**FilingLens** — a RAG question-answering system over SEC filings, evaluated on the
**FinanceBench** benchmark: 150 analyst questions over 368 filings (10-K, 10-Q, 8-K,
earnings releases) of 40 US companies, 2015–2023. Data: `data/RAW/` (PDFs),
`data/EVAL/financebench_open_source.jsonl` (questions/answers/evidence),
`data/PROCESSED/` (chunks, indexes, results — gitignored). Windows PC, CPU only, Python
venv `.venv`. Heavy embedding runs are done by the user on **Google Colab (T4 GPU)**.

**Status: complete (2026-11-08).** Accuracy closed at 119/150 = 79.3% end-to-end
(2026-10-15); the engineering steps in §8 are done — API, caches, guardrails, tracing,
tests + CI, docs, Docker (running locally on port 8000). Next: a separate project, an AI
agent that uses this API as a tool. Optional leftovers are listed at the end of §8.

## 2. Final pipeline (what runs today)
```
Question
 → Metadata filter (company + fiscal year from question text → ~3 filings)
 → Qwen3-Embedding-0.6B dense search on HEADED chunks (faiss_qwen_headed.index) top 20
   + BM25 on headed chunks (bm25_index_headed/) top 20      ← side by side
 → RRF merge (k=60) → up to 40 candidates
 → Cohere rerank-v4.0-pro (hosted API) → top 5
 → Cerebras gpt-oss-120b, prompt v2, reasoning "medium", 3,000 completion tokens
 → if answer is "NOT FOUND": retry once with top 10 (same reranked list)
```
- Headed chunk = original chunk prefixed with
  `[COMPANY | TYPE | PERIOD | page N | statement/note]` (`tools/add_chunk_headers.py`).
- In code: `Retriever(use_reranker="cohere", embedding="qwen_headed")` +
  `search(..., metadata_filter=True, rerank=True)`. **`DEFAULT_EMBEDDING` is still
  `"qwen"` (original chunks)** — pass `embedding="qwen_headed"` explicitly, or change the
  default when building the API.
- End-to-end command (reuses cached answers):
  `python tools/test_answers.py --include-holdout --n 200 --top-k 5 --retry-top-k 10 --no-targeted --embedding qwen_headed`

## 3. Results
| | Correct /150 | Notes |
|---|---|---|
| Qwen + rerank, original chunks, prompt v2 | 117 (78.0%) | stitched from 3 partial runs, regraded |
| **Qwen headed + rerank, prompt v2** | **119 (79.3%)** | one clean run; +10/−8 vs above, p = 0.81 |
| Retrieval only (71 no-LLM questions, answer chunk in top 5) | 60/71 (85%) | holdout 23/28 |

Rejected after measurement: BGE as default (n.s.), local cross-encoders (too slow on CPU),
prompt v3 (0/16 fixed), prompt v4 + reasoning "high" (0/18 calculation failures fixed,
5 empty answers). Full details and failure analysis in the experiment log.

## 4. Key files
| File | Role |
|---|---|
| `source/INGESTION/chunker.py` | Fixed 2000-char / 300-overlap chunks → `chunks.jsonl` (107,049 chunks) |
| `tools/add_chunk_headers.py` | Adds context headers → `chunks_headed.jsonl` (same chunks, same order) |
| `source/INGESTION/embedder.py` / `embedder2.py` / `embedder3.py` | MiniLM / BGE / Qwen constants + index builders. `embedder3.py [--headed]` builds `faiss_qwen[_headed].index`; `--headed` also writes `chunk_meta_headed.jsonl` (byte offsets into the headed file). Holds Qwen `QUERY_PROMPT` |
| `source/INGESTION/bm25_builder.py [--headed]` | BM25 index (`bm25_index/` or `bm25_index_headed/`) |
| `source/RETRIEVAL/retriever.py` | Hybrid retriever; `EMBEDDINGS` dict (minilm, qwen, qwen_headed; bge commented out) with per-entry chunks/meta/BM25 paths; filter, RRF, rerank hook |
| `source/RETRIEVAL/metadata_filter.py` | Company/year extraction from question text |
| `source/RETRIEVAL/cohere_reranker.py` | Cohere Rerank API client (handles the 10 calls/min trial limit) |
| `source/GENERATION/answer_generator.py` | Cerebras client. `PROMPT_TEMPLATE` = v2 (active); `PROMPT_TEMPLATE_V4`, `PROMPT_TEMPLATE_V1` kept for reference; `check_formula()` (diagnostic for v4) |
| `tools/test_answers.py` | End-to-end runner; `--embedding`, `--ids`; caches answers in `data/PROCESSED/answer_results.jsonl` keyed by model/retrieval/prompt/reasoning effort |
| `tools/evaluate_reranker.py [--embedding X]` | Retrieval-only eval on the 71 no-LLM questions; strict matching; header-stripped grading; McNemar vs `qwen` |
| `tools/compare_answer_runs.py [--pending]` | End-to-end comparison of two runs with manual verdicts (`manual_verdicts.json`) |
| `tools/colab_embed_qwen.py` | Colab embedding script (`INPUT_NAME` selects the chunk zip) |
| `tools/evaluate_retrieval.py` | Older evaluator — its default matcher is the lenient/buggy one; don't quote its numbers |

**`data/PROCESSED/` contents (cleaned 2026-10-15, ~2.0 GB):** `faiss.index` (MiniLM),
`faiss_bge.index`, `faiss_qwen.index`, `faiss_qwen_headed.index`, `chunks[_headed].jsonl`,
`chunk_meta[_headed].jsonl`, `bm25_index[_headed]/`, `answer_results.jsonl`,
`manual_verdicts.json`, `rerank_cache_*.json`, `retrieval_ranks_*.json`. Raw `*_vectors.npy`
files were deleted after verifying the FAISS indexes hold identical vectors
(`index.reconstruct_n` recovers them); `embedder3.py` rebuilds would need them re-created.
`chunks*.zip` (Colab uploads) were deleted too — re-zip the `.jsonl` if needed.

## 5. Evaluation rules (important)
- **Strict answer matching**: years ("FY2022"), "COVID-19" and unit-scaled ratios are never
  answer matches (the old matcher inflated results, e.g. BGE "74%" was really 56%).
- **Question split**: 71 "no-LLM" questions (answer literally in evidence) = 43 tune + 28
  holdout (seed 42, stratified) via `evaluate_reranker.no_llm_split`; 79 need calculation.
- **Grading**: numeric answers auto-graded; others judged by hand with fixed rules
  (CORRECT / PARTIAL / WRONG / NOT FOUND), stored in `manual_verdicts.json` keyed
  `"<run>|<financebench_id>"` and pinned to an answer-text hash.
- **Significance**: paired McNemar; with these sample sizes, gains < ~5–8 questions are noise.

## 6. User preferences / decisions (respect these)
- **No overfitting**: no rules for specific questions/companies; report holdout separately;
  leave ambiguous-gold questions alone.
- **Ask before changing scripts the user is about to run**, and before starting runs that
  spend API quota — the user likes to control when runs start/stop.
- Keep explanations short; the user likes tables and diagrams.
- **Git**: never commit to `main`. One branch per step (e.g. `docs/v3-results`,
  `feat/api`), commit there; the user pushes (`git push origin <branch>`) and merges via a
  GitHub pull request (use "Create a merge commit", not squash). The assistant's shell
  cannot authenticate to GitHub. **Since 2026-11-04 the user handles git entirely: don't
  remind about pushing, PRs or merge order — focus on building and testing locally.**
- ⚠️ `Desktop\FinanceBenchh` (two h's) is an unrelated `git init` copy with a nested repo —
  pulling there fails with "unrelated histories". Only `Desktop\FinanceBench` is the real
  repo. `C:\Users\Admin` also contains an accidental empty `.git`.

## 7. API keys & limits (keys live in `.env`, gitignored — never print/commit them)
| Service | Env var | Use | Limits |
|---|---|---|---|
| Cerebras | `CEREBRAS_API_KEY` | LLM `gpt-oss-120b` | 5 req/min, 150/hour, 2,400/day; 30K tokens/min, 1M/day |
| Cohere (trial) | `COHERE_API_KEY` | `rerank-v4.0-pro` | 10 calls/min, 1,000/month (~700 used in Oct 2026, estimated) |
| Groq | `GROQ_API_KEY` (code commented out) | retired | — |
- A full 150 run ≈ 45–60 min, ~0.6M Cerebras tokens, ~150–165 Cohere calls.
- PowerShell writes `.env` with a UTF-8 BOM if edited with `Set-Content` — edit via Python.

## 8. Engineering steps (all done) and what is left
Built in this order after accuracy work closed; each step on its own branch:
1. ✅ **Docs** (`docs/v3-results`): experiment log, LIMITATIONS.md, this file.
2. ✅ **API** (`feat/api`): `source/API/app.py` (FastAPI) + `source/API/pipeline.py`
   (final pipeline, loaded once, calls serialized by a lock) + `source/SHARED/cache.py`
   (SQLite answer cache, `data/PROCESSED/cache.db`, exact-match keys incl. pipeline
   version). Endpoints: `POST /query`, `POST /retrieve` (no LLM; `rerank` optional),
   `GET /health`, `DELETE /cache`. Run: `python -m uvicorn source.API.app:app --port 8000`,
   docs at `/docs`. Tested: 3M FY2018 capex → $1,577M (correct) in 3.7 s, 3 ms cached.
   The fallback slices one reranked top-10 list, so it costs no second Cohere call.
   `requirements.txt` added (pinned). No Redis / Supabase: single process, static
   indexes — swap the cache class for Redis only when running several workers.
3. ✅ **Guardrails** (`feat/guardrails`), rule-based, no LLM — `source/GUARDRAILS/`:
   `guards.py` (input: hidden chars, unreadable text, repeated chars/words, narrow
   prompt-injection regexes → HTTP 400; output signals only, answer never changed:
   figure grounding vs retrieved text, Cohere `retrieval_confidence` / `low_confidence`
   < 0.50, investment-advice disclaimer; no-company warning) and `rate_limit.py`
   (per-client sliding window, 10 /query + 30 /retrieve per min, env-configurable → 429).
   Retriever results now carry `rerank_score`. Validated by `tools/evaluate_guardrails.py`:
   0/150 benchmark questions blocked, 14/14 attacks blocked, 10/10 tricky legit questions
   passed; grounding flag 0/111 false alarms (threshold 0.35, page refs excluded) but it
   also flags 0/19 wrong answers — wrong answers use real figures from the wrong row, so
   it only catches invented numbers; confidence: every answered benchmark question ≥ 0.816,
   out-of-scope questions 0.28–0.39. Observation: for "is X a good investment" the LLM
   still answers Yes/No — the disclaimer is attached, the prompt was not changed.
4. ✅ **Tracing** (`feat/tracing`): `source/TRACING/tracer.py` — `start_trace()` / `span()`
   (context variable, works across FastAPI worker threads, no-op outside a request),
   `TraceStore` in a separate `data/PROCESSED/traces.db` (traces + spans; metadata only,
   200-char answer preview + cache key; **30-day retention**, `FILINGLENS_TRACE_RETENTION_DAYS`;
   cached answers keep no expiry; a failed save never breaks a request). API middleware
   traces `/query` and `/retrieve`, returns `X-Request-ID` + `trace_id`; spans for guards,
   cache, filter/embed/dense/bm25/rrf, `rerank.cohere`, `llm.answer`, `llm.retry_top10`,
   provider rate-limit waits. Endpoints `GET /traces`, `GET /traces/{id}`, `GET /stats`.
   Live-tested on 9 requests; finding: Qwen query embedding on CPU (~1.0 s p50) is the
   slowest step, more than Cohere (0.55 s) or the LLM (0.53 s). `cost_usd` is not
   configured (no prices set) — tokens and call counts are recorded.
5. ✅ **Tests + CI** (`feat/tests`): `pytest` — 74 tests + 1 xfail (the known VaR →
   Activision filter bug, marked strict so a fix is noticed), ~11 s, no API keys, no
   network, no `data/PROCESSED`. `tests/UNIT/` (metadata filter, RRF/helpers, guards incl.
   "0 of 150 benchmark questions blocked", cache, rate limiter, tracer, answer parsing,
   chunk headers) and `tests/INTEGRATION/test_api.py` (FastAPI with a fake pipeline:
   cache hit, 400/422/429/502/503, traces, stats). Run: `python -m pytest -rx`.
   `.github/workflows/tests.yml` runs it on every push/PR (Python 3.10, CPU-only torch).
   `requirements-dev.txt` holds pytest.
5b. ✅ **API extras** (`feat/api-extras`): in-memory LRU **query-embedding cache** in the
   retriever (`EmbeddingCache`, 1,024 entries; live 1,422 ms → 0 ms for a repeated
   question); **`GET /documents`** (41 companies / 368 filings, type, year, quarter, 8-K
   filing date, `?company=` filter); **cost estimates** (`source/TRACING/costs.py`,
   `cost_usd` per trace and in `/stats`; prices via `FILINGLENS_PRICE_*` env vars —
   defaults are list-price estimates, trial keys are free). `/health` shows embedding-cache
   stats and prices. Tests: 80 passed + 1 xfail.
6. ✅ **Cleanup + setup guide**: every README rewritten to match the code (root README:
   results, pipeline, quick start, API, config); `data/README.md` is the setup guide (where
   to get the PDFs, index build order incl. the Colab step); `.env.example` added; the
   placeholder `workflows/` and `meta/` folders removed; `.gitignore` comment fixed.
7. ✅ **Docker** (`Dockerfile`, `docker-compose.yml`, guide in `deployment/README.md`):
   python:3.10-slim + CPU torch (`torch==2.14.0+cpu` from the PyTorch index with PyPI as
   extra index — PyTorch-only index broke on a torch dependency) + Qwen model baked in
   (chown in the same layer, image ~4.2 GB) + `source/`. `data/PROCESSED` mounted read-only;
   cache/traces in volume `filinglens-state`; keys from `.env`. Docker Desktop is a per-user
   install (`%LOCALAPPDATA%\Programs\DockerDesktop`). Tested: healthy in ~20 s, /query
   2.3 s, cached 3 ms, injection 400, cache survives restart, ~1 GB RAM.
8. ✅ **Context setup for new chats**: project `CLAUDE.md` (auto-loaded map, commands,
   rules) and a graphify knowledge graph in `graphify-out/` (code + docs, `data/` excluded
   by `.graphifyignore`) — see §10. `documentation/LIMITATIONS.md` "not yet built" section
   corrected (L-13..L-16 now describe the remaining engineering limits).

**Optional leftovers** (not needed for "done"):
- Evaluation harness: config-driven runs (`experiments/*.yaml`), run registry, standard
  report, LLM judge validated against `manual_verdicts.json`, smoke set for CI.
- VaR → Activision metadata-filter bug (strict xfail test documents it).
- Back up `data/PROCESSED/answer_results.jsonl` and `manual_verdicts.json` (gitignored,
  hold the graded 79.3% results, not regenerable without re-running and re-grading).
- Optional prompt rule so "is X a good investment" is not answered Yes/No (user's call).
- A self-correcting answer loop was discussed and **skipped by the user** (2026-11-07);
  if revisited, prototype it on the 31 failures + ~30 correct questions before building.

## 9. Environment gotchas
- Background processes started by the assistant die when the Claude Code session ends.
- The assistant's scratchpad (temp dir) is wiped between sessions — keep durable code in
  the repo (`tools/`), not the scratchpad.
- Qwen query embedding on CPU ≈ 1.7 s/question.
- Colab: tell the user to paste/run cells directly rather than through Gemini (it has
  rewritten code before). `colab_embed_qwen.py` checks token lengths on a sample only —
  tokenizing all chunks at once crashed Colab's RAM.
- Docker Desktop is a per-user install: `docker` is on the saved user PATH, but a terminal
  (or VS Code) opened before the install doesn't see it — restart VS Code, or add
  `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin` to `$env:Path` for that window.
- graphify runs from its own uv tool environment (`uv tool install graphifyy`; CLI at
  `%APPDATA%\uv\tools\graphifyy\Scripts\graphify.exe`), not the project `.venv`.

## 10. How a new chat finds things
| Source | Loaded | Holds | Size (approx.) |
|---|---|---|---|
| `CLAUDE.md` (repo root) | **Automatically, every chat** | Project map, commands, the user's rules, how to use the graph | ~1–1.3K tokens |
| Claude memory (`~/.claude/projects/<project>/memory/`, index `MEMORY.md`) | Index automatically; files when relevant | Cross-session preferences: git is the user's job; project complete | ~0.1K index (+~0.6K files) |
| `HANDOFF.md` (this file) | On demand | Full decisions, results, engineering log | ~4–5K tokens |
| `graphify-out/GRAPH_REPORT.md` | On demand | Graph summary: communities, hub nodes, surprising links | ~3.6–4.5K tokens |
| `graphify-out/graph.json` via `graphify query "…"` | On demand | Answers "where is X / what calls Y" from the graph | a few hundred tokens per query |
| `graphify-out/graph.html` | For people | Interactive map of the code and docs — open in a browser | — |

Refresh the graph after changes: `/graphify . --update` (code-only changes need no LLM;
changed docs need one small extraction pass). `graphify-out/cost.json` records tokens spent.
