# HANDOFF — read this first in a new chat

**Purpose:** everything a new assistant session needs to continue this project without the
previous conversation. Last updated: 2026-10-15. Keep it current at the end of each session.

> New session: read this file, then `documentation/experiment_log/README.md` (results
> history) and `documentation/LIMITATIONS.md`. Then ask the user what to do next — the
> plan is in **§8**.

---

## 1. Project in one paragraph
**FilingLens** — a RAG question-answering system over SEC filings, evaluated on the
**FinanceBench** benchmark: 150 analyst questions over 368 filings (10-K, 10-Q, 8-K,
earnings releases) of 40 US companies, 2015–2023. Data: `data/RAW/` (PDFs),
`data/EVAL/financebench_open_source.jsonl` (questions/answers/evidence),
`data/PROCESSED/` (chunks, indexes, results — gitignored). Windows PC, CPU only, Python
venv `.venv`. Heavy embedding runs are done by the user on **Google Colab (T4 GPU)**.

**Status: accuracy work is closed at 119/150 = 79.3% end-to-end (2026-10-15).** The user
does not want more accuracy experiments; remaining work is engineering (§8). Afterwards
the user plans a separate project: an AI agent that uses this pipeline as a tool.

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
  cannot authenticate to GitHub.
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

## 8. ▶ NEXT STEPS — engineering (accuracy work is closed)
Order agreed with the user; each step on its own branch + PR:
1. ✅ **Docs** (`docs/v3-results`): experiment log, LIMITATIONS.md, this file.
2. **API** (`feat/api`): FastAPI — `POST /query` → answer, citations, retrieved chunks,
   NOT FOUND flag; `GET /health`. Load Qwen, indexes, Cohere and Cerebras clients once at
   startup. Use the final pipeline (§2) with `embedding="qwen_headed"`.
3. **Guardrails**: input (off-topic, prompt injection, no company identified) and output
   (answer numbers present in retrieved text, NOT FOUND/low-confidence flag, "not
   investment advice" note). `source/GUARDRAILS/` is empty.
4. **Tracing**: per-request latency per step, tokens, cost, retrieved chunk ids
   (Langfuse or JSON logs). `source/TRACING/` is empty.
5. **Deployment**: Dockerfile + docker-compose; mount `data/PROCESSED/` (~2 GB) rather
   than baking it in; secrets via env.
6. **Tests + CI**: unit tests (metadata filter, RRF, header detection, formula check),
   a small end-to-end smoke test, GitHub Actions.
7. **Cleanup**: the root and folder READMEs are still Week-0 templates (mention
   `01-DATA/`, pgvector, `ingest.py`, Langfuse) — rewrite to match reality.

## 9. Environment gotchas
- Background processes started by the assistant die when the Claude Code session ends.
- The assistant's scratchpad (temp dir) is wiped between sessions — keep durable code in
  the repo (`tools/`), not the scratchpad.
- Qwen query embedding on CPU ≈ 1.7 s/question.
- Colab: tell the user to paste/run cells directly rather than through Gemini (it has
  rewritten code before). `colab_embed_qwen.py` checks token lengths on a sample only —
  tokenizing all chunks at once crashed Colab's RAM.
