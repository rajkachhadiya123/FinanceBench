# HANDOFF — read this first in a new chat

**Purpose:** everything a new assistant session needs to continue this project without the
previous conversation. Last updated: 2026-10-12. Keep it current at the end of each session.

> New session: read this file, then `documentation/experiment_log/README.md` (detailed
> results history), then `documentation/LIMITATIONS.md`. Then ask the user what to do next —
> the immediate pending step is in **§8**.

---

## 1. Project in one paragraph
**FilingLens** — a RAG question-answering system over SEC filings, evaluated on the
**FinanceBench** benchmark: 150 analyst questions over 368 filings (10-K, 10-Q, 8-K,
earnings releases) of 40 US companies, 2015–2023. Data: `data/RAW/` (PDFs),
`data/EVAL/financebench_open_source.jsonl` (questions/answers/evidence),
`data/PROCESSED/` (chunks, indexes — gitignored). Windows PC, CPU only, Python venv `.venv`.
Heavy embedding runs are done by the user on **Google Colab (T4 GPU)**.

## 2. Current best pipeline (what runs today)
```
Question
 → Metadata filter (company + fiscal year from question text → ~3 filings)
 → Qwen3-Embedding-0.6B dense search (faiss_qwen.index) top 20
   + BM25 keyword search (bm25s) top 20            ← run side by side, not sequential
 → RRF merge (k=60) → up to 40 candidates
 → Cohere rerank-v4.0-pro (hosted API) → top 5
 → Cerebras gpt-oss-120b, prompt v2 → answer
 → if answer is "NOT FOUND": retry once with top 10 (same reranked list)  ← "Option A" fallback
```
- Default embedding in `source/RETRIEVAL/retriever.py`: `DEFAULT_EMBEDDING = "qwen"`
  (BGE entry commented out on user's request; MiniLM kept for comparison).
- Reranker is **off by default** in the Retriever; turned on via `Retriever(use_reranker="cohere")`
  + `search(..., rerank=True)`. `tools/test_answers.py` turns it on (`RERANKER = "cohere"`).

## 3. Results (strict scoring — see §6 for why "strict")
**Retrieval only** — 71 questions whose answer is written in the filing (answer-chunk in top k):

| Setup | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|
| V1 MiniLM + BM25 | 21 | 35 (49%) | 48 | 0.402 |
| V2 BGE-base + BM25 | 16 | 42 (59%) | 48 | 0.375 |
| V3 Qwen3-0.6B + BM25 | 22 | 43 (61%) | 51 | 0.431 |
| **V3 + Cohere rerank (top 40)** | **40** | **58 (82%)** | **61** | **0.674** |
Holdout (28 never-tuned questions) with reranker: **89% top 5**. Reranker gain p < 0.001.
Embedding model swaps were NOT significant; the reranker was.

**End-to-end, all 150 questions** (Cerebras gpt-oss-120b, top 5, reranker on):

| Version | Correct | Partial | Wrong | NOT FOUND |
|---|---|---|---|---|
| Prompt v1 | 106 (70.7%) | 6 | 20 | 18 |
| + prompt v2 (re-ran only the 26 wrong/partial) | 110 (73.3%) | 7 | 15 | 18 |
| **+ 10-chunk fallback (re-ran only the 18 NOT FOUND)** | **116 (77.3%)** | 7 | 16 | 11 |
⚠ 77.3% is a **combined** figure from three partial runs; a **clean full 150 run** with
prompt v2 + fallback has NOT been done yet (needed to detect regressions).

Prompt **v3** (v2 + "use consolidated statements / never assume zero / no metric
substitution") was tested on 16 wrong + 30 correct: **0 fixed, 1 correct lost → removed**.
Reason: chunks don't carry statement headings, so the LLM can't tell parent-company-only
tables from consolidated ones → this led to the **chunk headers** work (§8).

## 4. Analysis of the remaining 16 wrong answers (drives next steps)
| Cause | Count | Fix |
|---|---|---|
| A. Read wrong table (parent-company-only schedule, segment table, quarterly column) — Verizon quick ratio, AmEx largest liability, JPM liquidation, PepsiCo EBITDA, AMD D&A | 5 | **Chunk context headers** (in progress) |
| B. Wrong filing/period retrieved (Pfizer Upjohn, JnJ EPS guidance in prior-year Q4 release, Pfizer "this 10-K" with no year) | 3 | Metadata filter improvements |
| C. Metric convention (AES avg vs ending inventory, Netflix D&A scope, MGM EBITDAR used as EBIT) | 3 | Mostly leave (convention disputes) |
| D. Detail not retrieved in right filing (Ulta inventory drivers, Amcor acquisitions) | 2 | Chunk headers / chunking |
| E. Ambiguous question / debatable gold (Foot Locker CEO "similar", AmEx geographies, Best Buy "best category") | 3 | **Leave — fixing = overfitting** |
Remaining NOT FOUND (11): mostly multi-statement/multi-year calculations, two "there are
none" questions, and the JPM "VaR" question (known filter bug: "VaR" matched Activision).

## 5. Key files
| File | Role |
|---|---|
| `source/INGESTION/chunker.py` | Fixed 2000-char / 300-overlap chunks → `chunks.jsonl` (107,049 chunks) |
| `source/INGESTION/embedder.py` / `embedder2.py` / `embedder3.py` | MiniLM / BGE / **Qwen** constants + index builders. `embedder3.py` builds `faiss_qwen.index` from `qwen_vectors.npy` and holds `QUERY_PROMPT` (Qwen instruction, queries only) |
| `source/INGESTION/bm25_builder.py` | BM25 index (`bm25_index/`) |
| `source/RETRIEVAL/retriever.py` | Hybrid retriever, `EMBEDDINGS` dict, metadata filter, RRF, rerank hook |
| `source/RETRIEVAL/metadata_filter.py` | Company/year extraction from question text |
| `source/RETRIEVAL/cohere_reranker.py` | Cohere Rerank API client (handles 10 calls/min trial limit) |
| `source/GENERATION/answer_generator.py` | Cerebras gpt-oss-120b via HTTP; `PROMPT_TEMPLATE` (v2, active), `PROMPT_TEMPLATE_V1`; Groq code commented out |
| `tools/test_answers.py` | End-to-end runner: retrieval → rerank → LLM → grade; caches answers in `data/PROCESSED/answer_results.jsonl` tagged by model/retrieval/prompt |
| `tools/evaluate_reranker.py` | Retrieval-only eval on the 71 no-LLM questions; strict matching + split; caches Cohere scores |
| `tools/evaluate_retrieval.py` | Older evaluator (PDF reports; its default matcher is the lenient/buggy one) |
| `tools/add_chunk_headers.py` | **NEW** — adds context headers to every chunk → `chunks_headed.jsonl/.zip` |
| `tools/colab_embed_qwen.py` | Colab script: embeds a chunk zip with Qwen. `INPUT_NAME = "chunks_headed"` currently |
| `documentation/experiment_log/README.md` | Full V1→V3 history with workflows and tables |

## 6. Evaluation rules (important)
- **Strict answer matching**: the original matcher counted years in answers ("FY2022") as
  matches, parsed "COVID-19" as −19, and unit-scaled ratios (0.83 ≈ 833,033). This inflated
  results (e.g. BGE "74%" was really 56%). Strict matching is implemented in
  `tools/evaluate_reranker.py` and `tools/test_answers.py` (`answer_written_in_evidence`).
- **Question split**: 71 "no-LLM" questions (answer literally in evidence) = 43 tune + 28
  holdout (seed 42, stratified by type) — recreated by `evaluate_reranker.no_llm_split`.
  79 "LLM" questions need calculation/synthesis.
- **Grading**: numeric answers auto-graded; free-text/“WRONG” answers were **judged
  manually** by reading them against the expected answer (the auto-grader misses ~40% of
  correct free-text answers). An LLM judge was proposed but not built.
- **Significance**: paired McNemar test; with n=43–71, gains < ~5 questions are noise.

## 7. User preferences / decisions (respect these)
- **No overfitting**: no rules for specific questions/companies; judge changes on all 150 and
  report holdout separately. Ambiguous-gold questions (group E) are left alone.
- **Ask before changing scripts the user is about to run** (user rejected an unrequested edit
  to the Colab script). Keep explanations short; user likes tables and diagrams.
- Wants accuracy pushed up; agreed realistic target **80% → 85%** end-to-end.
- Chose: Qwen as default, BGE commented out; Cohere reranker; Cerebras gpt-oss-120b (Groq
  commented out); **Option A** fallback (top 10 on NOT FOUND, no recursive/agentic loops);
  prompt v2 (v3 removed); chunk-header approach **Option A = keep chunk boundaries, add header**.
- Git: work on branch `feat/metadata-filtering`; user pushes to GitHub themself
  (`git push origin feat/metadata-filtering` — the assistant's shell can't authenticate).

## 8. ▶ CURRENT STEP — chunk context headers (in progress)
**Done:** `tools/add_chunk_headers.py` produced `data/PROCESSED/chunks_headed.jsonl` and
`chunks_headed.zip` (53 MB): same 107,049 chunks, same order (verified vs `chunk_meta.jsonl`),
each prefixed with a header like:
```
[VERIZON | 10-K | FY2022 | page 55 | Consolidated Balance Sheets]
[AMERICANEXPRESS | 10-K | FY2022 | page 124 | Note 8 DEBT > Parent Company Only]
[VERIZON | 10-K | FY2022 | page 27 | Item 7 Management's Discussion and Analysis ...]
```
97% of chunks have a section label; header ≈ 17 tokens. Detector reads each filing top to
bottom, tracks current Item/section, carries it forward across pages, skips TOC pages and
in-sentence mentions; "(Parent Company only)" captions are sub-labels of the current note.

**Waiting on the user:** Colab run of `tools/colab_embed_qwen.py` on `chunks_headed.zip`.
The script saves shards to Drive (resumable) only when the zip is in **My Drive** (top level);
the user was given this cell to run:
```
from google.colab import drive; drive.mount('/content/drive')
!mv -n /content/chunks_headed.zip /content/drive/MyDrive/ && ls -lh /content/drive/MyDrive/chunks_headed.zip
%run /content/colab_embed_qwen.py
```
Expected output file: **`qwen_vectors_chunks_headed.npy`** (107049 × 1024, fp16, ~210 MB) →
user puts it in `data/PROCESSED/`.
Warning: Colab's Gemini previously rewrote code (an error mentioned a non-existent
`final_book_summary.zip`) — advise running the cell directly.

**Next steps after the vectors arrive** (not yet coded):
1. Validate vectors (shape, finite, norms ≈ 1; spot-check a few chunks re-embedded locally vs
   Colab, cosine ≈ 1) — same check done for the first Qwen run.
2. Build `faiss_qwen_headed.index` (adapt `embedder3.py` main: vectors path + index path).
3. Build a BM25 index over `chunks_headed.jsonl` into e.g. `bm25_index_headed/`
   (same tokenization as `bm25_builder.py`: stopwords "en" + English Snowball stemmer).
4. Add a `"qwen_headed"` entry to `EMBEDDINGS` in `retriever.py` — note the Retriever also
   hard-codes `CHUNKS_PATH`/`BM25_DIR`; the headed variant must read `chunks_headed.jsonl`
   and the headed BM25 dir (make these per-embedding settings). Keep old one for comparison.
5. Retrieval test: `tools/evaluate_reranker.py` on 71 questions, old vs headed (both with
   Cohere rerank). Then, if better, a **clean full 150 run** with prompt v2 + fallback:
   `python tools/test_answers.py --include-holdout --n 200 --top-k 5 --retry-top-k 10 --no-targeted`
   (tag the cache so headed-retrieval answers don't mix with old: `RETRIEVAL_LABEL`).
6. Update `documentation/experiment_log/README.md` and commit.

## 9. API keys & limits (keys live in `.env`, gitignored — never print/commit them)
| Service | Env var | Use | Limits seen |
|---|---|---|---|
| Cerebras | `CEREBRAS_API_KEY` | LLM `gpt-oss-120b` | 5 req/min, 150/hour, 2,400/day; 30K tokens/min, 1M/day |
| Cohere (trial) | `COHERE_API_KEY` | `rerank-v4.0-pro` | 10 calls/min, 1,000/month (~250 used so far) |
| Groq | `GROQ_API_KEY` (commented out ×2) | retired | 8K tokens/min, ~200K/day |
- A full 150 run ≈ 40–50 min (12 s spacing for the 5 req/min limit), ~0.6M Cerebras tokens,
  150 Cohere calls. Each question ≈ 4K tokens (prompt v2 ≈ 500 tokens of instructions).
- PowerShell writes `.env` with a UTF-8 BOM if edited with `Set-Content` — edit via Python.

## 10. Uncommitted work (as of this handoff)
Modified: `documentation/experiment_log/README.md`, `source/GENERATION/answer_generator.py`
(Cerebras + prompt v2), `source/RETRIEVAL/retriever.py` (Cohere hook), `tools/test_answers.py`
(Cerebras pacing, reranker on, `--ids`, prompt/retrieval cache tags), `tools/colab_embed_qwen.py`
(`INPUT_NAME`). New: `source/RETRIEVAL/cohere_reranker.py`, `tools/add_chunk_headers.py`,
`tools/evaluate_reranker.py`. Last commit: `6d0a99e` (Qwen default). Experiment log not yet
updated with the Cerebras/prompt-v2/fallback/v3/headers results.

## 11. Environment gotchas
- Background processes started by the assistant die when the Claude Code session ends
  (also via WMI). Windows Scheduled Tasks with `pythonw.exe` survive. Short runs: use the
  harness background mode and keep the session open.
- The assistant's scratchpad (temp dir) gets wiped between sessions — keep durable code in
  the repo (`tools/`), not the scratchpad.
- Qwen query embedding on CPU ≈ 1.7 s/question; reranking locally with bge-reranker-v2-m3
  would be ~1–1.5 min/question (why the hosted Cohere API is used).
- Colab: tell the user to paste/run cells directly rather than through Gemini.
