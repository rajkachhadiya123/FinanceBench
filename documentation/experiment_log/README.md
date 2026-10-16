# Experiment Log: Retrieval & Embedding Evolution

A running record of what we tried, what it measured, and what we learned.
**Update this file after every experiment** — append a new version section and
update the comparison table. Numbers here are measured, never estimated, unless
marked *(expected)*.

| Version | Embedding model | Status | Headline result |
|---|---|---|---|
| [V1](#v1--baseline-all-minilm-l6-v2) | `all-MiniLM-L6-v2` | Done | Baseline. 47% answer-in-top-5 (tune set) |
| [V2](#v2--bge-base-en-v15) | `BAAI/bge-base-en-v1.5` | Done (retired) | 53% answer-in-top-5 (tune set); +3 questions, not significant |
| [V3](#v3--qwen3-embedding-06b--reranking--chunk-headers) | `Qwen/Qwen3-Embedding-0.6B` + chunk headers | **Done, current default** | Qwen alone 61% top-5 (n.s. vs BGE). + Cohere rerank 82% (p < 0.001). + chunk headers 85%. **End-to-end: 119/150 = 79.3%** |

**Accuracy work closed on 2026-10-15 at 79.3% end-to-end.** Final pipeline: metadata
filter → Qwen3 dense (headed chunks) top 20 + BM25 top 20 → RRF → Cohere
`rerank-v4.0-pro` → top 5 → Cerebras `gpt-oss-120b` (prompt v2, reasoning medium) →
retry with top 10 on NOT FOUND. See [End-to-end results](#end-to-end-results-all-150-questions).

---

## How we measure (read this first)

All comparisons use the same rules, so versions are comparable:

- **Question split** (fixed, seed 42, stratified by question type)
  - **71 "no-LLM" questions** — the expected answer is literally written in the filing.
    Split into **43 tune** (used to compare methods) and **28 holdout** (touched only
    for the final number, to avoid overfitting).
  - **79 "LLM" questions** — the answer must be calculated or synthesised.
- **Retrieval metric — "answer in top-k"**: a retrieved chunk is from the correct
  filing **and** contains the expected answer. Reported at top 5 (what the LLM sees)
  and top 10/20.
- **Strict answer matching** (fixed in V2, see [Metric bug](#the-metric-bug-found-during-v2)):
  years are not counted as answer values; unit scaling only applies when the answer
  states a unit.
- **Significance**: paired McNemar test per question. With 43 questions, gains under
  ~5 questions are usually noise (p > 0.05).

---

## V1 — Baseline: `all-MiniLM-L6-v2`

### Technology
| Stage | Choice | Details |
|---|---|---|
| PDF parsing | PyMuPDF (`fitz`) | Plain text per page, 368 filings |
| Chunking | Fixed-size sliding window | 2,000 chars, 300 overlap → **107,049 chunks** |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 22M params, 384-dim, **256-token limit** |
| Vector index | FAISS `IndexFlatIP` | Exact cosine search |
| Keyword search | BM25 (`bm25s`) | English stopwords + Snowball stemmer |
| Fusion | Reciprocal Rank Fusion | k = 60, top 20 from each search |
| Metadata filter | Rule-based (company + fiscal year from question text) | Restricts search to ~3 filings on average |
| Reranker | Cross-encoder `ms-marco-MiniLM-L-6-v2` | Measured, **disabled** (see below) |

### Workflow
```mermaid
flowchart LR
    subgraph Indexing["Indexing (once)"]
        PDF[368 SEC PDFs] --> P[PyMuPDF text] --> C[Fixed 2000-char chunks<br/>107,049]
        C --> E1[MiniLM embed<br/>384-dim] --> F[(FAISS index)]
        C --> B1[BM25 index]
    end
    subgraph Query["Every question"]
        Q[Question] --> MF[Metadata filter<br/>company + year]
        MF --> D[MiniLM embed question<br/>→ FAISS top 20]
        MF --> K[BM25 top 20]
        D --> R[RRF merge]
        K --> R
        R --> T5[Top 5 chunks]
    end
    F -.-> D
    B1 -.-> K
```

### Results
| Measurement | Result |
|---|---|
| Metadata filter impact (old metric) | Wrong-document retrieval 61% → 18% — the biggest single gain so far |
| Answer in top 5 (43 tune, strict) | **20/43 (47%)** |
| Answer in top 10 / top 20 | 29/43 (67%) / 34/43 (79%) |
| Cross-encoder reranker (old metric) | +2.6pp but 60× slower → disabled |

### Flaws found
1. **Silent truncation** — MiniLM reads 256 tokens; chunks are ~450–500 tokens, so roughly
   the back half of every chunk never reached its vector.
2. **Weak general-purpose model** — small, older, not trained for question→passage retrieval.
3. **Chunks have no context** — a balance-sheet chunk doesn't say which company, year or
   statement it is.
4. **Tables cut mid-way** — year headers and numbers can land in different chunks.

---

## V2 — `BAAI/bge-base-en-v1.5`

### What changed and why
| | V1 | V2 |
|---|---|---|
| Model | all-MiniLM-L6-v2 | **bge-base-en-v1.5** |
| Parameters | 22M | 110M |
| Dimensions | 384 | 768 |
| Max input | 256 tokens | **512 tokens** (fixes most truncation) |
| Training | General sentence similarity | **Trained for question → passage retrieval** |
| Query handling | Plain | Query prefix `"Represent this sentence for searching relevant passages: "` |

Everything else (chunks, BM25, filter, RRF) unchanged, so the difference is the model alone.

**Embedding cost:** CPU ran at ~2 chunks/s (≈13+ hours for 107K chunks), so the index was
built on a **Google Colab T4 GPU** in ~20 minutes
([`tools/colab_embed_bge.ipynb`](../../tools/colab_embed_bge.ipynb) →
[`tools/build_bge_index.py`](../../tools/build_bge_index.py)) → `faiss_bge.index` (314 MB).

### Workflow
```mermaid
flowchart LR
    subgraph Indexing["Indexing (once, Colab GPU)"]
        C[Same 107,049 chunks] --> E2[BGE-base embed<br/>768-dim, 512 tokens] --> F2[(faiss_bge.index)]
        C --> B2[BM25 index - unchanged]
    end
    subgraph Query["Every question"]
        Q[Question] --> MF[Metadata filter]
        MF --> D2[prefix + question → BGE<br/>→ FAISS top 20]
        MF --> K2[BM25 top 20]
        D2 --> R[RRF merge] --> T5[Top 5] --> LLM[Groq gpt-oss-120b<br/>10-chunk fallback on NOT FOUND]
        K2 --> R
    end
    F2 -.-> D2
    B2 -.-> K2
```

### Results vs V1
| Measurement (strict metric) | V1 MiniLM | V2 BGE | Change |
|---|---|---|---|
| Answer in top 5 — 43 tune | 20/43 (47%) | **23/43 (53%)** | +3, p = 0.45 (not significant) |
| Answer in top 10 — 43 tune | 29/43 (67%) | 27/43 (63%) | −2 |
| Answer in top 5 — earlier 80-question set | 40/80 | 45/80 | +5, p = 0.27 (not significant) |
| Answer in top 5 / top 10 — all 71 no-LLM | — | 42/71 (59%) / 48/71 (68%) | — |
| Query time | ~0.03 s | ~0.06 s | still fast |

**Verdict:** BGE is directionally better and stays the default, but a 5× bigger model gave a
small, statistically unproven gain. **The embedding model was not the main bottleneck.**

### The metric bug (found during V2)
The first BGE comparison reported **74.2% vs 63.4% (p = 0.021)**. That was wrong. The old
matcher counted:
- years in the answer (`"in FY 2022"`) as the value to find — any chunk mentioning 2022 passed (22 false successes);
- `COVID-19` as the number −19;
- unit scaling on ratios — `0.83` matched `833,033`.

Fixed with strict matching. "No-LLM" questions shrank from 93 → **71**; true scores are the ones above.

### End-to-end with the LLM (V2 retrieval + Groq `openai/gpt-oss-120b`)
79 LLM questions, top 5 chunks, retry with 10 chunks on NOT FOUND:

| Outcome | Count |
|---|---|
| Correct | **35 (44%)** + 2 partly correct |
| Wrong | 10 (13%) |
| NOT FOUND (refused) | **30 (38%)** |
| Error (request > 8K tokens) | 2 |
| **Correct when the LLM answered** | **35/47 = 74%** |
| 10-chunk fallback | 33 retried → 9 answered → 5 correct |

### Other experiments during V2
| Experiment | Result | Decision |
|---|---|---|
| **Cross-encoder rerankers** (top 40 → rerank → top 5), 43 tune | MiniLM-CE: 26/43 (+3, p=0.51, 4.7 s/q)<br/>bge-reranker-base: 27/43 (+4, p=0.29, 22 s/q)<br/>mxbai / bge-reranker-v2-m3: not completed (too slow on CPU) | Small, not significant, too slow on CPU — revisit with GPU after V3 |
| **Vector database** — is FAISS limiting us? | FAISS = brute-force exact on 40/40 queries. Time: BGE query embed 77 ms (83%), BM25 15 ms, **FAISS 0.8 ms (1%)** | **No change needed.** Another vector DB cannot improve accuracy |
| **Where do missed answers rank?** (all 150) | see below | Led to V3 |

### Flaws found — where the right chunk actually ranks
| Rank of answer chunk / evidence page | No-LLM (71) | LLM (79) |
|---|---|---|
| 1–5 (LLM sees it) | 39 (55%) | 31 (39%) |
| 6–20 | 14 (20%) | 14 (18%) |
| 21–50 | 9 (13%) | 8 (10%) |
| 51–200 | 4 (6%) | **21 (27%)** |
| Not in top 200 | 5 (7%) | 5 (6%) |

Of the 80 misses: **35%** are ranked 6–20 (a reranker can fix), **44%** are ranked 51+ or
missing (only better retrieval can fix). Root causes:
1. **Chunks have no identity** — no company/year/statement label, so "balance sheet FY2018"
   has nothing to match against.
2. **Tables split from their year headers** by the fixed 2,000-char cut.
3. **Some chunks still exceed 512 tokens** (e.g. 635 seen) — the tail is truncated.
4. **LLM questions need 2–3 statements** (e.g. inventory + cost of sales); questions name the
   metric, not the line items.
5. Smaller issues: a filter bug (`"VaR"` matched Activision instead of JPM), 2 questions name no company.

---

## V3 — `Qwen3-Embedding-0.6B` + reranking + chunk headers

### Why this model
| | V2 BGE-base | V3 Qwen3-Embedding-0.6B |
|---|---|---|
| Parameters | 110M | 600M |
| Max input | 512 tokens | **32K tokens** — no truncation of any chunk or full page |
| Query handling | Fixed prefix | **Task instruction**, e.g. *"Given a financial question, retrieve the filing passage that answers it"* |
| Standing | Strong 2023 model | Among the top open models on MTEB retrieval at its size |
| Runs on | Colab T4 | Colab T4 |

Alternatives considered: `bge-m3` (8K, dense+sparse), `gte-large-en-v1.5`,
`voyage-finance-2` (finance-specific, paid API), Qwen3-Embedding-4B/8B (too heavy for a free GPU).
"Best" is benchmark-dependent; Qwen3-0.6B is the strongest *practical* choice for our hardware.

### Plan — test model and chunking together
V2 showed a model swap alone gives a small gain, so V3 tests **model × chunking** in one Colab
run on the 63 filings the tune questions use:

| | Original chunks | + Context header | Structure-aware chunks |
|---|---|---|---|
| **BGE-base (V2)** | 22/42 ✅ measured | *to test* | *to test* |
| **Qwen3-Embedding-0.6B** | *to test* | *to test* | *to test* |

- **Context header**: each chunk prefixed with `[COMPANY 10K FY2018 | page 57 | Consolidated Balance Sheet]`.
- **Structure-aware chunks**: never cross a page, split at paragraph breaks, ~1,400 chars, same header.
  (Table→Markdown conversion was tried and dropped — PyMuPDF fragmented tables and split words.)

### Workflow (as planned; structure-aware chunking was not built)
```mermaid
flowchart LR
    subgraph Indexing["Indexing (Colab GPU)"]
        PDF[PDFs] --> SC[Structure-aware chunks<br/>page + paragraph bounded]
        SC --> H[+ context header<br/>company, year, page, statement]
        H --> E3[Qwen3-Embedding-0.6B<br/>32K tokens] --> F3[(FAISS index)]
        H --> B3[BM25 index]
    end
    subgraph Query["Every question"]
        Q[Question] --> MF[Metadata filter]
        MF --> D3[instruction + question → Qwen3<br/>→ FAISS top 20]
        MF --> K3[BM25 top 20]
        D3 --> R[RRF merge] --> T5[Top 5] --> LLM[Groq LLM]
        K3 --> R
    end
    F3 -.-> D3
    B3 -.-> K3
```

### Success criteria
- Beat V2's **23/43** answer-in-top-5 on the tune set with p < 0.05, then confirm on the 28 holdout.
- Reduce LLM NOT FOUND answers below the current **30/79**.

### Results — step 1: Qwen3 on the original chunks (model alone)
Embedded on a Colab T4 ([`tools/colab_embed_qwen.py`](../../tools/colab_embed_qwen.py)) →
`qwen_vectors.npy` → [`source/INGESTION/embedder3.py`](../../source/INGESTION/embedder3.py) →
`faiss_qwen.index` (418 MB). Verified: local CPU embeddings of sampled chunks match the
Colab vectors (cosine ≈ 1.00, rows aligned). Selected with `Retriever(embedding="qwen")`.

Answer in top-k, 71 no-LLM questions, same pipeline (filter + dense + BM25 + RRF):

| Model | Tune 43 @5 | Holdout 28 @5 | **All 71 @5** | All 71 @10 | All 71 @20 | Query time |
|---|---|---|---|---|---|---|
| MiniLM (V1) | 20 (47%) | 15 (54%) | 35 (49%) | 48 (68%) | 56 (79%) | 0.21 s |
| BGE-base (V2) | 23 (53%) | 19 (68%) | 42 (59%) | 48 (68%) | 55 (77%) | 0.24 s |
| **Qwen3-0.6B (V3)** | **24 (56%)** | 19 (68%) | **43 (61%)** | **51 (72%)** | **58 (82%)** | 1.78 s (CPU) |

Significance (paired McNemar, all 71):
- Qwen vs BGE: @5 +5/−4 (p = 1.0), @10 +5/−2 (p = 0.45) — **not significant**.
- Qwen vs MiniLM: @5 +10/−2 (**p = 0.039**) — the only significant model gain so far, but it
  took two model upgrades to get there.

**Verdict:** Qwen3 is the best of the three by a small margin (mostly at top 10/20), and
~7× slower per query on CPU. As with V2, **the embedding model is not the bottleneck** —
three models spanning 22M → 600M parameters land within 8 questions of each other at top 5.
This supports testing chunking next (step 2 of V3).

### Results — step 2: Cohere reranker on top of Qwen hybrid search
Pipeline: filter → Qwen top 20 + BM25 top 20 → RRF (up to 40 candidates) →
**Cohere `rerank-v4.0-pro`** (hosted API, free trial key) → top k.
Code: [`source/RETRIEVAL/cohere_reranker.py`](../../source/RETRIEVAL/cohere_reranker.py),
`Retriever(use_reranker="cohere")` + `search(..., rerank=True)`;
measured with [`tools/evaluate_reranker.py`](../../tools/evaluate_reranker.py).

Answer in top k, 71 no-LLM questions:

| Configuration | @1 | @3 | **@5** | @10 | MRR |
|---|---|---|---|---|---|
| Qwen hybrid, no rerank | 22 (31%) | 34 (48%) | 43 (61%) | 51 (72%) | 0.433 |
| + rerank top 20 | 38 (54%) | 50 (70%) | 54 (76%) | 57 (80%) | 0.633 |
| **+ rerank top 40** | **40 (56%)** | **54 (76%)** | **58 (82%)** | **61 (86%)** | **0.674** |

| Split (rerank top 40) | @5 | MRR |
|---|---|---|
| Tune 43 | 33 (77%) — was 24 (56%) | 0.613 |
| Holdout 28 | **25 (89%)** — was 19 (68%) | 0.768 |

Significance vs no rerank (McNemar): top 40 @5 **+16 / −1, p < 0.001**; @1 +24 / −6, p = 0.001.
**The first large, significant gain since the metadata filter.** Holds on the untouched holdout.

By type @5 (no rerank → top 40): domain 17 → 24 / 27, metrics 9 → 12 / 13, novel 17 → 22 / 31.

Cost: ~0.9 s per question (API), one call per question. Trial key: 10 calls/min, 1,000/month.

**Why it works:** the first-stage searches score question and chunk separately; the
cross-encoder reads them together, so it fixes the "right chunk ranked 6–40" cases (the
35% of misses identified in V2). It cannot fix chunks never retrieved — remaining misses
are the target for chunking (step 3).

### Results — step 3: chunk context headers
Structure-aware re-chunking was not built. Instead every existing chunk got a one-line
header, with chunk boundaries, order and count unchanged
([`tools/add_chunk_headers.py`](../../tools/add_chunk_headers.py)):
```
[AMERICANEXPRESS | 10-K | FY2022 | page 124 | Note 8 DEBT > Parent Company Only]
[VERIZON | 10-K | FY2022 | page 55 | Consolidated Balance Sheets]
```
The section comes from a top-to-bottom scan of each filing that tracks the current Item,
statement, note or parent-company schedule, skips table-of-contents pages and in-sentence
mentions, and carries the label across pages. 97% of chunks get a section; the header is
~17 tokens. Motivation: 5 of the 16 wrong end-to-end answers came from the LLM reading
the wrong table (e.g. a parent-company-only schedule instead of the consolidated
statement), and a prompt rule alone (prompt v3) fixed none of them.

Headed chunks were re-embedded with Qwen on Colab (`tools/colab_embed_qwen.py`, which
now checks token lengths on a 2,000-chunk sample — tokenizing all 107K chunks at once
exhausted Colab RAM) and indexed with `embedder3.py --headed` / `bm25_builder.py --headed`
(`faiss_qwen_headed.index`, `bm25_index_headed/`, `chunk_meta_headed.jsonl`).
Selected with `Retriever(embedding="qwen_headed")`. Grading strips the header line, so a
header's page or note number can never count as an answer value.

Answer in top k, 71 no-LLM questions, same pipeline, only the chunk text differs:

| Configuration | @1 | @3 | **@5** | @10 | MRR |
|---|---|---|---|---|---|
| Qwen hybrid, no rerank | 22 | 34 | 43 (61%) | 51 | 0.433 |
| Qwen **headed** hybrid, no rerank | **28** | **41** | **47 (66%)** | 52 | **0.511** |
| Qwen + Cohere rerank top 40 | 40 | 54 | 58 (82%) | 61 | 0.674 |
| Qwen **headed** + Cohere rerank top 40 | 40 | **56** | **60 (85%)** | 61 | **0.682** |

McNemar, headed vs original: no rerank @1 +7/−1 (p = 0.07); with rerank @5 +5/−3
(p = 0.73). Tune 33 → 37, holdout 25 → 23 at @5. **Verdict:** headers clearly help the
first-stage search; after reranking the gain is small and not significant. Kept because
it costs nothing at query time and targets the wrong-table errors end-to-end (below).

---

## End-to-end results (all 150 questions)
Generation: Cerebras `gpt-oss-120b` (OpenAI-compatible HTTP, 5 req/min, 150/hour),
[`source/GENERATION/answer_generator.py`](../../source/GENERATION/answer_generator.py);
runner [`tools/test_answers.py`](../../tools/test_answers.py); comparison
[`tools/compare_answer_runs.py`](../../tools/compare_answer_runs.py).

**Grading.** Numeric answers are auto-graded; every other answer was judged by hand
against the expected answer, with the same rules for every run: CORRECT (core answer
matches; rounding or equivalent units accepted, e.g. −1.53% for −0.02), PARTIAL (right
direction, key element missing or wrong), WRONG, NOT FOUND (includes non-answers not
worded "NOT FOUND"). Verdicts are stored in `data/PROCESSED/manual_verdicts.json`,
pinned to a hash of the exact answer text.

### Pipelines compared
| Outcome | Qwen + rerank (original chunks)¹ | **Qwen headed + rerank** |
|---|---|---|
| **Correct** | 117 (78.0%) | **119 (79.3%)** |
| Partial | 5 | 3 |
| Wrong | 16 | 20 |
| NOT FOUND | 12 | 8 |

¹ Final answers stitched from three partial runs (prompt v1 for all 150, prompt v2 re-run
of the wrong/partial ones, top-10 retry of the NOT FOUND ones), regraded with the rules
above (originally reported as 116/7/16/11). The headed run is one clean run of all 150
with prompt v2 and the top-10 fallback.

| Group | Original | Headed |
|---|---|---|
| Answer written in the filing (71) | 57 | **59** |
| ├ tune (43) | 33 | 35 |
| └ holdout (28) | 24 | 24 |
| Needs calculation / synthesis (79) | 60 | 60 |
| By type: domain-relevant / metrics-generated / novel-generated (50 each) | 41 / 42 / 34 | 41 / 44 / 34 |

Headed vs original: **+10 / −8, McNemar p = 0.81 — not significant.** Fixed include 3 of
the 5 wrong-table errors (AmEx largest liability, Verizon quick ratio, AMD D&A margin),
Best Buy acquisitions and three former NOT FOUNDs. Lost include three yes/no flips
(Boeing legal, PepsiCo legal, AmEx operating margin) — likely run-to-run LLM variance,
which at ±8 questions is the size of the gains being measured.

### Prompt history
| Prompt | Change | Result | Status |
|---|---|---|---|
| v1 | Extract figures, then compute; units; fiscal-year column; NOT FOUND | 106/150 (first full run) | Replaced |
| **v2** | General rules from v1 failures: apply defined formulas exactly, keep signs, figure before judgement, specific "what drove" factors | +4 on the re-run wrong answers | **Active** |
| v3 | v2 + consolidated statements / never assume zero / no metric substitution | 0 of 16 fixed, 1 correct lost | Removed |
| v4 + reasoning "high" | New format: INPUTS (row label as printed, period, source) → FORMULA → checks → answer | On the 31 headed failures: 4 fixed (all non-calculation; 3 were earlier flips), **0 of 18 calculation failures fixed**, 5 empty answers (hidden reasoning used the whole 6,000-token budget) | Kept in code for reference, not used |

### Where the 31 remaining failures come from (headed run)
| Cause | ~Count |
|---|---|
| Needed figures not retrieved: multi-year / multi-statement questions, wrong filing | ~14 |
| LLM picked the wrong figure although every input was in its chunks | ~7 |
| Yes/no judgement flips between runs | ~4 |
| Ambiguous or debatable expected answer — left alone to avoid overfitting | ~4–5 |

Of the 11 calculation questions answered with a wrong number, 7 had every needed input in
the chunks (checked against the labelled evidence), 1 had a balance sheet cut off at a
chunk boundary, and 3 retrieved the wrong filing or period.

---

## Next ideas (not pursued — accuracy work closed at 79.3%)
- **Whole-statement retrieval** for calculation questions: fetch the complete statement
  pages for the requested filing and year, using the section labels in the chunk headers.
- **Multi-year metadata filter**: CAGR / year-over-year questions need earlier filings;
  fix the `VaR` → Activision bug, "this 10-K" with no year, guidance that sits in the
  previous year's Q4 release.
- A stronger answering model for calculation questions (test on the 7 good-chunk failures).
- Measure run-to-run variance before trusting any gain under ~8 questions.
- An LLM judge to replace manual grading of free-text answers.
