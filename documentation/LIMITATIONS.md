# Known Limitations

Honest assessment of the FilingLens pipeline as of 2026-10-15 (accuracy work closed).
Every item is measured or a named design decision. Full history and numbers:
[`experiment_log/README.md`](experiment_log/README.md).

## Current system and results

```
Question → metadata filter (company + fiscal year from the question text)
         → Qwen3-Embedding-0.6B dense top 20 + BM25 top 20 on headed chunks
         → RRF → Cohere rerank-v4.0-pro → top 5
         → Cerebras gpt-oss-120b (prompt v2) → answer; retry with top 10 on NOT FOUND
```

| Measure | Result |
|---|---|
| **End-to-end, all 150 questions** | **119 correct (79.3%)**, 3 partial, 20 wrong, 8 NOT FOUND |
| Answer chunk in top 5 (71 questions whose answer is written in the filing) | 60 (85%) |
| Same, holdout 28 (never tuned on) | 23 (82%) |

## Where the remaining 31 failures come from

| Cause | ~Count | Limitation |
|---|---|---|
| Needed figures not retrieved | ~14 | L-1, L-2, L-3 |
| LLM picks the wrong figure although every input is in its chunks | ~7 | L-6 |
| Yes/no judgements flip between runs | ~4 | L-7 |
| Ambiguous or debatable expected answer | ~4–5 | L-12 |

---

## Retrieval

**L-1. The metadata filter keeps one fiscal year.** Questions that span years (a 2-year
CAGR, "FY20 vs FY21") need earlier filings that the filter excludes. Forward-looking
guidance lives in the *previous* year's Q4 earnings release (e.g. FY2023 guidance in the
FY2022 Q4 release) and is filtered out. "This 10-K" with no year is not resolved. Known
bug: "VaR" in a JPMorgan question matches Activision.

**L-2. Calculation questions need several statements at once.** A ratio may need the
balance sheet, income statement and cash flow statement; top-5 chunks often cover only
some of them, which produces most NOT FOUND answers. Not addressed: per-statement
sub-queries exist in the retriever (`targeted=True`) but are off and unmeasured on the
current pipeline; whole-statement retrieval was proposed but not built.

**L-3. Fixed-size chunks cut tables.** Chunks are 2,000 characters with 300 overlap,
ignoring layout. A balance sheet can start partway into a chunk, so the top rows are in
the previous chunk (seen in one wrong answer). Context headers label each chunk with
its statement but do not change boundaries.

**L-4. Section labels are heuristic.** Headers come from a line-by-line heading
detector. A chunk spanning two sections gets only the first label; earnings releases and
8-Ks have little structure, so many of their chunks carry only company, type, period and
page; statement titles without "Consolidated" or "Balance Sheet" are deliberately not
labelled.

**L-5. External API limits.** Reranking uses Cohere's trial key (10 calls/min,
1,000/month). A full 150-question run spends ~150–165 calls. A local cross-encoder of
similar quality takes ~1–1.5 min per question on this CPU.

## Generation

**L-6. The LLM picks wrong figures in calculations.** In 7 of 11 wrong calculations
every input was in the chunks; the model divided by the wrong line, used a quarterly
column for a full year, substituted a similar metric, or dropped a sign. Two prompt
changes did not fix this: v3 (more rules) fixed 0 of 16; v4 (row labels, formula and
checks, with high reasoning effort) fixed 0 of 18 calculation failures and returned 5
empty answers when hidden reasoning used the whole token budget. Arithmetic is not
delegated to code.

**L-7. Answers vary between runs.** Temperature is 0, yet answers are not fully
repeatable. Between the original and headed runs 10 questions were gained and 8 lost,
including yes/no judgements (Boeing and PepsiCo legal matters) that the retrieval change
should barely affect. Run-to-run variance has not been measured directly; gains smaller
than ~8 questions cannot be attributed to a change without repeated runs.

**L-8. Rate limits shape the system.** Cerebras allows 5 requests/min and 150/hour; a
full run takes ~45–60 minutes. There is no batching or caching across users.

## Evaluation

**L-9. Free-text answers are graded by hand.** The automatic grader handles numbers
only; about 45% of answers needed a human verdict. Verdicts follow written rules and are
pinned to the answer text, but there is no LLM judge and no second grader.

**L-10. The baseline comparison is not a clean run.** The original-chunk result
(117/150) is stitched from three partial runs; the headed result (119/150) is one clean
run. Headed vs original is +10 / −8, p = 0.81 — not significant.

**L-11. Small samples.** With 71 retrieval questions or 150 end-to-end, differences of a
few questions are noise (paired McNemar tests are reported throughout). The 28-question
holdout is too small to confirm small gains.

**L-12. Some expected answers are debatable.** A few gold answers rely on a convention
(average vs ending inventory), an unusual figure (a spin-off cost of 77.78) or a
judgement ("similar company"). They are left as failures rather than tuned for.

## Not yet built

**L-13. No guardrails** — no off-topic or prompt-injection checks, no check that the
answer's numbers appear in the retrieved text, no confidence threshold, no financial
advice disclaimer. `source/GUARDRAILS/` is empty.

**L-14. No API** — the pipeline runs from scripts; nothing is exposed as a service.

**L-15. No tracing** — per-step latency, tokens and cost are printed by the test runner
but not recorded per request. `source/TRACING/` is empty.

**L-16. No deployment, tests or CI** — no Dockerfile, no unit tests, no automated checks.

**L-17. Local files instead of a database.** FAISS and bm25s indexes on disk (~2 GB in
`data/PROCESSED/`, gitignored), not PostgreSQL + pgvector. Fine for one user; no
concurrent writes.

## Data handling

**L-18. Charts are ignored.** Verified to be a non-issue for this benchmark (no question
depends on a chart), but a figure that appears only in a graphic cannot be found.

**L-19. Tables are linear text.** PyMuPDF extracts tables in reading order; row/column
structure is implicit. Table-to-Markdown conversion was tried and dropped because it
fragmented tables and split words.
