# Known Limitations

Honest assessment of the retrieval pipeline. Nothing here is hypothetical; every item is either measured or a known design decision we made and can name.

## Measured Results

| Config | Grounded answer | Correct doc | Hit@1 | Hit@3 | Hit@5 | MRR | Latency |
|---|---|---|---|---|---|---|---|
| Baseline (BM25 + dense + RRF) | 18.0% | 38.7% | 5.3% | — | 12.7% | 0.080 | 0.03s |
| **+ metadata filtering** | **46.7%** | **82.0%** | **17.3%** | **28.0%** | **32.7%** | **0.230** | 0.03s |
| + cross-encoder reranking | 49.3% | 83.3% | 12.7% | 25.3% | 32.0% | 0.197 | 1.82s |
| *Oracle doc filter (diagnostic)* | *62.0%* | *100%* | *26.0%* | *44.0%* | *49.3%* | *0.353* | *0.03s* |

"Grounded answer" is the primary metric: the expected value found **in the correct document**. The oracle row uses the benchmark's ground-truth document name — data leakage, reported only as a ceiling, never as a result.

### The metric cannot reach 100% — 26.7% of answers must be computed

**38% of questions (57/150) have an answer that does not appear in their own labelled evidence**, because the expected value must be calculated: ratios, multi-year averages, year-over-year deltas. Examples: *"FY2019 fixed asset turnover ratio"* (revenue ÷ average fixed assets), *"FY2017-FY2019 3 year average of capex as a % of revenue"*. No retriever can surface a number that was never written down.

Accounting for figures that are retrievable from a non-labelled page of the correct filing, the true satisfiable set is **110/150 (73.3%)**:

```
 40 (26.7%)  answer exists nowhere retrievable -- pure computation, needs the LLM
110 (73.3%)  answer IS retrievable  <-- the real ceiling
 ├─  70      currently found   = 46.7% of all, 63.6% OF ACHIEVABLE
 └─  40      currently missed
     ├─ ~23  wrong document (extraction is 82%, not 100%)
     └─ ~17  right document, wrong chunk
```

**Report "% of achievable" alongside the raw rate.** Measuring against an unreachable 100% understates retrieval quality by ~17pp and misdirects effort.

**How this was missed initially:** the loose "answer in text" metric read 60-62% across all four configurations — baseline, filtering, reranking and oracle alike. A metric that does not move while the system improves 2.6x is a metric at its ceiling. That signal was printed four times before it was investigated.

### Failure breakdown, current config (metadata filtering)

```
Right document AND right page :  49/150  (32.7%)
Right document, WRONG page    :  74/150  (49.3%)   ← now the dominant failure
WRONG document entirely       :  27/150  (18.0%)
```

---

## Retrieval

**L-1. ~~No metadata filtering~~ — RESOLVED.** Was the largest flaw at 61.3% wrong-document retrieval. Implemented in `source/RETRIEVAL/metadata_filter.py`: extracts company and fiscal year from the question text (not from the benchmark labels) and restricts search to matching filings, reducing the search space from 360 documents to an average of 2.9. Wrong-document fell 61.3% → 18.0%; grounded answers rose 18.0% → 46.7%.

Residual gaps: 12/150 questions name no company in their text at all (*"Based on the information provided..."*), and 2/150 have the correct filing excluded because the answer lives in an adjacent year — forward-looking FY2023 guidance appears in the FY2022 Q4 earnings release, not a FY2023 filing.

**L-2. Cross-encoder reranking measured, and not worth enabling.** Implemented and measured; gained only +2.6pp on grounded answers while *degrading* page precision (Hit@1 17.3% → 12.7%, MRR 0.230 → 0.197) and increasing latency 60× (0.03s → 1.82s per query). The cross-encoder optimizes for "does this chunk answer the question," which favours MD&A narrative over the terse financial-statement page the benchmark labels as evidence. Code is retained but disabled by default.

**L-2b. ~~Within-document retrieval is the dominant failure~~ — CORRECTED.** This was diagnosed wrongly. The oracle run's 38% failure rate was read as a retrieval problem, when most of it was the metric asking for values that do not exist in any document (see above). The measured reality:

- Retrieval recall **given the correct document is 81.7%** (76 of the 93 questions whose answer is present in their labelled evidence)
- Genuine within-document failures: **17 questions (11.3%)**, not 38%
- 17 further questions were satisfied from a *non-labelled* page of the correct filing, confirming that figures repeat within a filing and that document-scoped scoring is the right choice over page-scoped

The correct priority order follows from the recoverable counts, not from the oracle gap: company/year extraction (~23 questions recoverable, cheap) now outranks the embedding upgrade (~17 questions, ~90 minutes of compute).

**L-3. No query understanding.** Questions are passed through verbatim. No company/year extraction, no rewriting, no decomposition — so multi-part questions ("compare 2018 vs 2022 margins") get one undifferentiated search.

**L-4. Overlapping chunks can crowd out the top-5.** Chunks step 1700 characters but span 2000, so adjacent chunks share text. If chunk N is relevant, N±1 usually scores highly too — potentially consuming several of the five slots with near-duplicate content. De-duplication is implemented but disabled by default (not yet measured).

## Embedding

**L-5. Silent truncation — roughly 45% of every chunk never reaches its vector.** `all-MiniLM-L6-v2` accepts a maximum of 256 tokens. Our chunks are ~2000 characters ≈ 450 tokens. The model silently discards the remainder. Semantic search therefore cannot find anything that appears only in the back half of a chunk. This is a design mismatch we introduced by choosing chunk size before checking the model's input limit.

**L-6. Weak embedding model.** `all-MiniLM-L6-v2` is small (384 dimensions) and older. Stronger open models (`BGE-base-en-v1.5`, `gte-base`) score substantially higher on retrieval benchmarks and accept 512 tokens, which would also mitigate L-5.

**L-7. No domain adaptation.** A general-purpose model is being applied to dense financial and accounting language with no finance-specific tuning.

## Chunking

**L-8. Fixed-size cuts ignore document structure.** Chunks are cut at exact character counts with no awareness of tables, sections, or headings. A financial table can be split so that its numbers land in one chunk and its column headers (the fiscal years) land in another. The 300-character overlap means the complete version exists in a neighbouring chunk, so nothing is lost outright — but the split chunk itself can be misleading in isolation.

**L-9. Tables carry no structural information.** Tables are extracted as linear text. Row/column relationships are implicit in reading order only. We verified that plain extraction captures the *content* accurately (100% containment against ground-truth evidence text across 189 samples), but never verified that an LLM can correctly attribute a number to the right year column from that linearized form.

**L-10. Chunk boundaries ignore page boundaries by design.** This is intentional (a topic spanning a page break isn't cut), but it means a chunk can span two pages, which slightly complicates page-level scoring and citation.

## Evaluation methodology

**L-11. Page-level Hit@5 is overly strict.** Financial figures legitimately repeat within a single filing — a revenue number appears in the income statement, the MD&A narrative, the segment footnote, and the selected-financial-data table. FinanceBench labels only one page as evidence, so retrieving a *different valid page containing the same correct figure* is scored as a failure.

**L-12. "Answer value in retrieved text" (60%) is inflated by false positives.** A 2018 figure appears in the 2018, 2019, and 2020 filings alike. When we retrieve the wrong year's document, the expected value still "matches" — rewarding right-number-wrong-source. This metric should be scoped to the correct document.

**L-13. Numeric matching is loose.** Matching allows 1% relative tolerance *and* power-of-ten scale differences (to handle millions vs billions). That tolerance can produce accidental matches on unrelated figures.

**L-14. Text-answer matching is crude.** Non-numeric answers are judged by keyword overlap at a 50% threshold — no semantic understanding, no LLM judge.

**L-15. No dev/holdout split.** All 150 questions were used for this evaluation. The original plan called for 100 dev + 50 holdout. Continuing to tune against all 150 risks overfitting to the test set and producing optimistic final numbers.

**L-16. Single run, no confidence intervals or significance testing.** With n=150, differences of a few percentage points between configurations may be noise. No McNemar test yet.

## Not yet built

**L-17. No answer generation.** Retrieval only. No LLM is involved yet, so end-to-end answer accuracy is unmeasured.

**L-18. No guardrails.** No off-topic detection, prompt-injection checks, grounding verification, or low-confidence refusal.

**L-19. No tracing.** No latency, token, or cost measurement per request.

**L-20. Indexes are local files, not the planned database.** FAISS + bm25s on disk rather than PostgreSQL + pgvector. Deliberate (avoids a fiddly Windows pgvector build while iterating), but it means no metadata filtering in SQL, no concurrent access, and no persistence guarantees.

**L-21. No API, no tests, no CI.** Nothing is exposed as a service; there is no test suite and no automated quality gate.

## Data handling

**L-22. Charts and graphs are ignored.** Verified to be a non-issue for *this* dataset — no question depends on reading a chart, and every evidence entry is substantial text (minimum 67 characters, median 1,271). It remains a limitation for general use on filings where a figure appears only in a plotted graphic.

---

## Priority order for the next iteration

Ordered by questions recoverable per unit of effort, after correcting the L-2b misdiagnosis:

1. **Company/year extraction coverage (L-1 residual) — ~23 questions, cheap.** Pure query-side logic, no re-indexing. Concrete known cases: 12 questions name no company in their text at all (*"Based on the information provided..."*), and the year heuristic misses filings where the answer sits in an adjacent year — forward-looking FY2023 guidance appears in the FY2022 Q4 earnings release.
2. **Move to LLM answer generation — 40 questions are blocked on it.** 26.7% of questions require arithmetic over retrieved tables (ratios, averages, deltas). Retrieval is already doing all it can for these; only a model performing the computation can close them. This is a stronger argument for advancing to the next pipeline stage than for further retrieval tuning.
3. **L-5 / L-6 embedding — ~17 questions, ~90 minutes.** Fix the 256-token truncation (chunks are ~450 tokens, so ~45% of every chunk never reaches its vector) and upgrade to a 512-token model (BGE-base-en-v1.5 or gte-base). Lower priority than previously stated: the recoverable count is 17, not the 57 implied by the earlier misreading.
4. **L-15 dev/holdout split** — should be established *before* further tuning. Six configurations have now been run against all 150 questions; overfitting risk is accumulating.
5. **L-8 / L-9 chunking** — revisit alongside item 3, since chunk size must fit the model's input limit. That mismatch is what created L-5.

~~L-1 metadata filtering~~ — done, +28.7pp.
~~L-12 / L-11 metric fix~~ — done; the old loose metric was inflating results by 3.3×.
~~L-2 re-ranking~~ — measured and rejected; see L-2 above.
~~L-2b within-document bottleneck~~ — misdiagnosis, corrected above.
