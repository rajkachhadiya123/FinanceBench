# Known Limitations — Baseline (First Attempt)

Honest assessment of the first working end-to-end retrieval pipeline. Nothing here is hypothetical; every item is either measured or a known design decision we made and can name.

## Measured Results

| Metric | Result | What it measures |
|---|---|---|
| Hit@1 | 8/150 (5.3%) | Labelled evidence page was the top result |
| Hit@5 | 19/150 (12.7%) | Labelled evidence page appeared in top 5 |
| MRR | 0.080 | Average of 1/(rank of first correct page) |
| Answer value present in retrieved text | 90/150 (60.0%) | **Inflated — see L-12** |
| Correct document retrieved (any page) | 58/150 (38.7%) | Realistic ceiling for answer correctness |

### Failure breakdown (top 5)

```
Right document AND right page :  19/150  (12.7%)
Right document, WRONG page    :  39/150  (26.0%)
WRONG document entirely       :  92/150  (61.3%)   ← dominant failure mode
```

---

## Retrieval

**L-1. No metadata filtering — the single largest flaw.** 61.3% of questions retrieve from the wrong document entirely. The corpus holds 368 filings from only ~40 companies, so each company has ~9 near-identical documents (3M's 2018 and 2021 10-Ks share most of their language, headings, and line items). Semantic similarity cannot separate them because they genuinely *are* near-identical — the distinguishing information is the document's identity, not its prose. Questions name a company and fiscal year; we currently ignore that and search all 107,049 chunks.

**L-2. No re-ranking.** Final ordering comes straight from RRF fusion of two first-stage retrievers. A cross-encoder would re-score candidates by reading query and chunk together, which is the standard fix for the 26% "right document, wrong page" bucket.

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

Ranked by measured impact, not by ease:

1. **L-1 metadata filtering** — addresses 61.3% of failures. Collapses the search space from 107,049 chunks to roughly 300. Requires no re-indexing, so it is simultaneously the highest-impact and cheapest change to test.
2. **L-12 / L-11 fix the metric** — adopt document-scoped answer presence as the primary metric, so subsequent improvements are measured against something meaningful rather than against an artifact.
3. **L-5 / L-6 embedding** — eliminate truncation and upgrade the model. Requires re-embedding the full corpus (~70+ minutes).
4. **L-2 re-ranking** — targets the remaining 26% "right document, wrong page" bucket.
5. **L-15 dev/holdout split** — should be established *before* extensive tuning, to keep the final numbers honest.
