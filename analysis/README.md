# 08-NOTEBOOKS: Exploration & Analysis

Jupyter notebooks for data exploration and ad-hoc analysis.

## 📂 Notebooks (to be created)

### `01_FinanceBench_Overview.ipynb`
**Week 0**

Explore the dataset:
- Load questions and answers
- Analyze question types
- PDF metadata distribution
- Evidence page analysis

### `02_PDF_Parsing.ipynb`
**Week 1**

Test PDF parsing logic:
- Load a sample PDF
- Extract text by page
- Visualize page structure
- Check text quality

### `03_Chunking_Analysis.ipynb`
**Week 1**

Evaluate chunking strategies:
- Fixed-size chunks
- Recursive chunks
- Structure-aware chunks
- Visualize chunk sizes and overlaps

### `04_Retrieval_Performance.ipynb`
**Week 2-3**

Analyze retrieval results:
- Dense vs. keyword search comparison
- Rank fusion effects
- Reranker impact
- Failure case analysis

### `05_Answer_Quality.ipynb`
**Week 2-4**

Evaluate answer generation:
- Sample answers with citations
- Hallucination detection
- Faithfulness analysis
- Numeric accuracy check

### `06_Guardrails_Testing.ipynb`
**Week 4**

Test guardrail logic:
- Off-topic detection examples
- Injection detection
- Confidence calibration
- False positive/negative rates

### `07_Final_Results.ipynb`
**Week 6**

Compile final metrics:
- Version comparison table
- Accuracy by question type
- Latency distribution
- Cost analysis

## 🚀 Running Notebooks

```bash
cd 08-NOTEBOOKS
jupyter notebook
```

Then open a notebook in the browser.

## 📝 Notes

- Use `../` to reference parent directories
- Load data from `../01-DATA/`
- Save results to `../01-DATA/PROCESSED/`
- Keep notebooks for reference; re-run to verify results

---

**Status**: Ready for Week 0
