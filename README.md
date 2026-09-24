# FilingLens: Production-Grade RAG System for Financial Q&A

A complete, evaluated question-answering system over SEC financial filings using the FinanceBench benchmark.

## 📁 Project Structure

```
FinanceBench/
├── data/                 ← All data (inputs and outputs)
├── source/               ← Source code organized by pipeline stage
├── tests/                ← Unit and integration tests
├── experiments/          ← Experiment configuration files
├── tools/                ← CLI commands (ingest, query, evaluate)
├── deployment/           ← Docker & Docker Compose files
├── documentation/        ← Architecture & design documents
├── analysis/             ← Jupyter notebooks for exploration
├── workflows/            ← GitHub Actions CI/CD workflows
└── meta/                 ← Project metadata (pyproject.toml, .env, etc)
```

## 🚀 Quick Start

1. **Install dependencies** (Week 0-1)
   ```bash
   cd source
   pip install -r requirements.txt
   ```

2. **Ingest PDFs** (Week 1)
   ```bash
   python ../tools/ingest.py
   ```

3. **Run baseline evaluation** (Week 2)
   ```bash
   python ../tools/evaluate.py
   ```

4. **Iterate and improve** (Weeks 3-5)

## 📊 6-Week Timeline

| Week | Goal | Version |
|------|------|---------|
| 0 | Setup & exploration | - |
| 1 | PDF ingestion pipeline | v0.1.0 |
| 2 | Baseline + evaluation | v0.2.0 |
| 3 | Retrieval optimization | v0.3.0 |
| 4 | Tables & guardrails | v0.4.0 |
| 5 | Production hardening | v0.5.0 |
| 6 | Final release | v1.0.0 |

## 📈 Data

- **Questions**: 150 from financial analysts
- **PDFs**: 368 SEC filings from 40 US companies (2015-2023)
- **Location**: `01-DATA/`
  - `EVAL/`: Question/answer pairs and metadata
  - `RAW/`: Original PDF files
  - `PROCESSED/`: Generated chunks and embeddings (after ingestion)

## 📚 Documentation

See `07-DOCS/` for:
- Architecture diagram
- Decisions log
- Known limitations
- Results and metrics

---

**Status**: Week 0 setup complete ✓
