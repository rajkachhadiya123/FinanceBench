# data: Data Files

All data for the FilingLens project—inputs and generated outputs.

## 📂 Structure

### `RAW/` — Original PDFs
- **Content**: 368 SEC financial filings (10-Ks, 10-Qs, 8-Ks)
- **Format**: PDF files
- **Size**: ~0.66 GB
- **Source**: FinanceBench dataset (https://github.com/patronus-ai/financebench)
- **Companies**: 40 US public companies, years 2015–2023

### `PROCESSED/` — Generated Data
- **Content**: Created during ingestion pipeline (Week 1)
- **Expected outputs**:
  - `chunks.jsonl` — Text chunks with metadata
  - `embeddings.parquet` — Vector embeddings
  - `metadata.json` — Company/year/document mappings

### `EVAL/` — Evaluation Data
- **Content**: Question-answer pairs and benchmark data
- **Files**:
  - `financebench_open_source.jsonl` — 150 Q&A pairs with evidence
  - `financebench_document_information.jsonl` — PDF metadata
- **Format**: JSONL (JSON Lines)
- **Use**: Evaluate system accuracy in Weeks 2–6

## 📊 Data Format

### Questions (EVAL/)
```json
{
  "id": "1234",
  "question": "What is FY2022 revenue?",
  "answers": ["$1.5B", "$1,500,000,000"],
  "evidence_pages": [
    {
      "file_name": "Company_2022_10K.pdf",
      "page_number": 42,
      "page_text": "..."
    }
  ]
}
```

### Chunks (PROCESSED/, after ingestion)
```json
{
  "id": "chunk_001",
  "text": "...",
  "document": "Company_2022_10K.pdf",
  "page_number": 42,
  "company": "Company",
  "year": 2022,
  "section": "Financial Statements"
}
```

## ⚙️ Pipeline Steps

1. **Week 1**: PDFs (RAW) → Parse → Chunk → Embed → PROCESSED
2. **Week 2+**: PROCESSED + EVAL → Retrieval → LLM → Evaluate → Metrics

---

**Status**: RAW and EVAL data ready ✓
