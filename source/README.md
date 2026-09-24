# 02-CODE: Source Code

The application code organized by pipeline stage. Each module is independent and tested.

## 🔄 Pipeline Stages

```
Question → [INGESTION] → [RETRIEVAL] → [GENERATION] → [GUARDRAILS] → Answer + Traces
             (Week 1)      (Week 3)      (Week 4)      (Week 4)       (Week 5)
```

## 📂 Modules

### `INGESTION/` — Load & Process PDFs
**Week 1 deliverable: v0.1.0**

Reads PDFs, extracts text, chunks into smaller pieces, and prepares for embedding.

**Key files**:
- `pdf_parser.py` — Extract text by page from PDFs
- `text_cleaner.py` — Normalize text (remove duplicates, fix encoding)
- `chunker.py` — Split into chunks (recursive, structure-aware)
- `metadata_extractor.py` — Company, year, doc type, section labels

**Output**: `../01-DATA/PROCESSED/chunks.jsonl`

### `RETRIEVAL/` — Search & Rank
**Week 3 deliverable: v0.3.0**

Given a question, find the most relevant pages from the PDFs.

**Key files**:
- `embedding_store.py` — Query dense vector search (pgvector)
- `keyword_search.py` — BM25 full-text search
- `hybrid_retriever.py` — Combine dense + keyword with rank fusion
- `reranker.py` — Cross-encoder reranking (top-5)
- `metadata_filter.py` — Extract company/year from question

**Input**: Question string
**Output**: List of (page_number, text, score) tuples

### `GENERATION/` — Generate Answers
**Week 4 deliverable: v0.4.0**

Takes retrieved pages and uses an LLM to generate an answer with citations.

**Key files**:
- `prompt_builder.py` — Format retrieved context for LLM
- `answer_generator.py` — Call Groq API, extract citations
- `calculator.py` — Optional tool: evaluate formulas (for numeric answers)
- `citation_formatter.py` — Add page numbers to answers

**Input**: Question + retrieved pages
**Output**: Answer string with [Page X] citations

### `GUARDRAILS/` — Validate Input & Output
**Week 4 deliverable: v0.4.0**

Check for off-topic questions, prompt injection, and invalid answers.

**Key files**:
- `input_validator.py` — Reject off-topic, malicious questions
- `output_grounding.py` — Verify answer text comes from retrieved pages
- `confidence_checker.py` — Refuse low-confidence answers
- `financial_safety.py` — Detect investment advice, disclaimer checking

**Input**: Question (input validation) or Answer (output validation)
**Output**: Pass/Fail + reason

### `TRACING/` — Logging & Metrics
**Week 5 deliverable: v0.5.0**

Record latency, tokens used, cost per request, and detailed step traces.

**Key files**:
- `langfuse_client.py` — Initialize and configure Langfuse
- `step_tracer.py` — Log each pipeline step (ingestion, retrieval, generation)
- `cost_calculator.py` — Compute cost per LLM call
- `metrics_reporter.py` — Format results for storage and display

**Captured**:
- Latency (p50, p95, p99)
- Tokens (input, output, total)
- Cost per question and per 1000 questions
- Full trace hierarchy (question → steps → sub-steps)

### `SHARED/` — Utilities
Shared across all modules.

**Key files**:
- `config.py` — Load settings (database, API keys)
- `database.py` — PostgreSQL + pgvector connection
- `types.py` — Pydantic models (Chunk, Question, Answer, etc)
- `logging.py` — Structured logging

## 🧪 Testing

Each module has corresponding tests in `../03-TESTS/UNIT/` and `../03-TESTS/INTEGRATION/`

Example:
- `02-CODE/INGESTION/chunker.py` → `03-TESTS/UNIT/test_chunker.py`
- Full pipeline → `03-TESTS/INTEGRATION/test_end_to_end.py`

## 🔌 Dependencies

Install once in root:
```bash
pip install -r requirements.txt
```

Key libraries:
- `fastapi` — Web API
- `pydantic` — Data validation
- `psycopg2-binary` — PostgreSQL
- `pgvector` — Vector operations
- `sentence-transformers` — Embeddings
- `rank_bm25` — Keyword search
- `groq` — LLM API client
- `langfuse` — Tracing

---

**Status**: Ready for Week 1 development
