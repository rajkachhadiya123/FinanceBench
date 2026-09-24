# 03-TESTS: Test Suite

Automated tests to ensure code quality and catch regressions.

## 📂 Structure

### `UNIT/` — Module Tests
Test individual functions in isolation (Week 1+)

**Examples**:
- `test_pdf_parser.py` — PDF text extraction
- `test_chunker.py` — Text chunking logic
- `test_embedding_store.py` — Vector similarity search
- `test_guardrails.py` — Input/output validation

**Run**: `pytest 03-TESTS/UNIT/ -v`

### `INTEGRATION/` — Pipeline Tests
Test the full pipeline end-to-end (Week 2+)

**Examples**:
- `test_ingest_to_search.py` — Ingest → Query → Retrieve
- `test_qa_pipeline.py` — Question → Answer with metrics
- `test_api_endpoints.py` — FastAPI endpoints
- `test_guardrails_e2e.py` — Full guardrail flow

**Run**: `pytest 03-TESTS/INTEGRATION/ -v`

## 🎯 Coverage Goals

- **Unit tests**: 80%+ coverage per module (Week 2)
- **Integration tests**: All major workflows (Week 2)
- **Evaluation tests**: 150 benchmark questions (Week 2+)

## 🚀 Running Tests

```bash
# All tests
pytest 03-TESTS/ -v

# Specific module
pytest 03-TESTS/UNIT/test_chunker.py -v

# With coverage
pytest 03-TESTS/ --cov=02-CODE --cov-report=html

# Fast mode (skip integration)
pytest 03-TESTS/UNIT/ -v
```

## ⚡ CI/CD Integration

Tests run on every PR via GitHub Actions (Week 5):
- Linting with `flake8`
- Type checking with `mypy`
- Unit tests
- Integration tests
- Evaluation gate (must not decrease accuracy)

See `../09-CI/` for workflow files.

---

**Status**: Ready for Week 1
