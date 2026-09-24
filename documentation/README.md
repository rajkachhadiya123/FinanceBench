# 07-DOCS: Documentation

Design, architecture, and decision records.

## 📂 Files (to be created)

### `ARCHITECTURE.md`
System diagram and component overview.

**Content**:
- High-level pipeline diagram
- Component responsibilities
- Data flow (question → answer)
- Database schema
- API contract

### `DECISIONS.md`
Major decisions and trade-offs (Week 1+)

**Format**: Date | Decision | Rationale | Alternatives Considered

**Examples**:
- Week 1: Why recursive chunking over fixed-size?
- Week 3: Why rank fusion over pure semantic search?
- Week 4: Why local reranker vs. API-based?

### `LIMITATIONS.md`
Known issues and honest failure analysis (Week 6)

**Content**:
- What the system handles well
- Known failure modes
- Open research questions
- Future improvements

### `RESULTS.md`
Final metrics and before/after comparison (Week 6)

**Tables**:
- Version progression (v0.1 → v1.0)
- Retrieval metrics by version
- Answer accuracy improvement
- Latency and cost breakdown

### `API.md` (Week 5)
API endpoint documentation.

**Content**:
- `POST /query` — Ask a question
- `GET /health` — Health check
- Request/response formats
- Error codes

### `DATA_GUIDE.md`
Detailed information about the FinanceBench dataset.

**Content**:
- Question types and distribution
- PDF metadata
- Evidence format
- Data quality notes

---

**Status**: Start documentation in Week 1
