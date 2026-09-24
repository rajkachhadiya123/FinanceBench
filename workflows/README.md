# 09-CI: GitHub Actions CI/CD

Automated testing and evaluation on every push and PR.

## 📂 Workflows (to be created)

### `.github/workflows/lint.yml`
**Runs on**: Every push and PR

**Checks**:
- Linting with `flake8`
- Type checking with `mypy`
- Code formatting with `black`

**Fails if**: Code has style issues

### `.github/workflows/test.yml`
**Runs on**: Every push and PR

**Tests**:
- Unit tests (`pytest 03-TESTS/UNIT/`)
- Integration tests (`pytest 03-TESTS/INTEGRATION/`)

**Fails if**: Any test fails

### `.github/workflows/evaluate.yml`
**Runs on**: Every PR (Week 2+)

**Evaluation Gate**:
- Run system on 100 dev questions
- Compare retrieval accuracy vs. baseline
- Compare answer accuracy vs. baseline
- Run McNemar statistical test

**Fails if**: Metrics drop below baseline

**Output**: Comment on PR with results

```
✓ Retrieval Hit@5: 62% (baseline: 60%) +2%
✓ Answer Accuracy: 48% (baseline: 45%) +3%
✓ All changes passed! ✅
```

### `.github/workflows/release.yml`
**Runs on**: Tag push (Week 6)

**Steps**:
1. Run full evaluation on all 150 questions
2. Generate results markdown
3. Create GitHub Release with metrics
4. Tag Docker image and push to registry (optional)

## 🔧 Configuration

### Secrets (GitHub repo settings)
```
GROQ_API_KEY         # For evaluation
DATABASE_URL         # Test database
LANGFUSE_PUBLIC_KEY  # Optional tracing
LANGFUSE_SECRET_KEY
```

### Environment Variables
Defined in workflow YAML files:
```yaml
env:
  PYTHON_VERSION: "3.11"
  POSTGRES_VERSION: "16"
```

## 📊 Evaluation Gate Details (Week 2+)

When you open a PR:

1. **Dev set evaluation** (100 questions, ~2 min)
   - Retrieval: hit@5, recall@5, MRR
   - Answers: accuracy, faithfulness

2. **Statistical test** (McNemar)
   - P-value for accuracy change
   - Reject if p > 0.05 and accuracy drops

3. **Report**
   - Comment with metrics table
   - Red ✗ if fails, Green ✓ if passes

## 🚀 Local Testing

Test workflows locally with `act`:

```bash
# Install act
choco install act-cli

# Run a specific workflow
act -j test
act -j lint
act -j evaluate
```

---

**Status**: Start Week 1, implement evaluation gate Week 2
