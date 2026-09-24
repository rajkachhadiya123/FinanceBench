# 05-SCRIPTS: Command-Line Tools

Ready-to-use CLI scripts for common tasks.

## 🛠️ Main Scripts

### `ingest.py` — Load & Process PDFs
**Week 1**

Reads all PDFs from `01-DATA/RAW/`, chunks them, and stores in database.

```bash
python ingest.py \
  --pdf-dir ../01-DATA/RAW \
  --output ../01-DATA/PROCESSED \
  --chunk-size 512 \
  --strategy recursive
```

**Output**: Chunks in PostgreSQL + pgvector index

### `query.py` — Search Database
**Week 2**

Search for relevant pages given a question.

```bash
python query.py \
  --question "What is FY2022 revenue?" \
  --top-k 5
```

**Output**: Top K pages with scores

### `evaluate.py` — Run Full Evaluation
**Week 2+**

Run the system on all 150 benchmark questions and measure accuracy.

```bash
python evaluate.py \
  --config ../04-CONFIG/v0.3_retrieval.yaml \
  --dataset ../01-DATA/EVAL/financebench_open_source.jsonl \
  --output ../01-DATA/PROCESSED/results_v0.3.json
```

**Output**: JSON with metrics (hit@k, accuracy, latency, cost)

### `compare_versions.py` — Compare Results
**Week 3+**

Compare metrics across versions and run statistical tests.

```bash
python compare_versions.py \
  --versions v0.1 v0.2 v0.3 \
  --metric accuracy
```

**Output**: CSV table with results and McNemar test p-values

### `serve.py` — Start FastAPI Server
**Week 5**

Run the production API locally.

```bash
python serve.py --port 8000
```

**API endpoint**: `POST /query` with question JSON

### `trace.py` — View Traces
**Week 5**

Display latency and cost breakdowns from Langfuse.

```bash
python trace.py --question-id abc123
```

---

**Status**: Ready for Week 1
