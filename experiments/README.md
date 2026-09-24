# 04-CONFIG: Experiment Configurations

YAML files defining versions and settings for each experiment iteration.

## 📂 Structure

Each version has its own config file:

```
04-CONFIG/
├── v0.1_ingestion.yaml      # Week 1: Basic chunking
├── v0.2_baseline.yaml       # Week 2: Baseline retrieval
├── v0.3_retrieval.yaml      # Week 3: Optimized retrieval
├── v0.4_guardrails.yaml     # Week 4: Guardrails & tables
├── v0.5_production.yaml     # Week 5: Tracing & hardening
└── v1.0_final.yaml          # Week 6: Final release
```

## 📋 Example Config Format

```yaml
version: "0.3.0"
name: "Hybrid Search with Reranking"
week: 3

# INGESTION settings
ingestion:
  chunk_size: 512
  chunk_overlap: 100
  strategy: "recursive"

# RETRIEVAL settings
retrieval:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  dense_weight: 0.6
  keyword_weight: 0.4
  top_k: 20
  rerank_top_k: 5
  reranker: "BAAI/bge-reranker-base"

# LLM settings
llm:
  model: "mixtral-8x7b-32768"
  provider: "groq"
  temperature: 0.1
  max_tokens: 1024

# GUARDRAILS
guardrails:
  enabled: false  # Week 4+
  confidence_threshold: 0.7
```

## 🔄 Usage

```bash
python 05-SCRIPTS/evaluate.py --config 04-CONFIG/v0.3_retrieval.yaml
```

---

**Status**: Ready for Week 1
