# documentation

| Document | Contents |
|---|---|
| [experiment_log/README.md](experiment_log/README.md) | The full history: V1 (MiniLM) → V2 (BGE) → V3 (Qwen + Cohere rerank + chunk headers), every measurement, end-to-end results on all 150 questions, prompt versions, failure analysis, ideas not pursued |
| [LIMITATIONS.md](LIMITATIONS.md) | Honest assessment of the current system: where the remaining 31 failures come from, retrieval / generation / evaluation limitations, what is not built |
| `reports/*.pdf` | Early retrieval reports (baseline, metadata filter, reranking, oracle) from `tools/evaluate_retrieval.py`. They predate the strict answer matcher, so their headline percentages are optimistic — the experiment log has the corrected numbers |

Working notes for continuing development are in [HANDOFF.md](../HANDOFF.md) at the repository root.
