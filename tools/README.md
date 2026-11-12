# tools

Command-line scripts for evaluation and data preparation. Run them from the repository root
(`python tools/<script>.py`). Scripts that call the LLM (DeepSeek) or Cohere need the keys in `.env`.

## Evaluation

| Script | What it measures | API cost |
|---|---|---|
| `test_answers.py` | **End-to-end answers**: retrieval → rerank → LLM → automatic grading of numeric answers. Answers are appended to `data/PROCESSED/answer_results.jsonl` and reused on reruns (tagged by model, retrieval, prompt, reasoning effort) | ~1–2 LLM + 1 Cohere call per uncached question |
| `evaluate_reranker.py` | **Retrieval only** on the 71 questions whose answer is written in the filing (43 tune / 28 holdout): answer chunk in the top 1/3/5/10, MRR, McNemar test against the baseline | 1 Cohere call per question, cached |
| `compare_answer_runs.py` | Compares two end-to-end runs question by question using `manual_verdicts.json`; `--pending` lists answers that still need a human verdict | none |
| `evaluate_guardrails.py` | Input guards on all 150 questions plus attack and tricky-legitimate sets; output signals on the 150 graded answers | Cohere only for scores not yet cached |
| `evaluate_retrieval.py` | Older retrieval evaluator that wrote the PDF reports in `documentation/reports/`. Its default answer matcher is lenient — use `evaluate_reranker.py` for numbers | none |

Common commands:

```bash
# the 79.3% run (all 150 questions, top 5, retry with top 10 on NOT FOUND)
python tools/test_answers.py --include-holdout --n 200 --top-k 5 --retry-top-k 10 --no-targeted --embedding qwen_headed

# re-run only chosen questions (one financebench_id per line)
python tools/test_answers.py --include-holdout --n 200 --embedding qwen_headed --ids my_ids.txt

# retrieval, headed chunks vs the original chunks
python tools/evaluate_reranker.py --embedding qwen_headed
```

## Data preparation

| Script | Role |
|---|---|
| `add_chunk_headers.py` | Prefixes every chunk with `[COMPANY \| TYPE \| PERIOD \| page N \| section]` → `chunks_headed.jsonl` + `.zip`. `--sample` prints a review sample without writing |
| `colab_embed_qwen.py` | Google Colab (T4 GPU) script that embeds a chunk zip with Qwen3-Embedding-0.6B; resumable via Google Drive |
| `colab_embed_bge.ipynb`, `build_bge_index.py` | The same for BGE-base (retired) |

The full rebuild order is in [data/README.md](../data/README.md).
