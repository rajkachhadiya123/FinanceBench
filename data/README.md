# data

| Folder | Contents | In git? | Size |
|---|---|---|---|
| `EVAL/` | The 150 FinanceBench questions and the filing list | ✅ yes | ~1 MB |
| `RAW/` | The 368 source PDFs | ❌ gitignored | ~0.7 GB |
| `PROCESSED/` | Chunks, indexes, caches and evaluation results | ❌ gitignored | ~2 GB |

## EVAL/ (in the repository)

| File | Contents |
|---|---|
| `financebench_open_source.jsonl` | 150 questions: `question`, `answer`, `question_type`, `doc_name`, `evidence` (text and page number), `justification` |
| `financebench_document_information.jsonl` | One line per filing: `doc_name`, `company`, `gics_sector`, `doc_type`, `doc_period`, `doc_link` (original URL) |

Both come from the [FinanceBench](https://github.com/patronus-ai/financebench) open-source release.

## RAW/ — getting the PDFs

The 368 PDFs are not in this repository. Download them from the FinanceBench repository's
`pdfs/` folder (or from each filing's `doc_link` in `financebench_document_information.jsonl`)
into `data/RAW/`, keeping the original names (`3M_2018_10K.pdf`, `JOHNSON_JOHNSON_2022Q4_EARNINGS.pdf`, …).
The file name encodes company, period and type, which the metadata filter relies on.

## PROCESSED/ — building the indexes

The API needs these files (the "headed" set; the others are older variants kept for comparison):

| File | Built by | Used for |
|---|---|---|
| `chunks.jsonl`, `chunk_meta.jsonl` | `chunker.py`, `embedder.py` | Original chunks (107,049) and their metadata |
| `chunks_headed.jsonl` | `tools/add_chunk_headers.py` | The same chunks with context headers |
| `chunk_meta_headed.jsonl`, `faiss_qwen_headed.index` | `embedder3.py --headed` | Dense search (Qwen3-Embedding-0.6B, 1024-dim) |
| `bm25_index_headed/` | `bm25_builder.py --headed` | Keyword search |

Build order, from the repository root:

```bash
# 1. PDFs → 2,000-character chunks with 300 overlap
python source/INGESTION/chunker.py                 # → chunks.jsonl

# 2. chunk metadata (also builds the small MiniLM index; CPU, slow-ish)
python source/INGESTION/embedder.py                # → chunk_meta.jsonl, faiss.index

# 3. context headers  "[COMPANY | TYPE | PERIOD | page N | section]"
python tools/add_chunk_headers.py                  # → chunks_headed.jsonl + chunks_headed.zip

# 4. Qwen embeddings on a GPU (Google Colab T4, ~45–90 min): upload chunks_headed.zip,
#    run tools/colab_embed_qwen.py (INPUT_NAME = "chunks_headed"), download
#    qwen_vectors_chunks_headed.npy into data/PROCESSED/

# 5. FAISS index + matching metadata, and the BM25 index
python source/INGESTION/embedder3.py --headed      # → faiss_qwen_headed.index, chunk_meta_headed.jsonl
python source/INGESTION/bm25_builder.py --headed   # → bm25_index_headed/
```

Embedding 107K chunks with a 0.6B model is impractical on CPU, hence Colab in step 4. The raw
`*_vectors.npy` files can be deleted after step 5: the FAISS index stores the same vectors.

### Other files in PROCESSED/

| File | Contents |
|---|---|
| `faiss.index`, `faiss_bge.index`, `faiss_qwen.index`, `bm25_index/` | Earlier variants (MiniLM, BGE, Qwen on unheaded chunks) for comparison |
| `cache.db` | API answer cache (created automatically; safe to delete) |
| `traces.db` | API request traces, kept 30 days (created automatically) |
| `answer_results.jsonl` | Every evaluation answer, tagged with model / retrieval / prompt |
| `manual_verdicts.json` | Hand-graded verdicts for free-text answers |
| `rerank_cache_*.json`, `retrieval_ranks_*.json`, `guardrail_eval.json` | Evaluation caches and outputs |

`answer_results.jsonl` and `manual_verdicts.json` hold the graded results behind the reported
79.3% and cannot be regenerated without re-running (and re-grading) the evaluation — keep a backup.
