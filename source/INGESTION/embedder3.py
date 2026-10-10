"""
FAISS index for Qwen/Qwen3-Embedding-0.6B (V3).

The chunk vectors are computed on a Colab GPU (tools/colab_embed_qwen.py) --
a 0.6B model is impractical on CPU for 107K chunks. This module holds the
model constants the retriever imports, and turns the downloaded
qwen_vectors.npy into faiss_qwen.index:

    python source/INGESTION/embedder3.py

Contract shared with embedder.py / embedder2.py: FAISS id N == line N of
chunks.jsonl, and chunk_meta.jsonl is reused unchanged.

Query side: Qwen3 is instruction-aware. Chunks were embedded with NO
instruction; queries must be prefixed with QUERY_PROMPT, in Qwen3's
"Instruct: ...\\nQuery:" format, or retrieval quality drops.
"""
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "PROCESSED"
VECTORS_PATH = PROCESSED / "qwen_vectors.npy"
META_PATH = PROCESSED / "chunk_meta.jsonl"
INDEX_PATH = PROCESSED / "faiss_qwen.index"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
MAX_SEQ_LENGTH = 1024  # matches the Colab run; every chunk fit (no truncation)

QUERY_PROMPT = (
    "Instruct: Given a question about a company's SEC filing, "
    "retrieve the passage that contains the answer\nQuery:"
)


def main():
    vectors = np.load(VECTORS_PATH).astype(np.float32)
    with open(META_PATH, "rb") as f:
        expected_rows = sum(1 for _ in f)

    print(f"Loaded {VECTORS_PATH.name}: shape {vectors.shape}")
    if vectors.shape != (expected_rows, EMBED_DIM):
        raise SystemExit(f"Expected shape ({expected_rows}, {EMBED_DIM}); refusing to build.")
    if not np.isfinite(vectors).all():
        raise SystemExit("Vectors contain NaN/inf; refusing to build.")

    # Stored as fp16 for the download; renormalize so inner product == cosine exactly.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index written to {INDEX_PATH} ({index.ntotal} vectors, "
          f"{INDEX_PATH.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
