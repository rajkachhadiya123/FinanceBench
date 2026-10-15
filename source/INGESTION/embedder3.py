"""
FAISS index for Qwen/Qwen3-Embedding-0.6B (V3).

The chunk vectors are computed on a Colab GPU (tools/colab_embed_qwen.py) --
a 0.6B model is impractical on CPU for 107K chunks. This module holds the
model constants the retriever imports, and turns the downloaded
qwen_vectors.npy into faiss_qwen.index:

    python source/INGESTION/embedder3.py            # chunks.jsonl        -> faiss_qwen.index
    python source/INGESTION/embedder3.py --headed   # chunks_headed.jsonl -> faiss_qwen_headed.index

Contract shared with embedder.py / embedder2.py: FAISS id N == line N of
chunks.jsonl, and chunk_meta.jsonl is reused unchanged. The headed variant
(tools/add_chunk_headers.py) has the same chunks in the same order, but its
lines are longer, so it gets its own chunk_meta_headed.jsonl with byte offsets
into chunks_headed.jsonl.

Query side: Qwen3 is instruction-aware. Chunks were embedded with NO
instruction; queries must be prefixed with QUERY_PROMPT, in Qwen3's
"Instruct: ...\\nQuery:" format, or retrieval quality drops.
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "PROCESSED"
VECTORS_PATH = PROCESSED / "qwen_vectors.npy"
META_PATH = PROCESSED / "chunk_meta.jsonl"
INDEX_PATH = PROCESSED / "faiss_qwen.index"

# Chunks with context headers (V3 step 3), embedded by tools/colab_embed_qwen.py
HEADED_CHUNKS_PATH = PROCESSED / "chunks_headed.jsonl"
HEADED_VECTORS_PATH = PROCESSED / "qwen_vectors_chunks_headed.npy"
HEADED_META_PATH = PROCESSED / "chunk_meta_headed.jsonl"
HEADED_INDEX_PATH = PROCESSED / "faiss_qwen_headed.index"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
MAX_SEQ_LENGTH = 1024  # matches the Colab run; every chunk fit (no truncation)

QUERY_PROMPT = (
    "Instruct: Given a question about a company's SEC filing, "
    "retrieve the passage that contains the answer\nQuery:"
)


def write_headed_meta():
    """chunk_meta.jsonl rows with byte offsets into chunks_headed.jsonl; order verified."""
    with open(META_PATH, "rb") as f:
        meta = [json.loads(line) for line in f]
    offset = 0
    with open(HEADED_CHUNKS_PATH, "rb") as src, open(HEADED_META_PATH, "w", encoding="utf-8") as out:
        for n, line in enumerate(src):
            chunk_id = json.loads(line)["chunk_id"]
            if n >= len(meta) or meta[n]["chunk_id"] != chunk_id:
                raise SystemExit(f"Line {n} of {HEADED_CHUNKS_PATH.name} ({chunk_id}) "
                                 f"does not match {META_PATH.name}; refusing to build.")
            out.write(json.dumps({**meta[n], "byte_offset": offset}) + "\n")
            offset += len(line)
    if n + 1 != len(meta):
        raise SystemExit(f"{HEADED_CHUNKS_PATH.name} has {n + 1} lines, expected {len(meta)}.")
    print(f"Metadata written to {HEADED_META_PATH} ({len(meta)} rows, same order as {META_PATH.name})")


def main():
    headed = "--headed" in sys.argv
    vectors_path, meta_path, index_path = (
        (HEADED_VECTORS_PATH, HEADED_META_PATH, HEADED_INDEX_PATH) if headed
        else (VECTORS_PATH, META_PATH, INDEX_PATH))
    if headed:
        write_headed_meta()

    vectors = np.load(vectors_path).astype(np.float32)
    with open(meta_path, "rb") as f:
        expected_rows = sum(1 for _ in f)

    print(f"Loaded {vectors_path.name}: shape {vectors.shape}")
    if vectors.shape != (expected_rows, EMBED_DIM):
        raise SystemExit(f"Expected shape ({expected_rows}, {EMBED_DIM}); refusing to build.")
    if not np.isfinite(vectors).all():
        raise SystemExit("Vectors contain NaN/inf; refusing to build.")

    # Stored as fp16 for the download; renormalize so inner product == cosine exactly.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    print(f"FAISS index written to {index_path} ({index.ntotal} vectors, "
          f"{index_path.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
