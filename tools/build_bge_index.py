"""
Turn the bge_vectors.npy produced by tools/colab_embed_bge.ipynb into
data/PROCESSED/faiss_bge.index -- the same file embedder2.py would write.

Usage:
    python tools/build_bge_index.py [path/to/bge_vectors.npy]
Defaults to ~/Downloads/bge_vectors.npy.

Checks before writing, so a bad download never becomes a silently wrong index:
  - row count == lines in chunk_meta.jsonl (vector N must be chunk N)
  - 768 dimensions, no NaN/inf, vectors unit-length (normalized)
"""
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "PROCESSED"
META_PATH = PROCESSED / "chunk_meta.jsonl"
INDEX_PATH = PROCESSED / "faiss_bge.index"
DEFAULT_VECTORS = Path.home() / "Downloads" / "bge_vectors.npy"

EMBED_DIM = 768


def main():
    vectors_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VECTORS
    if not vectors_path.exists():
        raise SystemExit(f"Missing {vectors_path}")

    vectors = np.load(vectors_path).astype(np.float32)
    with open(META_PATH, "rb") as f:
        expected_rows = sum(1 for _ in f)

    print(f"Loaded {vectors_path.name}: shape {vectors.shape}")
    if vectors.shape != (expected_rows, EMBED_DIM):
        raise SystemExit(f"Expected shape ({expected_rows}, {EMBED_DIM}); refusing to build.")
    if not np.isfinite(vectors).all():
        raise SystemExit("Vectors contain NaN/inf; refusing to build.")

    # fp16 storage shifts norms slightly; renormalize so inner product == cosine exactly.
    norms = np.linalg.norm(vectors, axis=1)
    print(f"Norms before renormalizing: min {norms.min():.4f}, max {norms.max():.4f}")
    if norms.min() < 0.98 or norms.max() > 1.02:
        raise SystemExit("Vectors are not normalized; the Colab run did not match embedder2.py.")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index written to {INDEX_PATH} ({index.ntotal} vectors, "
          f"{INDEX_PATH.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
