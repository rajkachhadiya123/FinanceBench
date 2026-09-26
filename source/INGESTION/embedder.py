"""
Dense embedding + FAISS index builder.

Reads data/PROCESSED/chunks.jsonl, embeds each chunk's text with a local
sentence-transformers model, and writes a FAISS index for similarity search.

Two outputs, and their ordering is the contract that ties everything together:
  - faiss.index      : the vectors; FAISS id N == line N of chunks.jsonl
  - chunk_meta.jsonl : line N holds metadata for FAISS id N, plus the byte
                       offset of that chunk in chunks.jsonl so its full text
                       can be seek-read on demand instead of held in memory.

bm25_builder.py streams the same file in the same order, so position N refers
to the same chunk in both indexes.
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "PROCESSED" / "chunks.jsonl"
INDEX_PATH = ROOT / "data" / "PROCESSED" / "faiss.index"
META_PATH = ROOT / "data" / "PROCESSED" / "chunk_meta.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 256


def iter_chunks_with_offsets(path: Path):
    """Yield (byte_offset, chunk_dict) for each line, reading in binary so offsets are exact."""
    with open(path, "rb") as f:
        offset = 0
        for raw_line in f:
            yield offset, json.loads(raw_line.decode("utf-8"))
            offset += len(raw_line)


def main():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}. Run chunker.py first.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Reading chunks from {CHUNKS_PATH.name} ...")
    start_time = time.time()

    vector_batches = []
    batch_texts = []
    total = 0

    with open(META_PATH, "w", encoding="utf-8") as meta_f:
        for byte_offset, chunk in iter_chunks_with_offsets(CHUNKS_PATH):
            meta_f.write(
                json.dumps(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_name": chunk["doc_name"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "byte_offset": byte_offset,
                    }
                )
                + "\n"
            )
            batch_texts.append(chunk["text"])

            if len(batch_texts) >= BATCH_SIZE:
                vector_batches.append(_embed(model, batch_texts))
                total += len(batch_texts)
                batch_texts = []
                _report(total, start_time)

        if batch_texts:
            vector_batches.append(_embed(model, batch_texts))
            total += len(batch_texts)
            _report(total, start_time)

    print(f"\nEmbedded {total} chunks in {time.time() - start_time:.0f}s")

    vectors = np.vstack(vector_batches).astype(np.float32)
    del vector_batches
    print(f"Vector array shape: {vectors.shape} ({vectors.nbytes / 1024**2:.0f} MB)")

    # Vectors are L2-normalized, so inner product == cosine similarity.
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"FAISS index written to {INDEX_PATH} ({index.ntotal} vectors)")
    print(f"Metadata written to {META_PATH}")


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == inner product
        show_progress_bar=False,
    )


def _report(total: int, start_time: float):
    if total % (BATCH_SIZE * 20) == 0:
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed else 0
        print(f"  {total} chunks | {rate:.0f}/s | {elapsed:.0f}s elapsed")


if __name__ == "__main__":
    main()
