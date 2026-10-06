"""
Dense embedding + FAISS index builder using BAAI/bge-base-en-v1.5.

Same contract as embedder.py (FAISS id N == line N of chunks.jsonl), but a
stronger model written to a separate index so both can be evaluated side by
side:
  - faiss_bge.index : 768-dim vectors from bge-base-en-v1.5

Why this model over all-MiniLM-L6-v2:
  - Reads 512 tokens instead of 256. Our ~2000-char chunks are ~450-500
    tokens, so MiniLM silently dropped roughly the back half of every chunk.
  - Trained for query -> passage retrieval, not general sentence similarity.

chunk_meta.jsonl is not rewritten: its contents depend only on chunks.jsonl,
not on the model, so the file embedder.py produced is reused as-is.

IMPORTANT for the query side: BGE expects queries (not passages) to carry
QUERY_PREFIX. The retriever must prepend it when encoding questions, and must
load the same MODEL_NAME, or similarity scores are meaningless.
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "PROCESSED" / "chunks.jsonl"
INDEX_PATH = ROOT / "data" / "PROCESSED" / "faiss_bge.index"
SHARD_DIR = ROOT / "data" / "PROCESSED" / "bge_shards"
SHARD_SIZE = 4096  # chunks per checkpoint; ~10-25 min of CPU work each
TOTAL_CHUNKS_HINT = 107049  # for the time-remaining estimate only; not used for correctness

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
MAX_SEQ_LENGTH = 512  # the model's hard limit; set explicitly so it is never silently lower
BATCH_SIZE = 64  # smaller than embedder.py: 512-token inputs on a 110M model use far more memory

# Applied to queries only, never to chunks. Defined here so the retriever can
# import it instead of retyping the string.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def iter_chunk_texts(path: Path):
    """Yield each chunk's text in file order, so vector N matches line N."""
    with open(path, "rb") as f:
        for raw_line in f:
            yield json.loads(raw_line.decode("utf-8"))["text"]


def main():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}. Run chunker.py first.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"Device: {model.device} | max_seq_length: {model.max_seq_length}")

    # Embedding takes hours on CPU, so work is saved in shards of SHARD_SIZE
    # chunks. A rerun skips every shard already on disk and resumes after it.
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    done_shards = len(_finished_shards())
    skip = done_shards * SHARD_SIZE
    if skip:
        print(f"Resuming: {done_shards} shards ({skip} chunks) already on disk")

    print(f"Reading chunks from {CHUNKS_PATH.name} ...")
    start_time = time.time()

    shard_texts = []
    shard_index = done_shards
    embedded_this_run = 0

    for position, text in enumerate(iter_chunk_texts(CHUNKS_PATH)):
        if position < skip:
            continue
        shard_texts.append(text)
        if len(shard_texts) == SHARD_SIZE:
            embedded_this_run += _write_shard(model, shard_texts, shard_index, start_time,
                                              embedded_this_run, skip)
            shard_index += 1
            shard_texts = []

    if shard_texts:
        embedded_this_run += _write_shard(model, shard_texts, shard_index, start_time,
                                          embedded_this_run, skip)

    print(f"\nEmbedded {embedded_this_run} chunks this run in {time.time() - start_time:.0f}s")

    vector_parts, truncated = [], 0
    for shard_path in _finished_shards():
        with np.load(shard_path) as shard:
            vector_parts.append(shard["vectors"])
            truncated += int(shard["truncated"])
    vectors = np.vstack(vector_parts).astype(np.float32)
    del vector_parts
    total = len(vectors)

    print(f"Chunks over {MAX_SEQ_LENGTH} tokens (tail truncated): "
          f"{truncated} ({truncated / total:.1%})")
    print(f"Vector array shape: {vectors.shape} ({vectors.nbytes / 1024**2:.0f} MB)")

    # Vectors are L2-normalized, so inner product == cosine similarity.
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"FAISS index written to {INDEX_PATH} ({index.ntotal} vectors)")
    print(f"Shards in {SHARD_DIR} can be deleted once the index is verified.")


def _finished_shards() -> list[Path]:
    # Five digits exactly, so a leftover "shard_00003.tmp.npz" is never counted.
    return sorted(SHARD_DIR.glob("shard_[0-9][0-9][0-9][0-9][0-9].npz"))


def _write_shard(model: SentenceTransformer, texts: list[str], shard_index: int,
                 start_time: float, embedded_before: int, skipped: int) -> int:
    """Embed one shard in batches, save it, and return how many chunks it held."""
    vector_batches, truncated = [], 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        truncated += _count_truncated(model, batch)
        vector_batches.append(_embed(model, batch))
        _report(skipped + embedded_before + i + len(batch), embedded_before + i + len(batch),
                start_time)

    # Write to a temp name then rename, so an interrupted save never leaves a
    # truncated shard that a resume would trust.
    final_path = SHARD_DIR / f"shard_{shard_index:05d}.npz"
    tmp_path = SHARD_DIR / f"shard_{shard_index:05d}.tmp.npz"
    np.savez(tmp_path, vectors=np.vstack(vector_batches), truncated=truncated)
    tmp_path.replace(final_path)
    return len(texts)


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # No prefix here: BGE's instruction is for queries only.
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == inner product
        show_progress_bar=False,
    )


def _count_truncated(model: SentenceTransformer, texts: list[str]) -> int:
    lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)["input_ids"]]
    return sum(length > MAX_SEQ_LENGTH for length in lengths)


def _report(done_overall: int, done_this_run: int, start_time: float):
    if done_this_run % (BATCH_SIZE * 16) == 0:
        elapsed = time.time() - start_time
        rate = done_this_run / elapsed if elapsed else 0
        remaining = (TOTAL_CHUNKS_HINT - done_overall) / rate if rate else 0
        print(f"  {done_overall}/{TOTAL_CHUNKS_HINT} chunks | {rate:.1f}/s | "
              f"{elapsed / 60:.0f} min elapsed | ~{remaining / 3600:.1f} h left", flush=True)


if __name__ == "__main__":
    main()
