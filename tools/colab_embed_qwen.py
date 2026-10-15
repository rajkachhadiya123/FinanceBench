"""
Embed every chunk with Qwen/Qwen3-Embedding-0.6B on a Google Colab GPU (V3).

Paste this whole file into ONE Colab cell and run it. Steps are in
documentation/experiment_log/README.md (V3) and below.

  Set INPUT_NAME below to the chunk file being embedded:
     "chunks_headed" -> data/PROCESSED/chunks_headed.zip (chunks with context headers, V3 step 3)
     "chunks"        -> data/PROCESSED/chunks.zip         (original chunks, V3 step 1)
  Outputs are named after it, so runs never overwrite or resume each other.

  1. colab.research.google.com -> New notebook
     Runtime -> Change runtime type -> T4 GPU -> Save.
  2. Provide <INPUT_NAME>.zip in ONE of two ways:
       A. Drag it into Colab's Files panel (folder icon, left) -> lands in /content.
          Simple, but if Colab disconnects everything is lost and you start over.
       B. Upload it to the top level of Google Drive ("My Drive").
          Progress is saved to My Drive/qwen_shards_<INPUT_NAME>/, so a disconnect just resumes.
  3. Paste this file into a cell and run it (option B asks for Drive access).
  4. Wait ~45-90 min.
  5. When it prints DONE: option A downloads qwen_vectors_<INPUT_NAME>.npy (~210 MB)
     in the browser; option B saves it to My Drive. Put it in data/PROCESSED/ on your PC.

Must stay consistent with how the retriever will query the index:
  - chunks are embedded with NO instruction (Qwen3 uses the instruction on queries only)
  - vectors L2-normalized, so FAISS inner product == cosine similarity
  - vector N == line N of chunks.jsonl (same order as chunk_meta.jsonl)
"""
import subprocess
import sys

# Qwen3-Embedding needs recent transformers / sentence-transformers.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "sentence-transformers>=3.0", "transformers>=4.51"], check=True)

import gc  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from google.colab import drive  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

INPUT_NAME = "chunks_headed"  # <-- the only setting to change between runs

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
# Qwen3 accepts 32K tokens; 1024 covers every chunk (BGE's 512 truncated some at ~635)
# while keeping T4 memory and time reasonable.
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 32
SHARD_SIZE = 8192
EXPECTED_CHUNKS = 107049  # lines in chunks.jsonl / chunk_meta.jsonl
TOKEN_CHECK_SAMPLE = 2000  # chunks tokenized up front to report lengths

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again."
print("GPU:", torch.cuda.get_device_name(0))

# Option A (file uploaded into Colab) wins if present; otherwise option B (Drive).
LOCAL_ZIP = Path(f"/content/{INPUT_NAME}.zip")
USE_DRIVE = not LOCAL_ZIP.exists()
if USE_DRIVE:
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive")
    print("Using Google Drive (resumable).")
else:
    BASE = Path("/content")
    print(f"Using the uploaded {LOCAL_ZIP} -- keep this tab open; a disconnect loses progress.")
ZIP_PATH = BASE / f"{INPUT_NAME}.zip"
# Separate names per input, so this run never resumes from another run's shards.
SHARD_DIR = BASE / ("qwen_shards" if INPUT_NAME == "chunks" else f"qwen_shards_{INPUT_NAME}")
OUT_PATH = BASE / ("qwen_vectors.npy" if INPUT_NAME == "chunks" else f"qwen_vectors_{INPUT_NAME}.npy")
assert ZIP_PATH.exists(), (f"{INPUT_NAME}.zip not found: drag it into the Files panel, "
                           "or upload it to the top level of My Drive")
SHARD_DIR.mkdir(exist_ok=True)

with zipfile.ZipFile(ZIP_PATH) as zf:
    name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
    with zf.open(name) as f:
        texts = [json.loads(line)["text"] for line in f]  # file order preserved
print(f"Loaded {len(texts)} chunks from {name}")
assert len(texts) == EXPECTED_CHUNKS, f"Expected {EXPECTED_CHUNKS} chunks: wrong or stale {INPUT_NAME}.zip"
print("First chunk starts with:", texts[0][:100].replace("\n", " / "))

model = SentenceTransformer(
    MODEL_NAME, device="cuda",
    model_kwargs={"torch_dtype": torch.float16},   # T4 has no bf16; fp16 halves time and memory
    tokenizer_kwargs={"padding_side": "left"},     # recommended for Qwen3 last-token pooling
)
model.max_seq_length = MAX_SEQ_LENGTH
assert model.get_sentence_embedding_dimension() == EMBED_DIM

# Token-length check on a sample only: tokenizing all 107K chunks in one call
# holds every token's Python/Rust objects at once and exhausts Colab's ~12 GB RAM.
sample = random.Random(0).sample(texts, TOKEN_CHECK_SAMPLE)
lengths = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in sample]
over = sum(n > MAX_SEQ_LENGTH for n in lengths)
print(f"Tokens per chunk ({TOKEN_CHECK_SAMPLE} sampled): median {int(np.median(lengths))}, "
      f"max {max(lengths)} | over {MAX_SEQ_LENGTH}: {over} ({over / len(sample):.2%})")
del sample, lengths
gc.collect()

n_shards = (len(texts) + SHARD_SIZE - 1) // SHARD_SIZE
start = time.time()
for s in range(n_shards):
    path = SHARD_DIR / f"shard_{s:03d}.npy"
    if path.exists():
        continue  # done in an earlier (disconnected) run
    part = texts[s * SHARD_SIZE:(s + 1) * SHARD_SIZE]
    # No prompt: Qwen3's instruction belongs on queries only.
    vecs = model.encode(part, batch_size=BATCH_SIZE, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    np.save(path, vecs.astype(np.float16))  # fp16 halves the download; renormalized locally
    print(f"  shard {s + 1}/{n_shards} saved | {(time.time() - start) / 60:.1f} min", flush=True)

vectors = np.concatenate([np.load(SHARD_DIR / f"shard_{s:03d}.npy") for s in range(n_shards)])
assert vectors.shape == (EXPECTED_CHUNKS, EMBED_DIM), vectors.shape
assert np.isfinite(vectors).all()
np.save(OUT_PATH, vectors)
print(f"DONE: {OUT_PATH} {vectors.shape} ({OUT_PATH.stat().st_size / 1024**2:.0f} MB). "
      "Put it in data/PROCESSED/ on your PC.")
if not USE_DRIVE:
    from google.colab import files
    files.download(str(OUT_PATH))  # browser download; if it stalls, use the Files panel's Download
