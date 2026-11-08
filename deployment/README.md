# deployment

Reserved for **Docker** packaging — not built yet.

Today the API runs directly with Python (see the [root README](../README.md#quick-start)):

```bash
python -m uvicorn source.API.app:app --port 8000
```

The plan for a container:

- a `Dockerfile` with the Python dependencies (CPU-only PyTorch) and the `source/` code;
- `data/PROCESSED/` (~2 GB of indexes) **mounted as a volume**, not copied into the image;
- API keys passed from `.env` at run time (`--env-file`), never baked into the image;
- `cache.db` and `traces.db` on the mounted volume so they survive container restarts;
- one API process per container — the SQLite cache and the in-memory rate limiter are per
  process, so several replicas would need a shared store (e.g. Redis).
