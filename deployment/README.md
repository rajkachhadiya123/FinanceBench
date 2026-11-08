# deployment — running FilingLens in Docker

The `Dockerfile` and `docker-compose.yml` are in the repository root.

```
┌──────────────── container: filinglens-api ────────────────┐
│ Python 3.10 · CPU-only PyTorch · source/ · Qwen model      │
│ uvicorn source.API.app:app  →  port 8000                   │
│                                                            │
│ /app/data/PROCESSED  ◄── ./data/PROCESSED  (read-only)     │
│ /app/state           ◄── volume "filinglens-state"         │
│                          (cache.db, traces.db)             │
│ API keys             ◄── .env  (at run time, never baked)  │
└────────────────────────────────────────────────────────────┘
```

| In the image (~3 GB) | Outside the image |
|---|---|
| Code, Python dependencies, the Qwen query-embedding model | Indexes and chunks (`data/PROCESSED`, ~2 GB), API keys (`.env`), cache and traces (Docker volume) |

## 1. Install Docker (once)

Windows: install **Docker Desktop** (https://www.docker.com/products/docker-desktop/ or
`winget install Docker.DockerDesktop` from an administrator terminal), restart when asked,
start Docker Desktop and keep the WSL 2 backend enabled. Check in a new terminal:

```bash
docker --version
docker compose version
```

## 2. Prerequisites in the project folder

- `.env` with `CEREBRAS_API_KEY` and `COHERE_API_KEY` (copy `.env.example`).
- `data/PROCESSED/` with the headed indexes (see [data/README.md](../data/README.md)):
  `faiss_qwen_headed.index`, `chunks_headed.jsonl`, `chunk_meta_headed.jsonl`, `bm25_index_headed/`.
  If the project is in OneDrive, make sure these files are kept on the device, not online-only.

## 3. Build and start

```bash
docker compose up --build -d
```

The first build downloads PyTorch and the Qwen model (several minutes, ~3 GB). Later builds
reuse those layers; a code change rebuilds in seconds. After start, the API needs ~30–60 s to
load the model and indexes.

```bash
docker compose ps                  # STATUS shows "healthy" once the API is ready
docker compose logs -f api         # follow the logs (Ctrl+C stops following, not the API)
```

## 4. Use it

Open **http://localhost:8000/docs**, or:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d "{\"question\": \"What was Nike's total revenue in FY2023?\"}"
```

## 5. Stop, restart, update

| Task | Command |
|---|---|
| Stop | `docker compose down` (cache and traces are kept in the volume) |
| Start again | `docker compose up -d` |
| Rebuild after a code change | `docker compose up --build -d` |
| Clear cache and traces too | `docker compose down -v` (deletes the volume) |
| Disk used by images | `docker system df` |

## Settings

All optional settings from `.env.example` work in `.env` (rate limits, trace retention,
prices, model names). `FILINGLENS_CACHE_PATH` and `FILINGLENS_TRACE_PATH` are fixed by
`docker-compose.yml` to the volume inside the container.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError` for an index at startup | `data/PROCESSED` is missing files, or OneDrive keeps them online-only |
| Container exits, `CEREBRAS_API_KEY not set` / `COHERE_API_KEY not set` | `.env` missing or the keys are empty |
| Killed while loading (exit code 137) | Not enough memory for Docker: give it at least 4 GB (Docker Desktop → Settings → Resources) |
| Port 8000 already in use | Stop the locally running API, or change `"8000:8000"` to `"8001:8000"` |
| Slow first answers | Normal: the query-embedding model runs on CPU (~1 s); repeated questions are cached |

## Limits of this setup

One API process per container. The answer cache (SQLite) and the rate limiter (memory) belong
to that process, so running several replicas behind a load balancer would need a shared store
such as Redis.
