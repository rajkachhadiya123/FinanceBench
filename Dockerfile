# FilingLens API image.
#   docker compose up --build        (see docker-compose.yml and deployment/README.md)
#
# The image holds the code, the Python dependencies (CPU-only PyTorch) and the Qwen
# query-embedding model. It does NOT hold the ~2 GB of indexes (mounted read-only from
# data/PROCESSED) or any API key (passed from .env at run time).
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# OpenMP runtime needed by faiss-cpu on a slim base image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 1. Dependencies first, so code changes do not reinstall them.
#    The CPU wheel ("+cpu", only on the PyTorch index) avoids ~2 GB of CUDA libraries and
#    satisfies requirements.txt's torch pin. PyPI stays available for torch's own
#    dependencies, which the PyTorch index does not always carry as wheels.
COPY requirements.txt .
RUN pip install "torch==2.14.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
 && pip install -r requirements.txt

# Non-root user; /app/state holds cache.db and traces.db (a Docker volume).
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/state /app/data/PROCESSED \
 && chown app:app /app/state

# 2. Bake the query-embedding model into the image (~1.2 GB), so the container starts
#    without downloading it and works offline apart from the Cerebras / Cohere calls.
#    Ownership is set in the same layer: a separate chown would store the model twice.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')" \
 && chown -R app:app /opt/hf-cache

# 3. Application code. Indexes, caches and traces live outside the image.
COPY source/ source/
USER app

ENV FILINGLENS_CACHE_PATH=/app/state/cache.db \
    FILINGLENS_TRACE_PATH=/app/state/traces.db \
    HF_HUB_OFFLINE=1

EXPOSE 8000

# The model and indexes take ~30-60 s to load; health checks wait for that.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["python", "-m", "uvicorn", "source.API.app:app", "--host", "0.0.0.0", "--port", "8000"]
