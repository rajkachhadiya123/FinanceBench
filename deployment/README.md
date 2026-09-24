# 06-DOCKER: Containerization

Docker configuration for reproducible deployment.

## 📂 Files

### `Dockerfile`
Builds the application image with all dependencies.

```bash
docker build -t filinglens:latest .
docker run -p 8000:8000 filinglens:latest
```

### `docker-compose.yml` (Week 5)
One-command setup: PostgreSQL + pgvector + FastAPI + Langfuse

```bash
docker-compose up -d
```

**Containers**:
- `postgres` — Database with pgvector
- `app` — FastAPI server
- `langfuse` — Tracing (optional)

### `.dockerignore`
Exclude unnecessary files from build (node_modules, tests, etc)

## 🚀 Quick Start

**Week 5 - Run Full System**:

```bash
cd 06-DOCKER
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Database: localhost:5432
- Langfuse: http://localhost:3000 (if enabled)

## 🔧 Configuration

Environment variables in `docker-compose.yml`:

```yaml
environment:
  DATABASE_URL: postgresql://postgres:password@postgres:5432/filinglens
  GROQ_API_KEY: ${GROQ_API_KEY}
  LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY}
  LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY}
```

Set in shell before running:
```bash
export GROQ_API_KEY=your_key_here
docker-compose up
```

---

**Status**: Ready for Week 5
