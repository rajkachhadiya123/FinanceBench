# 10-PROJECT: Project Files

Configuration and metadata for the entire project.

## 📂 Files (to be created)

### `pyproject.toml`
Python project configuration and dependencies.

**Content**:
```toml
[project]
name = "filinglens"
version = "0.1.0"
description = "Production-grade RAG system for financial Q&A"

[project.dependencies]
fastapi = "^0.104"
pydantic = "^2.0"
psycopg2-binary = "^2.9"
sentence-transformers = "^2.2"
groq = "^0.5"
# ... more dependencies
```

**Usage**:
```bash
pip install -e .              # Install in dev mode
pip install -e ".[dev]"       # With dev dependencies
```

### `.env.example`
Template for environment variables.

**Content**:
```
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/filinglens

# LLM API
GROQ_API_KEY=your_key_here

# Tracing (optional)
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key

# Application
LOG_LEVEL=INFO
DEBUG=false
```

**Usage**:
```bash
cp .env.example .env
# Edit .env with your keys
source .env  # or: set -a && source .env && set +a (bash)
```

### `.gitignore`
Files to exclude from git.

**Content**:
```
# Virtual environments
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# Data (too large)
01-DATA/RAW/
01-DATA/PROCESSED/

# Environment
.env
.env.local

# Python
__pycache__/
*.pyc
*.pyo
dist/
build/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
*.log
```

### `requirements.txt` (or `requirements-dev.txt`)
Frozen dependencies for reproducibility.

**Generated from pyproject.toml**:
```bash
pip freeze > requirements.txt
```

### `CLAUDE.md` (Optional)
Instructions for Claude Code / AI assistance on this project.

**Content**:
- Project context and goals
- Code style preferences
- Common commands
- How to run tests/evaluation
- Known issues and TODOs

---

**Status**: Create Week 0
