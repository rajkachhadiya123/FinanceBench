# FilingLens MCP server

Project 1's search over SEC filings, exposed as MCP tools. It **imports**
`source/` and never modifies it: the Cohere reranker, metadata filter and
the rest of the pipeline run exactly as in Project 1.

| Tool | What it does | External API |
|---|---|---|
| `list_filings(company?)` | Companies, or one company's filings with exact `doc_name` values | – |
| `search_filings(query, company?, year?, doc_type?, top_k=5, rerank=True)` | Project 1 retrieval: Qwen + BM25 → RRF → Cohere rerank | Cohere: 1 call per search (none with `rerank=false`) |
| `read_page(doc_name, page, neighbours=0)` | Whole PDF page(s) from `data/RAW` | – |
| `verify_number(value, doc_name, page)` | Is this figure printed on that page? (Project 1's grounding check) | – |
| `calculator(expression)` | Safe arithmetic (`+ - * / **`, brackets; no `eval`) | – |

Pages are **0-based PDF indexes**, as in search results and Project 1's citations;
the number printed on a page can differ (3M_2018_10K page 59 is printed "60").
No tool calls an LLM.

## Run

Needs `data/PROCESSED` and `data/RAW` (see `data/README.md`) and `COHERE_API_KEY` in `.env`.

```
.venv\Scripts\python.exe -m pip install -r mcp_server/requirements.txt
.venv\Scripts\python.exe -m mcp_server.server          # stdio; normally started by the client
```

Startup takes ~2 s. Loading the catalogue and the search indexes (torch, Qwen,
FAISS: ~20 s) then runs in the background; a tool call made earlier waits for it.

## Inspect (MCP Inspector, needs Node.js)

```
npx -y @modelcontextprotocol/inspector@2.13.0 .venv/Scripts/python.exe -m mcp_server.server
```

Opens a browser UI: **Connect → Tools → List Tools**, pick a tool, fill in the
arguments, **Run Tool**. Command-line mode, for scripts:

```
npx -y @modelcontextprotocol/inspector@2.13.0 --cli .venv/Scripts/python.exe -m mcp_server.server --method tools/list
npx -y @modelcontextprotocol/inspector@2.13.0 --cli .venv/Scripts/python.exe -m mcp_server.server --method tools/call --tool-name calculator --tool-arg "expression=(1577-1373)/1373*100"
```

## Connect to Claude Code

```
claude mcp add filinglens --scope local -e "PYTHONPATH=<repo root>" -- "<repo root>\.venv\Scripts\python.exe" -m mcp_server.server
claude mcp list                                          # filinglens: ✔ Connected
```

`--scope local` keeps it private to you in this project (stored in `~/.claude.json`,
not in the repo). Start a new Claude Code session to load the tools.

## Tests

```
.venv\Scripts\python.exe -m pytest tests/MCP
```

No data, keys or network needed: a fake catalogue, tiny generated PDFs and a fake
retriever. `test_stdio.py` starts the real server process and checks that stdout
carries only JSON-RPC (a stray `print` there breaks the client).
