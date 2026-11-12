"""
FilingLens MCP server: exposes Project 1 (source/) as tools. Project 1 is
imported, never modified.

Run (stdio, which is what Claude Code and the MCP Inspector use):
    .venv\\Scripts\\python.exe -m mcp_server.server

    list_filings    which filings exist (exact doc_name values)
    search_filings  Project 1's hybrid search + Cohere rerank, with optional filters
    read_page       a whole PDF page (0-based), optionally with neighbours
    verify_number   is this value printed on that page?
    calculator      safe arithmetic

Only search_filings calls an external API (Cohere rerank, 1 call per search
unless rerank=False). No tool calls an LLM.
"""

import sys
import threading
from contextlib import asynccontextmanager
from typing import Callable, Generic, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.calculator import CalculatorError, evaluate
from mcp_server.filings import CompanyNotFound, FilingCatalog
from mcp_server.pages import PageError, PageReader
from mcp_server.verify import find_value

@asynccontextmanager
async def _lifespan(_server):
    # Runs once the SDK owns stdin/stdout, so Project 1's print()s land on stderr.
    # daemon=True: never blocks shutdown. A tool call meanwhile waits on the lock.
    threading.Thread(target=_warm_up, daemon=True).start()
    yield {}


mcp = MCPServer(
    "filinglens",
    lifespan=_lifespan,
    instructions=(
        "Tools for answering questions from SEC filings (FinanceBench: 368 filings, 41 companies). "
        "Typical flow: search_filings to find the passage -> read_page for the full table -> "
        "calculator for any arithmetic -> verify_number for each figure you cite. "
        "Use list_filings to get exact doc_name values. Page numbers are 0-based PDF indexes."
    ),
)

T = TypeVar("T")


class Lazy(Generic[T]):
    """Built on first use (loading torch + indexes takes seconds), once, thread-safely."""

    def __init__(self, build: Callable[[], T]):
        self._build = build
        self._value: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        with self._lock:
            if self._value is None:
                self._value = self._build()
            return self._value

    def set(self, value: T | None) -> None:
        """Tests put small fakes here."""
        with self._lock:
            self._value = value


def _load_retriever():
    from mcp_server.search import load_retriever  # imports torch: keep it off the startup path
    return load_retriever()


catalog: Lazy[FilingCatalog] = Lazy(FilingCatalog.from_meta_file)
pages: Lazy[PageReader] = Lazy(lambda: PageReader(catalog.get().doc_names))
retriever = Lazy(_load_retriever)
# The retriever seeks in a shared file and reuses a buffer: one search at a time.
_search_lock = threading.Lock()


@mcp.tool()
def list_filings(company: str | None = None) -> dict:
    """
    List the SEC filings available (FinanceBench: 368 filings, 41 companies).

    Without a company: one row per company (name, number of filings, types,
    fiscal-year range). With a company: every filing of that company, with the
    exact doc_name to pass to search_filings, read_page and verify_number.
    Names are matched loosely: "AmEx", "Pepsi", "J&J" and "3M" all work.
    Filing types: 10K (annual), 10Q (quarterly), 8K (event), EARNINGS (call/release).
    """
    if not company or not company.strip():
        return {"companies": catalog.get().summary()}
    try:
        return catalog.get().filings_of(company)
    except CompanyNotFound as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def search_filings(query: str, company: str | None = None, year: int | None = None,
                   doc_type: str | None = None, top_k: int = 5, rerank: bool = True) -> dict:
    """
    Search the filings for passages that answer a question (hybrid semantic +
    keyword search, then Cohere reranking -- Project 1's retrieval).

    query: the question or what to look for, e.g. "3M capital expenditure 2018".
    company / year / doc_type: optional filters. year is the FISCAL year in the
      filing name (a FY2018 10-K reports 2018, 2017 and 2016 figures). doc_type is
      10K, 10Q, 8K or EARNINGS. With no filters, the company and year are read from
      the query; if no company is recognised, all filings are searched.
    top_k: passages to return (1-20). rerank: false skips Cohere (faster, uses no
      API quota, but the order is less accurate).
    Each result has doc_name and page_start/page_end (0-based) for read_page.
    """
    from mcp_server.search import NoFilings, search_filings as run_search

    try:
        cat, ret = catalog.get(), retriever.get()
        with _search_lock:
            return run_search(ret, cat, query, company, year, doc_type, top_k, rerank)
    except (CompanyNotFound, NoFilings, ValueError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def read_page(doc_name: str, page: int, neighbours: int = 0) -> dict:
    """
    Read a whole page of a filing, e.g. to see a full table that search returned
    only part of.

    doc_name: exact name from list_filings or a search result (case-insensitive).
    page: 0-based PDF page index, as in search results (page_start) and citations.
    neighbours: also return this many pages before and after (0-2), for tables
      that continue on the next page.
    """
    try:
        return pages.get().read(doc_name, page, neighbours)
    except PageError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def verify_number(value: float, doc_name: str, page: int) -> dict:
    """
    Check whether a figure is printed on a page, before citing it.

    value: the number as you will state it, e.g. 1577 or 1.577 (billions) or 8.7.
    Sign is ignored, units may differ (1.577 billion matches "1,577" in a table
    in millions), and rounding is allowed (0.5% tolerance). Each match comes with
    the text just before it -- usually its row label -- so check it is the right line.
    Calculated results (ratios, growth rates) are usually NOT printed: verify their
    inputs instead.
    """
    try:
        name, text = pages.get().page_text(doc_name, page)
    except PageError as exc:
        raise ToolError(str(exc)) from exc
    return {"value": value, "doc_name": name, "page": page, **find_value(value, text)}


@mcp.tool()
def calculator(expression: str) -> dict:
    """
    Evaluate an arithmetic expression safely.

    Allowed: numbers, + - * / ** and brackets. No thousands separators
    (write 1577, not 1,577) and no functions or variables.
    Example: "(1577 - 1373) / 1373 * 100" -> 14.857975237 (percent growth).
    """
    try:
        return {"expression": expression, "result": evaluate(expression)}
    except CalculatorError as exc:
        # ToolError = "a failure we expected": the model sees the message and can retry.
        # Any other exception would reach it only as "Error executing tool calculator".
        raise ToolError(str(exc)) from exc


def _warm_up() -> None:
    """Load everything while the client connects; a tool call meanwhile waits on the lock."""
    for name, lazy in (("catalogue", catalog), ("retriever", retriever)):
        try:
            lazy.get()
        except Exception as exc:  # e.g. no data/PROCESSED in CI: the tool call reports it
            print(f"[filinglens] {name} not loaded at startup: {exc}", file=sys.stderr)


if __name__ == "__main__":
    mcp.run()
