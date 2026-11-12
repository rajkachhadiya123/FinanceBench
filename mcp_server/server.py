"""
FilingLens MCP server: exposes Project 1 (source/) as tools. Project 1 is
imported, never modified.

Run (stdio, which is what Claude Code and the MCP Inspector use):
    .venv\\Scripts\\python.exe -m mcp_server.server
"""

import sys
import threading

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.calculator import CalculatorError, evaluate
from mcp_server.filings import CompanyNotFound, FilingCatalog

mcp = MCPServer(
    "filinglens",
    instructions=(
        "Tools for answering questions from SEC filings (FinanceBench). "
        "Use list_filings to find a company's exact doc_name values. "
        "Use calculator for any arithmetic instead of computing in your head."
    ),
)

# Loaded on first use, not at import, so the server starts instantly; __main__
# also warms it up in the background (loading torch takes ~13 s).
# Tests replace it with a small fake catalogue (set_catalog).
_catalog: FilingCatalog | None = None
_catalog_lock = threading.Lock()  # the warm-up thread and a tool call may race


def get_catalog() -> FilingCatalog:
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = FilingCatalog.from_meta_file()
        return _catalog


def _warm_up() -> None:
    try:
        get_catalog()
    except Exception as exc:  # e.g. no data/PROCESSED (CI): the tool call reports it
        print(f"[filinglens] catalogue not loaded at startup: {exc}", file=sys.stderr)


def set_catalog(catalog: FilingCatalog | None) -> None:
    global _catalog
    _catalog = catalog


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
    catalog = get_catalog()
    if not company or not company.strip():
        return {"companies": catalog.summary()}
    try:
        return catalog.filings_of(company)
    except CompanyNotFound as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    # Start loading while the client connects; a tool call before it finishes
    # simply waits on the lock. daemon=True: never blocks shutdown.
    threading.Thread(target=_warm_up, daemon=True).start()
    mcp.run()
