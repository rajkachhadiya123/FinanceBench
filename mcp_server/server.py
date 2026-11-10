"""
FilingLens MCP server: exposes Project 1 (source/) as tools. Project 1 is
imported, never modified.

Run (stdio, which is what Claude Code and the MCP Inspector use):
    .venv\\Scripts\\python.exe -m mcp_server.server
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.calculator import CalculatorError, evaluate

mcp = MCPServer(
    "filinglens",
    instructions=(
        "Tools for answering questions from SEC filings (FinanceBench). "
        "Use calculator for any arithmetic instead of computing in your head."
    ),
)


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


if __name__ == "__main__":
    mcp.run()
