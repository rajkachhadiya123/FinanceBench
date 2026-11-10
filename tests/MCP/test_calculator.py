"""calculator tool: the arithmetic itself, and the tool as an MCP client sees it."""

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.calculator import CalculatorError, evaluate
from mcp_server.server import mcp


@pytest.mark.parametrize("expression, expected", [
    ("1577 - 1373", 204),
    ("(1577 - 1373) / 1373 * 100", 14.8579752367),  # growth in percent
    ("-5 + +3", -2),
    ("2 ** 10", 1024),
    ("0.1 + 0.2", 0.3),                              # rounded: no float noise
    ("  8738 / 36500  ", 0.2393972603),
])
def test_arithmetic(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize("expression, reason", [
    ("__import__('os').remove('x')", "not allowed"),  # function call
    ("open", "not allowed"),                          # a name
    ("(1).__class__", "not allowed"),                 # attribute access
    ("'a' * 3", "Only numbers"),                      # a string
    ("True + 1", "Only numbers"),                     # bool is an int in Python
    ("1,577 + 1", "thousands separators"),            # would silently be a tuple
    ("9 ** 9 ** 9", "Exponent"),                      # would hang on a huge int
    ("10.0 ** 99 * 10.0 ** 99 * 10.0 ** 99 * 10.0 ** 99", "too large"),
    ("1 / 0", "Division by zero"),
    ("1 +", "Not a valid"),
    ("1" * 301, "longer than"),
])
def test_rejected(expression, reason):
    with pytest.raises(CalculatorError, match=reason):
        evaluate(expression)


def call(name, arguments):
    return asyncio.run(mcp.call_tool(name, arguments))


def test_tool_is_listed_with_its_schema():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert tools["calculator"].input_schema["required"] == ["expression"]


def test_tool_returns_result():
    result = call("calculator", {"expression": "(1577 - 1373) / 1373 * 100"})
    assert result.is_error is False
    assert json.loads(result.content[0].text) == {
        "expression": "(1577 - 1373) / 1373 * 100", "result": 14.8579752367}


def test_tool_error_message_reaches_the_model():
    # A ToolError (not a generic crash) means the client is told *why*.
    with pytest.raises(ToolError, match="thousands separators"):
        call("calculator", {"expression": "1,577 * 2"})
