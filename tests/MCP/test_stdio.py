"""
End-to-end: start the server as a subprocess and talk to it over stdio, the
same way Claude Code and the MCP Inspector do. Catches what in-process tests
cannot: a print() polluting stdout, an import error at startup, a bad __main__.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]


async def _session_round_trip():
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=ROOT)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            ok = await session.call_tool("calculator", {"expression": "2 * (3 + 4)"})
            bad = await session.call_tool("calculator", {"expression": "open('x')"})
            return tools, ok, bad


def test_server_over_stdio():
    tools, ok, bad = asyncio.run(asyncio.wait_for(_session_round_trip(), timeout=60))
    assert "calculator" in [t.name for t in tools.tools]
    assert json.loads(ok.content[0].text)["result"] == 14
    assert bad.is_error is True
    assert "not allowed" in bad.content[0].text
