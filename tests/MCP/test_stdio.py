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
    assert {"calculator", "list_filings"} <= {t.name for t in tools.tools}
    assert json.loads(ok.content[0].text)["result"] == 14
    assert bad.is_error is True
    assert "not allowed" in bad.content[0].text


def test_stdout_carries_only_protocol_messages():
    """
    Regression: PyMuPDF's "fitz API is deprecated" warning goes to stdout. Imported
    at server start, it reached the client as an unparseable protocol line. Every
    line the server writes to stdout must be JSON-RPC.
    """
    import subprocess

    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                          "clientInfo": {"name": "test", "version": "0"}}}
    proc = subprocess.Popen([sys.executable, "-m", "mcp_server.server"], cwd=ROOT,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        lines = []
        while True:
            line = proc.stdout.readline()
            assert line, f"server closed stdout; got {lines!r}"
            lines.append(line)
            message = json.loads(line)  # raises on any stray print
            if message.get("id") == 1:
                break
        assert "result" in message
    finally:
        proc.kill()
        proc.wait(timeout=30)
