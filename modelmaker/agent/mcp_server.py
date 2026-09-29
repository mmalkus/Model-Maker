"""Stdio MCP server the claude CLI loop (see loop.ClaudeCliLoop) talks to.
It holds no state and runs no tools itself: it lists the current build's
tools and forwards each call to the Model-Maker API server over HTTP,
where AgentBuild.call_tool applies the same guards, limits and logging as
for any other loop. Authenticated with the per-build token the API
generated, so it can only ever act on the build it was started for.

    python -m modelmaker.agent.mcp_server --url http://127.0.0.1:8000 --token <token>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# A tool call can include a model fit on real data -- well beyond any
# normal HTTP timeout.
CALL_TIMEOUT_SECONDS = 30 * 60


def _request(url: str, token: str, payload: dict[str, Any] | None = None, timeout: float = 60) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", "X-Agent-Token": token},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        return {"error": f"Model-Maker API {e.code}: {detail}"}
    except urllib.error.URLError as e:
        return {"error": f"can't reach Model-Maker at {url}: {e.reason}"}


def build_server(base_url: str, token: str) -> Server:
    server = Server("modelmaker")
    base = base_url.rstrip("/") + "/api/agent/mcp"

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools = await asyncio.to_thread(_request, f"{base}/tools", token)
        if isinstance(tools, dict) and "error" in tools:
            raise RuntimeError(tools["error"])
        return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"]) for t in tools]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        result = await asyncio.to_thread(
            _request, f"{base}/call", token, {"name": name, "args": arguments or {}}, CALL_TIMEOUT_SECONDS
        )
        is_error = isinstance(result, dict) and "error" in result
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, default=str))], isError=is_error
        )

    return server


async def _main(base_url: str, token: str) -> None:
    server = build_server(base_url, token)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="Model-Maker API base URL")
    parser.add_argument("--token", required=True, help="the build's agent token")
    args = parser.parse_args()
    asyncio.run(_main(args.url, args.token))


if __name__ == "__main__":
    main()
