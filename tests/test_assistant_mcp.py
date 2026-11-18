"""modelmaker-mcp (assistant_mcp.py): Model-Maker's chat tools for Claude Code,
against a real backend over HTTP -- the browser app's way (--url) and the
VS Code way (a discovery file) -- and once through a real stdio MCP client."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

import modelmaker.api as api_module
from modelmaker import assistant_mcp as mm

REPO = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def backend(tmp_path, monkeypatch):
    import uvicorn

    from modelmaker.session import ProjectSession

    monkeypatch.setattr(api_module, "SESSION", ProjectSession(recovery_path=tmp_path / "r.json"))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(api_module.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn didn't start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def call(resolver, tool, /, **args):
    result, is_error = mm.call(resolver, tool, args)
    assert not is_error, result
    return result


def test_builds_a_model_from_an_empty_folder(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = mm.Resolver(backend, None, discovery=tmp_path / "none")

    opened = call(r, "open_project", folder="pd_model", create=True)
    assert (tmp_path / "pd_model" / "model.json").is_file() and opened["blocks"] == []

    assert call(r, "list_block_types")["tags"][0]["tag"] == "data_sources"
    assert "entries" in call(r, "list_files", path=str(tmp_path))
    call(r, "add_lane", name="Data")
    call(r, "add_lane", name="Estimation")
    load = call(r, "add_block", category="demo_credit_data", name="Loans", lane="Data")
    split = call(r, "add_block", category="train_test_split", lane="Estimation", params={"seed": 5})
    out_port = load["outputs"][0].split(":")[0]
    call(r, "connect", from_block="Loans", from_port=out_port, to_block=split["id"], to_port="df")

    ran = call(r, "run")
    assert ran["statuses"] == {"green": 2}, ran
    columns = call(r, "get_input_columns", block=split["id"])["ports"]["df"]
    first = columns[0]["name"]
    assert call(r, "set_column_role", block=load["id"], column=first, role="excluded")["column_roles"] == {first: "excluded"}
    renamed = call(r, "set_params", block=split["id"], name="Holdout split", params={"test_size": 0.25})
    assert renamed["name"] == "Holdout split" and renamed["params"]["seed"] == 5
    assert call(r, "check")["ok"] is True
    assert "def " in call(r, "compile")["source"]

    call(r, "disconnect", to_block=split["id"], to_port="df")
    call(r, "delete_block", block="Holdout split")
    assert [b["name"] for b in call(r, "get_graph")["blocks"]] == ["Loans"]


def test_mistakes_come_back_as_errors_the_model_can_fix(backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = mm.Resolver(backend, None, discovery=tmp_path / "none")
    for name, args, message in [
        ("open_project", {"folder": "nope"}, "pass create: true"),
        ("add_block", {"category": "filter", "lane": "Nowhere"}, 'no lane "Nowhere"'),
        ("set_params", {"block": "ghost"}, 'no block "ghost"'),
        ("no_such_tool", {}, "no tool"),
    ]:
        result, is_error = mm.call(r, name, args)
        assert is_error and message in result["error"], (name, result)


def test_finds_the_vscode_canvas_used_last(backend, tmp_path):
    discovery = tmp_path / "vscode-backends"
    discovery.mkdir()
    (discovery / "1-1.json").write_text(json.dumps({"folder": "/old", "url": backend, "token": None, "active_at": 1}))
    (discovery / "1-2.json").write_text(json.dumps({"folder": "/new", "url": backend, "token": None, "active_at": 2}))
    (discovery / "1-3.json").write_text(json.dumps({"folder": "/dead", "url": "http://127.0.0.1:1", "token": None, "active_at": 3}))
    (discovery / "broken.json").write_text("{")
    r = mm.Resolver(None, None, discovery=discovery)
    target = r.target()
    assert target.vscode and target.folder == "/new"
    assert call(r, "get_graph")["folder"] == "/new"
    r.pinned = "/old"
    assert r.target().folder == "/old"


def test_same_tools_as_the_vscode_extension():
    package = json.loads((REPO / "vscode" / "package.json").read_text("utf-8"))
    vscode = {t["name"].removeprefix("modelmaker_") for t in package["contributes"]["languageModelTools"]}
    assert set(mm.TOOLS) == vscode


def test_a_real_mcp_client_lists_and_calls_the_tools(backend):
    import asyncio
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "modelmaker.assistant_mcp", "--url", backend])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            result = await session.call_tool("get_graph", {})
            return names, json.loads(result.content[0].text), result.isError

    names, graph, is_error = asyncio.run(go())
    assert "run" in names and "open_project" in names
    assert not is_error and "blocks" in graph


def test_a_whole_run_rereads_a_source_whose_settings_changed(backend, tmp_path, monkeypatch):
    # Run all alone leaves a stale source alone, and the split goes red.
    monkeypatch.chdir(tmp_path)
    r = mm.Resolver(backend, None, discovery=tmp_path / "none")
    call(r, "open_project", folder="p", create=True)
    load = call(r, "add_block", category="demo_credit_data")
    assert load["data_source"] is True
    split = call(r, "add_block", category="train_test_split")
    call(r, "connect", from_block=load["id"], from_port=load["outputs"][0].split(":")[0], to_block=split["id"], to_port="df")
    assert call(r, "run")["statuses"] == {"green": 2}
    call(r, "set_column_role", block=load["id"], column="default_flag", role="target")
    ran = call(r, "run")
    assert ran["statuses"] == {"green": 2} and ran["refreshed_sources"] == [load["name"]], ran
