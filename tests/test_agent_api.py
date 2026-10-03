"""Phase 2 of the AI model builder: the HTTP surface -- build endpoints, the
canvas lock, plan/build LLM settings, and the MCP bridge the claude CLI
loop uses (exercised for real: a stdio MCP client talking to
agent/mcp_server.py, which calls back into a live API server)."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelmaker import api
from modelmaker.agent.controller import BuildController
from modelmaker.agent.loop import ScriptedLoop
from modelmaker.llm.settings import LLMSettingsStore
from modelmaker.runslot import RunSlot
from modelmaker.session import ProjectSession

DATA = Path(__file__).resolve().parents[1] / "sample_data" / "credit_risk_data.csv"
ROOT = Path(__file__).resolve().parents[1]


def _plan(anchor):
    def turn(call):
        r = call(
            "submit_plan",
            {
                "plan": {
                    "summary": "split",
                    "stages": [{"key": "est", "name": "Estimation", "goal": "70/30 holdout split of the anchor"}],
                }
            },
        )
        assert r.get("ok"), r
        return "planned"

    return turn


def _ask(call):
    call("ask_user", {"question": "Continue?"})
    return ""


@pytest.fixture
def client(tmp_path, monkeypatch):
    session = ProjectSession(recovery_path=tmp_path / "r.json")
    slot = RunSlot()
    scripts: dict[str, list] = {}

    def factory(choice, phase):
        return ScriptedLoop(scripts.get(phase, []))

    monkeypatch.setattr(api, "SESSION", session)
    monkeypatch.setattr(api, "RUN_SLOT", slot)
    monkeypatch.setattr(api, "AGENT", BuildController(lambda: api.SESSION, slot, factory))
    monkeypatch.setattr(api, "LLM_SETTINGS", LLMSettingsStore(active_provider="claude_cli"))
    c = TestClient(api.app)
    load = c.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(DATA)}}).json()
    assert c.post(f"/api/blocks/{load['id']}/run").json()["status"] == "green"
    c.post(f"/api/blocks/{load['id']}/column_role", json={"column": "default_flag", "role": "target"})
    c.post(f"/api/blocks/{load['id']}/column_role", json={"column": "interest_rate", "role": "excluded"})
    c.post(f"/api/blocks/{load['id']}/run")
    c.anchor = load["id"]
    c.scripts = scripts
    return c


def _wait(c, *phases, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = c.get("/api/agent/builds/current").json()["build"]
        if b["phase"] in phases:
            return b
        time.sleep(0.1)
    raise AssertionError(f"build never reached {phases}: {b['phase']}")


def test_build_endpoints_and_canvas_lock(client):
    c = client
    c.scripts.update({"plan": [_plan(c.anchor)], "build": [_ask]})
    started = c.post("/api/agent/builds", json={"goal": "split it", "anchors": [c.anchor]})
    assert started.status_code == 200, started.text
    b = _wait(c, "awaiting_approval")
    assert b["plan"]["lane_layout"]["est"]["name"] == "Estimation"
    # Nothing's locked while planning/reviewing.
    assert c.post("/api/blocks", json={"category": "filter", "params": {"expr": "dti > 0"}}).status_code == 200

    assert c.post("/api/agent/builds/current/approve").status_code == 200
    b = _wait(c, "awaiting_input")
    assert b["pending_question"] == "Continue?"
    state = c.get("/api/agent/builds/current").json()
    assert state["canvas_locked"] is True
    # Edits, runs and undo are refused while the build holds the canvas...
    assert c.post("/api/blocks", json={"category": "filter"}).status_code == 409
    assert c.post("/api/undo").status_code == 409
    assert c.post("/api/run_all").status_code == 409
    # ...reads and settings aren't.
    assert c.get("/api/graph").status_code == 200
    assert c.put("/api/llm/settings", json={}).status_code == 200
    # Second build can't start.
    assert c.post("/api/agent/builds", json={"goal": "again", "anchors": [c.anchor]}).status_code == 409

    stopped = c.post("/api/agent/builds/current/stop").json()
    assert stopped["phase"] == "stopped"
    assert c.post("/api/blocks", json={"category": "filter"}).status_code == 200


def test_events_are_paged_by_cursor(client):
    c = client
    c.scripts.update({"plan": [_plan(c.anchor)]})
    c.post("/api/agent/builds", json={"goal": "split it", "anchors": [c.anchor]})
    b = _wait(c, "awaiting_approval")
    n = b["next_cursor"]
    assert n == len(b["events"]) > 0
    again = c.get(f"/api/agent/builds/current?cursor={n}").json()["build"]
    assert again["events"] == [] and again["next_cursor"] == n


def test_mcp_endpoints_need_the_build_token(client):
    c = client
    c.scripts.update({"plan": [_plan(c.anchor)]})
    c.post("/api/agent/builds", json={"goal": "split it", "anchors": [c.anchor]})
    _wait(c, "awaiting_approval")
    assert c.get("/api/agent/mcp/tools").status_code == 403
    assert c.get("/api/agent/mcp/tools", headers={"X-Agent-Token": "wrong"}).status_code == 403
    token = api.AGENT.token
    tools = c.get("/api/agent/mcp/tools", headers={"X-Agent-Token": token}).json()
    # Awaiting approval, the model has no tools at all -- the planner is done
    # and the builder hasn't started.
    assert tools == []
    r = c.post("/api/agent/mcp/call", json={"name": "get_graph", "args": {}}, headers={"X-Agent-Token": token}).json()
    assert "isn't available" in r["error"]


def test_agent_llm_settings(client):
    c = client
    s = c.get("/api/llm/settings").json()["agent"]
    assert s["plan"]["provider"] == "claude_cli" and not s["plan"]["provider_set"]
    assert "claude_cli" in s["capable_providers"]
    bad = c.put("/api/llm/settings", json={"agent_plan": {"provider": "stub"}})
    assert bad.status_code == 400
    ok = c.put("/api/llm/settings", json={"agent_plan": {"provider": "anthropic", "model": "claude-opus-5"}}).json()["agent"]
    assert ok["plan"] == {"provider": "anthropic", "model": "claude-opus-5", "provider_set": True, "model_set": True}
    assert ok["build"]["provider"] == "claude_cli"  # independent
    cleared = c.put("/api/llm/settings", json={"agent_plan": {"provider": None, "model": None}}).json()["agent"]
    assert cleared["plan"]["provider_set"] is False


def test_start_refuses_providers_that_cant_drive_a_build(client, monkeypatch):
    c = client
    monkeypatch.setattr(api, "LLM_SETTINGS", LLMSettingsStore(active_provider="stub"))
    r = c.post("/api/agent/builds", json={"goal": "x", "anchors": [c.anchor]})
    assert r.status_code == 400 and "can't drive" in r.json()["detail"]
    r = c.post(
        "/api/agent/builds",
        json={"goal": "x", "anchors": [c.anchor], "plan_llm": {"provider": "claude_cli"}, "build_llm": {"provider": "claude_cli"}},
    )
    assert r.status_code == 200


def test_block_output_carries_provenance(client):
    c = client
    bid = c.anchor
    assert c.get("/api/graph").json()["blocks"][bid]["provenance"] is None


# ---- the MCP bridge, for real ---------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(client):
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn didn't start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def test_mcp_bridge_drives_the_tool_layer(client, live_server):
    """A real MCP client -> stdio mcp_server.py -> HTTP -> AgentBuild.call_tool
    -> the session, i.e. exactly the path the claude CLI loop uses."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    c = client
    c.scripts.update({"plan": [lambda call: "(the MCP client plans instead)"]})
    c.post("/api/agent/builds", json={"goal": "split it", "anchors": [c.anchor]})
    b = _wait(c, "awaiting_approval")
    # Put the build back into planning so the MCP client can act as the planner.
    api.AGENT.build.set_phase("planning")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "modelmaker.agent.mcp_server", f"--url={live_server}", f"--token={api.AGENT.token}"],
        env={"PYTHONPATH": str(ROOT)},
    )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                graph = await session.call_tool("get_graph", {})
                summary = await session.call_tool("get_output_summary", {"block": c.anchor})
                bad = await session.call_tool("add_block", {"category": "filter", "lane": "x"})
                plan = await session.call_tool(
                    "submit_plan",
                    {"plan": {"summary": "s", "stages": [{"key": "est", "name": "Estimation", "goal": "split"}]}},
                )
                return tools, graph, summary, bad, plan

    tools, graph, summary, bad, plan = asyncio.run(run())
    assert {"get_graph", "submit_plan", "describe_block_type"} <= tools and "add_block" not in tools
    assert c.anchor in graph.content[0].text
    s = json.loads(summary.content[0].text)
    assert s["row_count"] == 5000 and "West" not in summary.content[0].text
    assert bad.isError and "isn't available in the planning phase" in bad.content[0].text
    assert json.loads(plan.content[0].text)["ok"] is True
    assert api.AGENT.build.phase == "awaiting_approval"
    events = [e for e in api.AGENT.build.events_since(0) if e["kind"] == "tool"]
    assert [e["tool"] for e in events][-4:] == ["get_graph", "get_output_summary", "add_block", "submit_plan"]


def test_auto_build_and_build_log_endpoints(client):
    c = client
    c.scripts.update({"plan": [_plan(c.anchor)], "build": [_ask]})
    started = c.post("/api/agent/builds", json={"goal": "split it", "anchors": [c.anchor], "auto_build": True})
    assert started.status_code == 200, started.text
    assert started.json()["options"]["auto_build"] is True
    # No approve call: the plan goes straight to building.
    b = _wait(c, "awaiting_input")
    assert b["log_path"]
    stopped = c.post("/api/agent/builds/current/stop").json()

    log = c.get(f"/api/agent/builds/{stopped['id']}/log").json()
    assert log["phase"] == "stopped" and log["models"]["plan"]["provider"] == "claude_cli"
    assert any(e["kind"] == "auto_approved" for e in log["events"])
    assert c.get("/api/agent/builds/build_00000000/log").status_code == 404
    assert c.get("/api/agent/builds/..%2Fsecrets/log").status_code in (400, 404)


def test_custom_blocks_off_reaches_the_build_and_its_tools(client):
    c = client
    c.scripts.update({"plan": [_plan(c.anchor)], "build": [_ask]})
    started = c.post("/api/agent/builds", json={"goal": "split it", "anchors": [c.anchor], "allow_custom_blocks": False})
    assert started.json()["options"]["allow_custom_blocks"] is False
    _wait(c, "awaiting_approval")
    c.post("/api/agent/builds/current/approve")
    _wait(c, "awaiting_input")
    api.AGENT.build.set_phase("building")  # as if the model were mid-turn
    tools = {t["name"] for t in c.get("/api/agent/mcp/tools", headers={"X-Agent-Token": api.AGENT.token}).json()}
    assert "add_block" in tools and "add_custom_block" not in tools
    c.post("/api/agent/builds/current/stop")


def test_small_context_is_a_setting_builds_pick_up(client):
    c = client
    assert c.get("/api/llm/settings").json()["agent"]["small_context"] is False
    assert c.put("/api/llm/settings", json={"agent_small_context": True}).json()["agent"]["small_context"] is True
    # Untouched by other updates.
    assert c.put("/api/llm/settings", json={"agent_plan": {"model": "m"}}).json()["agent"]["small_context"] is True
    b = c.post("/api/agent/builds", json={"goal": "x", "anchors": [c.anchor]}).json()
    assert b["options"]["small_context"] is True
    c.post("/api/agent/stop")
