"""Several workspaces in one process (see modelmaker/workspace.py): through
the HTTP API, with a resolver that picks the workspace from a header (see
api.set_workspace_resolver), two workspaces can't see each other's graph,
runs, settings, env vars, AI build or recovery snapshot."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from modelmaker import api, library_store
from modelmaker.agent.loop import ScriptedLoop
from modelmaker.execution.env import task_environ
from modelmaker.workspace import Workspace

DATA = Path(__file__).resolve().parents[1] / "sample_data" / "credit_risk_data.csv"
HEADER = "X-Test-Workspace"


@pytest.fixture
def workspaces(tmp_path):
    a = Workspace("a", cache_dir=tmp_path / "a-cache", projects_dir=tmp_path / "a-projects")
    b = Workspace("b", cache_dir=tmp_path / "b-cache", projects_dir=tmp_path / "b-projects")
    by_id = {"a": a, "b": b}

    def resolve(request):
        ws = by_id.get(request.headers.get(HEADER, ""))
        if ws is None:
            raise HTTPException(404, "no such workspace")
        return ws

    api.set_workspace_resolver(resolve)
    try:
        yield a, b
    finally:
        api.set_workspace_resolver(None)
        for ws in (a, b):
            ws.session.runner.cancel()
            ws.agent.join(30)


def _client(ws: Workspace) -> TestClient:
    return TestClient(api.app, headers={HEADER: ws.id})


def _csv(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _green_read(c: TestClient, path: str) -> str:
    block = c.post("/api/blocks", json={"category": "read_csv", "params": {"path": path}}).json()
    assert c.post(f"/api/blocks/{block['id']}/refresh").json()["status"] == "green"
    return block["id"]


def _until(cond, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


# ---- the resolver -------------------------------------------------------------------------------


def test_the_resolver_picks_each_requests_workspace_and_may_refuse(workspaces):
    a, b = workspaces
    assert a.session is not api.SESSION and b.session is not api.SESSION
    no_header = TestClient(api.app)
    assert no_header.get("/api/graph").status_code == 404
    # Resolution is lazy: routes that touch no per-user state never ask.
    assert no_header.get("/api/health").status_code == 200
    # A refusal on a mutating request (which the AI-build canvas lock checks
    # first) is still the resolver's answer, not a server error.
    assert no_header.post("/api/blocks", json={"category": "filter"}).status_code == 404


def test_without_a_resolver_every_request_gets_the_default_workspace(tmp_path, monkeypatch):
    from modelmaker.session import ProjectSession

    monkeypatch.setattr(api, "SESSION", ProjectSession(recovery_path=tmp_path / "r.json"))
    c = TestClient(api.app)
    block = c.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}}).json()
    assert block["id"] in api.DEFAULT_WORKSPACE.session.graph.blocks
    assert api.current_workspace() is api.DEFAULT_WORKSPACE
    other = Workspace("other", cache_dir=tmp_path / "other")
    with api.workspace_context(other):
        assert api.current_workspace() is other
    assert api.current_workspace() is api.DEFAULT_WORKSPACE


# ---- graph ----------------------------------------------------------------------------------------


def test_graph_and_undo_are_per_workspace(workspaces):
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    in_a = ca.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}}).json()
    assert in_a["id"] in ca.get("/api/graph").json()["blocks"]
    graph_b = cb.get("/api/graph").json()
    assert graph_b["blocks"] == {} and graph_b["can_undo"] is False

    in_b = cb.post("/api/blocks", json={"category": "filter", "params": {"expr": "b > 1"}}).json()
    assert ca.post("/api/undo").status_code == 200
    assert ca.get("/api/graph").json()["blocks"] == {}
    assert list(cb.get("/api/graph").json()["blocks"]) == [in_b["id"]]
    assert cb.get("/api/graph").json()["can_redo"] is False

    # Each workspace's Save/Load starts in its own projects folder.
    assert ca.get("/api/project/default_dir").json()["path"] == str(a.projects_dir)
    assert cb.get("/api/project/default_dir").json()["path"] == str(b.projects_dir)


# ---- runs -----------------------------------------------------------------------------------------


def test_runs_cancel_and_cache_are_per_workspace(workspaces, tmp_path, monkeypatch):
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    read_a = _green_read(ca, _csv(tmp_path, "a.csv", "x\n1\n2\n"))
    library_store.add_block(
        "slow.py",
        {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]},
        "slow",
        "def slow(df):\n    import time\n    time.sleep(60)\n    return df\n",
    )
    slow = ca.post("/api/blocks", json={"library": {"file": "slow.py", "id": "slow"}}).json()
    ca.post("/api/wires", json={"from_block": read_a, "from_port": "out", "to_block": slow["id"], "to_port": "df"})

    # Start A's slow run without waiting for it.
    monkeypatch.setattr(api, "RUN_WAIT_SECONDS", 0.2)
    assert ca.post(f"/api/blocks/{slow['id']}/run").status_code == 200
    _until(lambda: a.session.runner.is_running(slow["id"]))
    assert a.run_slot.busy() and not b.run_slot.busy()
    # A's slot is taken; B's isn't -- B runs as if A weren't there.
    assert ca.post(f"/api/blocks/{read_a}/refresh").status_code == 409
    monkeypatch.setattr(api, "RUN_WAIT_SECONDS", 120.0)
    read_b = _green_read(cb, _csv(tmp_path, "b.csv", "y\n3\n"))

    # Stop in B has nothing of A's to stop.
    assert cb.post("/api/run/cancel").json() == {"cancelled": False}
    assert a.session.runner.is_running(slow["id"])
    assert ca.post("/api/run/cancel").json() == {"cancelled": True}
    _until(lambda: not a.run_slot.busy())
    assert ca.get("/api/graph").json()["blocks"][slow["id"]]["status"] != "green"
    assert cb.get("/api/graph").json()["blocks"][read_b]["status"] == "green"

    # Results are cached in each workspace's own directory.
    key_a = a.session.runner.state[read_a].last_successful_key
    key_b = b.session.runner.state[read_b].last_successful_key
    files_a = {p.stem for p in a.cache_dir.glob("*.mmw")}
    files_b = {p.stem for p in b.cache_dir.glob("*.mmw")}
    assert key_a in files_a and key_a not in files_b
    assert key_b in files_b and key_b not in files_a
    assert b.session.runner.cache.get(key_a) is None


# ---- settings and env vars ---------------------------------------------------------------------


def test_llm_settings_are_per_workspace(workspaces, monkeypatch):
    for var in ("MODELMAKER_OPENAI_API_KEY", "OPENAI_API_KEY", "MODELMAKER_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    set_a = ca.put("/api/llm/settings", json={"active_provider": "openai", "settings": {"openai": {"api_key": "sk-only-a", "model": "gpt-a"}}})
    assert set_a.status_code == 200 and "sk-only-a" not in set_a.text
    assert set_a.json()["settings"]["openai"]["api_key_set"] is True

    got_b = cb.get("/api/llm/settings").json()
    assert got_b["active_provider"] == "claude_cli"
    assert got_b["settings"]["openai"]["api_key_set"] is False
    assert got_b["settings"]["openai"]["model"] is None

    # And the providers each workspace builds: A's key never reaches B's.
    assert a.provider().api_key == "sk-only-a" and a.provider().model == "gpt-a"
    with pytest.raises(RuntimeError, match="no OpenAI API key"):
        b.provider("openai")
    assert a.resolve_api_key("openai") == "sk-only-a" and b.resolve_api_key("openai") is None
    assert "sk-only-a" not in os.environ.values()


def test_env_vars_are_per_workspace_and_never_in_os_environ(workspaces, monkeypatch):
    monkeypatch.delenv("MM_WS_TEST_VAR", raising=False)
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    assert ca.put("/api/env_vars/MM_WS_TEST_VAR", json={"value": "from-a"}).json()["source"] == "override"
    assert cb.get("/api/env_vars/MM_WS_TEST_VAR").json() == {"name": "MM_WS_TEST_VAR", "is_set": False, "source": None}
    assert "MM_WS_TEST_VAR" not in os.environ

    # A process-level variable shows through in every workspace, and an
    # override shadows it only where it was set.
    monkeypatch.setenv("MM_WS_TEST_VAR", "from-process")
    assert ca.get("/api/env_vars/MM_WS_TEST_VAR").json()["source"] == "override"
    assert cb.get("/api/env_vars/MM_WS_TEST_VAR").json()["source"] == "env"
    assert ca.delete("/api/env_vars/MM_WS_TEST_VAR").json()["source"] == "env"
    assert os.environ["MM_WS_TEST_VAR"] == "from-process"


ENV_BLOCK = (
    "def show_env(df, name: str = 'MM_WS_TEST_VAR'):\n"
    "    import os\n"
    "    return df.with_columns(pl.lit(os.environ.get(name, '<unset>')).alias('seen'))\n"
)


def _seen(c: TestClient, read: str, name: str) -> str:
    block = c.post("/api/blocks", json={"library": {"file": "envcheck.py", "id": "show_env"}, "params": {"name": name}}).json()
    c.post("/api/wires", json={"from_block": read, "from_port": "out", "to_block": block["id"], "to_port": "df"})
    out = c.post(f"/api/blocks/{block['id']}/run").json()
    assert out["status"] == "green", out
    return c.get(f"/api/blocks/{block['id']}/preview").json()["rows"][0]["seen"]


def test_a_task_sees_its_own_workspaces_env_vars_only(workspaces, tmp_path, monkeypatch):
    monkeypatch.delenv("MM_WS_TEST_VAR", raising=False)
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    library_store.add_block("envcheck.py", {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}, "show_env", ENV_BLOCK)
    ca.put("/api/env_vars/MM_WS_TEST_VAR", json={"value": "from-a"})
    cb.put("/api/env_vars/MM_WS_TEST_VAR", json={"value": "from-b"})
    # A secret-named override is accepted (it's the user's), but the
    # denylist still keeps it out of block code.
    ca.put("/api/env_vars/OPENAI_API_KEY", json={"value": "sk-not-for-blocks"})

    assert _seen(ca, _green_read(ca, _csv(tmp_path, "a.csv", "x\n1\n")), "MM_WS_TEST_VAR") == "from-a"
    assert _seen(cb, _green_read(cb, _csv(tmp_path, "b.csv", "x\n2\n")), "MM_WS_TEST_VAR") == "from-b"
    assert _seen(ca, _green_read(ca, _csv(tmp_path, "a2.csv", "x\n3\n")), "OPENAI_API_KEY") == "<unset>"

    # Try it (a library block called on a canvas block's output) gets them too.
    read = _green_read(cb, _csv(tmp_path, "b2.csv", "x\n4\n"))
    tried = cb.post("/api/library/files/envcheck.py/try", json={"block": "show_env", "input_block": read}).json()
    assert tried["ok"] and tried["outputs"]["out"]["head"][0]["seen"] == "from-b"


def test_task_environ_layers_overrides_but_keeps_the_denylist(monkeypatch):
    monkeypatch.setenv("MM_WS_BASE", "base")
    monkeypatch.setenv("MODELMAKER_WORKER_ENV_DENY", "DWH_ADMIN_*")
    env = task_environ({"MM_WS_BASE": "override", "DWH_URI": "postgresql://x", "DWH_ADMIN_URI": "postgresql://root", "GITHUB_TOKEN": "t", "MODELMAKER_WORKER_ENV_DENY": ""})
    assert env["MM_WS_BASE"] == "override" and env["DWH_URI"] == "postgresql://x"
    # Denied by name, by the built-in list, and by the admin's own list --
    # which an override can't lift.
    assert "DWH_ADMIN_URI" not in env and "GITHUB_TOKEN" not in env
    assert os.environ["MM_WS_BASE"] == "base"


def test_a_source_probe_looks_up_the_workspaces_variable(workspaces, tmp_path, monkeypatch):
    """Check for changes on Read SQL reads its connection variable from the
    workspace, like the block itself."""
    from modelmaker import sources

    seen: list[str] = []
    monkeypatch.setattr(sources, "_sql_fingerprint", lambda d, p, environ: seen.append(environ.get(p["connection_env"])) or "x")
    monkeypatch.delenv("MM_WS_DB", raising=False)
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    ca.put("/api/env_vars/MM_WS_DB", json={"value": "sqlite://a"})
    params = {"connection_env": "MM_WS_DB", "query": "SELECT 1", "probe_query": "SELECT 1"}
    for c in (ca, cb):
        block = c.post("/api/blocks", json={"category": "read_sql", "params": params}).json()
        c.post(f"/api/blocks/{block['id']}/check_changes")
    assert seen == ["sqlite://a", None]


# ---- AI build -------------------------------------------------------------------------------------


def _plan(call):
    r = call(
        "submit_plan",
        {"plan": {"summary": "split", "stages": [{"key": "est", "name": "Estimation", "goal": "70/30 holdout split", "blocks": ["train_test_split"]}]}},
    )
    assert r.get("ok"), r
    return "planned"


def _ask(call):
    call("ask_user", {"question": "Continue?"})
    return ""


def _anchor(c: TestClient) -> str:
    load = c.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(DATA)}}).json()
    assert c.post(f"/api/blocks/{load['id']}/run").json()["status"] == "green"
    c.post(f"/api/blocks/{load['id']}/column_role", json={"column": "default_flag", "role": "target"})
    c.post(f"/api/blocks/{load['id']}/column_role", json={"column": "interest_rate", "role": "excluded"})
    c.post(f"/api/blocks/{load['id']}/run")
    return load["id"]


def test_an_ai_build_and_its_canvas_lock_are_per_workspace(workspaces):
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    scripts = {"plan": [_plan], "build": [_ask]}
    a.agent.loop_factory = lambda choice, phase: ScriptedLoop(scripts[phase])
    b.agent.loop_factory = lambda choice, phase: ScriptedLoop([])
    for ws in (a, b):
        ws.llm_settings.active_provider = "claude_cli"
    anchor = _anchor(ca)
    _anchor(cb)

    assert ca.post("/api/agent/builds", json={"goal": "split it", "anchors": [anchor]}).status_code == 200
    _until(lambda: a.agent.build.phase == "awaiting_approval")
    a.agent.join(30)
    assert ca.post("/api/agent/builds/current/approve").status_code == 200
    _until(lambda: a.agent.build.phase == "awaiting_input")
    a.agent.join(30)
    assert ca.get("/api/agent/builds/current").json()["canvas_locked"] is True
    assert ca.post("/api/blocks", json={"category": "filter"}).status_code == 409

    # B has no build, no lock, and no access to A's MCP bridge.
    assert cb.get("/api/agent/builds/current").json() == {"build": None}
    assert cb.post("/api/blocks", json={"category": "filter"}).status_code == 200
    assert cb.get("/api/agent/mcp/tools", headers={"X-Agent-Token": a.agent.token}).status_code == 403
    assert ca.get("/api/agent/mcp/tools", headers={"X-Agent-Token": a.agent.token}).status_code == 200

    assert ca.post("/api/agent/builds/current/stop").json()["phase"] == "stopped"
    # A's build log went to A's cache directory, which B never lists.
    assert (a.cache_dir / "ai_builds").is_dir() and not (b.cache_dir / "ai_builds").exists()
    assert cb.get("/api/agent/builds/continuable").json() == []


# ---- recovery -------------------------------------------------------------------------------------


def test_recovery_snapshots_are_per_workspace(workspaces):
    a, b = workspaces
    ca, cb = _client(a), _client(b)
    ca.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}})
    assert ca.get("/api/project/recovery").json()["recovery"]["block_count"] == 1
    assert (a.cache_dir / "recovery.json").is_file()
    assert cb.get("/api/project/recovery").json() == {"recovery": None}
    assert cb.post("/api/project/recover").status_code == 404

    # After a restart (a fresh workspace on the same directory), A's work
    # comes back in A only.
    a2 = Workspace("a", cache_dir=a.cache_dir)
    with api.workspace_context(a2):
        assert a2.session.recovery_info()["block_count"] == 1
