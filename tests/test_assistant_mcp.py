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
            init = await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            result = await session.call_tool("get_graph", {})
            return init.instructions, names, json.loads(result.content[0].text), result.isError

    instructions, names, graph, is_error = asyncio.run(go())
    # The backend's own API is off-limits to the assistant (issue #92).
    assert instructions == mm.INSTRUCTIONS and "never read that token" in instructions
    assert "run" in names and "open_project" in names
    assert not is_error and "blocks" in graph


def test_a_whole_run_rereads_a_source_whose_settings_changed(backend, tmp_path, monkeypatch):
    # Tagging a role makes the source stale; Run all re-reads it.
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
    assert ran["statuses"] == {"green": 2}, ran



def test_run_can_start_a_streaming_run(backend, tmp_path, monkeypatch):
    # The out-of-memory hint says "run ... with streaming: true": the tool can.
    monkeypatch.chdir(tmp_path)
    r = mm.Resolver(backend, None, discovery=tmp_path / "none")
    call(r, "open_project", folder="p", create=True)
    load = call(r, "add_block", category="demo_credit_data")
    split = call(r, "add_block", category="train_test_split")
    call(r, "connect", from_block=load["id"], from_port=load["outputs"][0].split(":")[0], to_block=split["id"], to_port="df")
    posted = []
    real_post = mm.Target.post
    monkeypatch.setattr(mm.Target, "post", lambda self, path, *a, **k: (posted.append(path), real_post(self, path, *a, **k))[1])
    ran = call(r, "run", streaming=True)
    assert posted == ["/run_all_streaming"] and "red" not in ran["statuses"], ran
    result, is_error = mm.call(r, "run", {"block": split["id"], "streaming": True})
    assert is_error and "leave out block" in result["error"]
    assert "streaming: true" in mm.OUT_OF_MEMORY_NEXT and mm.OUT_OF_MEMORY_NEXT.index("lazy") < mm.OUT_OF_MEMORY_NEXT.index("reduce it")

def test_an_assistant_writes_and_checks_a_library_block(backend, tmp_path, monkeypatch, block_library_dir):
    from tests.test_embedding import LAZY_SOURCE

    monkeypatch.chdir(tmp_path)
    r = mm.Resolver(backend, None, discovery=tmp_path / "none")
    call(r, "open_project", folder="p", create=True)
    guide = call(r, "get_block_guide")
    assert any("Large data" in rule for rule in guide["rules"])
    Path(guide["library_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(guide["library_dir"]) / "loans.py").write_text(LAZY_SOURCE)
    assert call(r, "check_block_file", name="loans.py")["ok"] is True
    csv = tmp_path / "panel.csv"
    csv.write_text("loan_id,period,dpd,extra\n1,202001,0,a\n")
    block = call(r, "add_block", category="monthly_panel", params={"path": str(csv)})
    assert call(r, "run")["statuses"] == {"green": 1}
    assert [c["name"] for c in call(r, "get_output_summary", block=block["id"])["columns"]] == ["loan_id", "period", "dpd"]


def test_an_edited_library_file_is_flagged_on_its_placed_copies(backend, tmp_path, monkeypatch, block_library_dir):
    # A placed block keeps the code it was placed with: after its file is
    # edited, the tools say so until update_block moves it (#93).
    from tests.test_embedding import LAZY_SOURCE

    monkeypatch.chdir(tmp_path)
    r = mm.Resolver(backend, None, discovery=tmp_path / "none")
    call(r, "open_project", folder="p", create=True)
    library = Path(call(r, "get_block_guide")["library_dir"])
    library.mkdir(parents=True, exist_ok=True)
    (library / "loans.py").write_text(LAZY_SOURCE)
    call(r, "check_block_file", name="loans.py")
    csv = tmp_path / "panel.csv"
    csv.write_text("loan_id,period,dpd,extra\n1,202001,0,a\n")
    block = call(r, "add_block", category="monthly_panel", params={"path": str(csv)})
    assert "outdated" not in block

    v2 = LAZY_SOURCE.replace('"version": 1', '"version": 2').replace('.select("loan_id", "period", "dpd")', '.select("loan_id", "dpd")')
    v2 = v2.replace('["loan_id", "period", "dpd"]', '["loan_id", "dpd"]')
    (library / "loans.py").write_text(v2)
    checked = call(r, "check_block_file", name="loans.py")
    assert checked["ok"] and [c["block"] for c in checked["placed_copies_outdated"]] == [block["id"]], checked
    assert "update_block" in checked["next"]
    placed = next(b for b in call(r, "get_graph")["blocks"] if b["id"] == block["id"])
    assert placed["outdated"] == {"file": "loans.py", "runs_version": 1, "library_version": 2}
    ran = call(r, "run")
    assert [b["id"] for b in ran["outdated_blocks"]] == [block["id"]] and "update_block" in ran["outdated_note"]

    preview = call(r, "update_block", block=block["id"])
    assert (preview["from_version"], preview["to_version"]) == (1, 2) and '"period"' in preview["code_diff"]
    updated = call(r, "update_block", block=block["id"], apply=True)
    assert updated["applied"]["to_version"] == 2 and "outdated" not in updated["block"]
    assert "outdated_blocks" not in call(r, "run")
    assert [c["name"] for c in call(r, "get_output_summary", block=block["id"])["columns"]] == ["loan_id", "dpd"]

    result, is_error = mm.call(r, "update_block", {"block": call(r, "add_block", category="filter")["id"]})
    assert is_error and "built-in" in result["error"]


def test_opens_canvases_in_the_editor_the_tools_run_for(backend, tmp_path, monkeypatch):
    discovery = tmp_path / "vscode-backends"
    monkeypatch.delenv("MODELMAKER_EDITOR_URI_SCHEME", raising=False)
    assert mm.editor_uri_scheme(discovery) == "vscode"
    discovery.mkdir()
    (discovery / "1-1.json").write_text(json.dumps({"folder": "/a", "url": backend, "token": None, "active_at": 1, "uri_scheme": "cursor"}))
    assert mm.editor_uri_scheme(discovery) == "cursor"
    # The MCP setup's choice wins; a malformed one is ignored.
    monkeypatch.setenv("MODELMAKER_EDITOR_URI_SCHEME", "windsurf")
    assert mm.editor_uri_scheme(discovery) == "windsurf"
    monkeypatch.setenv("MODELMAKER_EDITOR_URI_SCHEME", "not a scheme://")
    assert mm.editor_uri_scheme(discovery) == "cursor"

    launched = []
    monkeypatch.setattr(mm, "_launch_uri", lambda uri, scheme="vscode": launched.append((uri, scheme)) or False)
    r = mm.Resolver(None, None, discovery=discovery)
    with pytest.raises(mm.ToolInputError, match="couldn't ask Cursor"):
        mm._open_in_vscode(r, str(tmp_path / "elsewhere"), True)
    assert launched and launched[0][0].startswith("cursor://quantology.modelmaker/open?") and launched[0][1] == "cursor"


class _FakeTarget:
    """A backend whose graph reports a run for the first `busy` reads."""

    def __init__(self, busy: int) -> None:
        self.busy = busy
        self.posted: list[str] = []
        self.folder = None
        self.refuse: str | None = None

    def get(self, endpoint, /, **params):
        if endpoint == "/assistant/graph":
            running = self.busy > 0
            self.busy -= 1
            status = "running" if running else "green"
            block = {"id": "b_1", "name": "Stage", "status": status, **({"running_seconds": 448.0} if running else {})}
            return {"blocks": [block], "wires": [], "lanes": [], "running": running}
        assert endpoint == "/assistant/check"
        return {"leaks": [], "suspicious_features": [], "hints": {}, "ok": True}

    def post(self, path, body=None, timeout=60):
        self.posted.append(path)
        if self.refuse:
            raise mm.BackendError(self.refuse)
        return {"started": True}


class _FakeResolver:
    def __init__(self, target):
        self.t = target

    def target(self):
        return self.t


def test_wait_for_run_waits_and_run_queues_behind_a_run(monkeypatch):
    # Long runs (#96): no polling loops of the assistant's own.
    monkeypatch.setattr(mm.time, "sleep", lambda s: None)
    t = _FakeTarget(busy=3)
    waited = call(_FakeResolver(t), "wait_for_run")
    assert waited == {"statuses": {"green": 1}, "blocks": []} and not t.posted

    t = _FakeTarget(busy=10**9)
    still = call(_FakeResolver(t), "wait_for_run", seconds=0)
    assert still["still_running"] and "wait_for_run" in still["note"]
    assert still["blocks"] == [{"id": "b_1", "name": "Stage", "status": "running", "running_seconds": 448.0}]

    # A run called while another is going waits for it, then starts its own.
    t = _FakeTarget(busy=2)
    ran = call(_FakeResolver(t), "run")
    assert t.posted == ["/run_all"] and ran["statuses"] == {"green": 1}
    t = _FakeTarget(busy=10**9)
    monkeypatch.setattr(mm, "RUN_TIMEOUT_SECONDS", 0)
    queued = call(_FakeResolver(t), "run")
    assert queued["still_running"] and not t.posted and "hasn't started" in queued["note"]


def test_run_started_in_between_points_to_wait_for_run_and_bad_seconds_are_refused(monkeypatch):
    monkeypatch.setattr(mm.time, "sleep", lambda s: None)
    # Idle when checked, but the canvas starts a run before this one posts.
    t = _FakeTarget(busy=0)
    t.refuse = "a run is already in progress"
    result, is_error = mm.call(_FakeResolver(t), "run", {})
    assert is_error and "already in progress" in result["error"] and "wait_for_run" in result["error"]
    # Any other refusal passes through as it is.
    t.refuse = "no project is open"
    result, is_error = mm.call(_FakeResolver(t), "run", {})
    assert is_error and result["error"] == "no project is open"

    for bad in (float("nan"), "soon", float("inf")):
        result, is_error = mm.call(_FakeResolver(_FakeTarget(busy=0)), "wait_for_run", {"seconds": bad})
        assert is_error and "seconds must be a number" in result["error"]
    assert mm.call(_FakeResolver(_FakeTarget(busy=0)), "wait_for_run", {"seconds": "5"})[1] is False
