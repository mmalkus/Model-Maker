"""The hooks an application hosting several users in one process builds on
(a server: each user's block code started as their own OS user, in their
own folder; their own library, git identity and file confinement), and
which leave the local app's behaviour unchanged when unused:

- worker_pool.WorkerLaunch: a command prefix, a base environment and a
  task cwd for a pool's workers;
- LocalExecutor(task_timeout=...): a wall-clock limit per task;
- Workspace(root=..., library_dir=..., git_env=...);
- background threads (runs, AI builds) keep the request's workspace;
- AnthropicProvider / AnthropicLoop take a base_url, like the others.
"""

from __future__ import annotations

import contextvars
import os
import subprocess
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from modelmaker import api, gitops, library_store
from modelmaker.execution import Task
from modelmaker.execution.local import LocalExecutor
from modelmaker.runner import MP_CONTEXT
from modelmaker.runslot import RunSlot
from modelmaker.worker_pool import WorkerLaunch, WorkerPool
from modelmaker.workspace import Workspace

posix_only = pytest.mark.skipif(os.name != "posix", reason="the exec launcher is POSIX-only")


def _call(code: str, env: dict[str, str] | None = None) -> Task:
    name = code.split("def ", 1)[1].split("(", 1)[0]
    return Task("run_worker_entry", (name, code, {}, None), env)


def _run(executor: LocalExecutor, task: Task):
    (result,) = executor.wait([executor.submit(task)], threading.Event())
    return result


PROBE = (
    "def probe():\n"
    "    import os\n"
    "    raw = open('/proc/self/environ', 'rb').read() if os.path.exists('/proc/self/environ') else b''\n"
    "    return {'env': dict(os.environ), 'cwd': os.getcwd(), 'pid': os.getpid(), 'exec_env': raw.decode(errors='replace')}\n"
)


# ---- WorkerLaunch ------------------------------------------------------------------------------


@posix_only
@pytest.mark.parametrize("warm", [0, 1])
def test_a_launch_sets_the_prefix_base_environment_and_cwd(tmp_path, monkeypatch, warm):
    monkeypatch.setenv("MM_PARENT_ONLY", "parent")
    marker = tmp_path / "prefix-ran"
    # A prefix that records the pid it execs as, then execs the worker.
    prefix = ("/bin/sh", "-c", f'echo $$ >> "{marker}"; exec "$0" "$@"')
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "MM_BASE": "base", "ANTHROPIC_API_KEY": "sk-base"}
    work = tmp_path / "work"
    work.mkdir()
    launch = WorkerLaunch(command_prefix=prefix, environ=base, cwd=str(work))
    pool = WorkerPool(MP_CONTEXT, warm, launcher="exec", launch=launch)
    ex = LocalExecutor(pool)
    try:
        ex.prewarm()
        result = _run(ex, _call(PROBE, {"MM_TASK": "task", "OPENAI_API_KEY": "sk-task"}))
    finally:
        ex.close()
    assert result.ok, result.error
    out = result.value
    assert out["cwd"] == str(work)
    assert out["env"]["MM_BASE"] == "base" and out["env"]["MM_TASK"] == "task"
    # Not this process's environment; and the secrets denylist applies to
    # the base and the task's variables alike.
    assert "MM_PARENT_ONLY" not in out["env"]
    assert "ANTHROPIC_API_KEY" not in out["env"] and "OPENAI_API_KEY" not in out["env"]
    if out["exec_env"]:
        # What it was exec'd with, too.
        assert "MM_PARENT_ONLY" not in out["exec_env"] and "sk-base" not in out["exec_env"]
    # (A spare's replacement may have started since: every worker goes through it.)
    assert out["pid"] in [int(line) for line in marker.read_text().split()]
    assert ex.base_environ == base


def test_a_launch_needs_the_exec_launcher():
    with pytest.raises(ValueError, match="exec launcher"):
        WorkerPool(MP_CONTEXT, 0, launcher="multiprocessing", launch=WorkerLaunch(cwd="/"))
    # The default launch is no launch at all.
    assert WorkerPool(MP_CONTEXT, 0, launcher="multiprocessing").launch.is_default


@posix_only
def test_spares_belong_to_their_pools_launch(tmp_path):
    def pool(tag):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "MM_POOL": tag}
        return WorkerPool(MP_CONTEXT, 1, launcher="exec", launch=WorkerLaunch(environ=env, cwd=str(tmp_path)))

    a, b = LocalExecutor(pool("a")), LocalExecutor(pool("b"))
    try:
        a.prewarm()
        b.prewarm()
        for _ in range(2):
            assert _run(a, _call(PROBE)).value["env"]["MM_POOL"] == "a"
            assert _run(b, _call(PROBE)).value["env"]["MM_POOL"] == "b"
    finally:
        a.close()
        b.close()


# ---- task_timeout ------------------------------------------------------------------------------


def test_a_task_over_its_time_limit_is_stopped_and_fails():
    ex = LocalExecutor(WorkerPool(MP_CONTEXT, 0), task_timeout=1.5)
    slow = "def slow():\n    import time\n    time.sleep(60)\n"
    quick = "def quick():\n    return 7\n"
    try:
        t0 = time.monotonic()
        handles = [ex.submit(_call(slow)), ex.submit(_call(quick))]
        slow_result, quick_result = ex.wait(handles, threading.Event())
    finally:
        ex.close()
    assert time.monotonic() - t0 < 30
    assert not slow_result.ok and "time limit" in slow_result.error and not slow_result.crashed
    assert quick_result.ok and quick_result.value == 7


# ---- per-workspace library, git and confinement ------------------------------------------------


HEADER = "X-Test-Workspace"


@pytest.fixture
def hosted(tmp_path):
    def make(name):
        root = tmp_path / name
        (root / "projects").mkdir(parents=True)
        return Workspace(
            name,
            cache_dir=root / "cache",
            projects_dir=root / "projects",
            root=root,
            library_dir=root / "library",
            git_env={
                # The suite sets GIT_AUTHOR_* for every test; a workspace's
                # environment is whole, so they're simply not in it.
                **{k: v for k, v in os.environ.items() if not k.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_"))},
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "user.name",
                "GIT_CONFIG_VALUE_0": f"user-{name}",
                "GIT_CONFIG_KEY_1": "user.email",
                "GIT_CONFIG_VALUE_1": f"{name}@example.com",
            },
        )

    by_id = {"a": make("a"), "b": make("b")}

    def resolve(request):
        ws = by_id.get(request.headers.get(HEADER, ""))
        if ws is None:
            raise HTTPException(404, "no such workspace")
        return ws

    api.set_workspace_resolver(resolve)
    try:
        yield by_id["a"], by_id["b"]
    finally:
        api.set_workspace_resolver(None)


def _client(ws: Workspace) -> TestClient:
    return TestClient(api.app, headers={HEADER: ws.id})


LIB_FILE = '"""Mine."""\n\nimport polars as pl\n\nBLOCKS = {"mine": {"inputs": [{"name": "df"}], "outputs": [{"name": "out"}]}}\n\n\ndef mine(df):\n    return df\n'


def test_each_workspace_has_its_own_library(hosted):
    a, b = hosted
    ca, cb = _client(a), _client(b)
    assert ca.put("/api/library/files/mine.py", json={"source": LIB_FILE, "create": True}).status_code == 200
    assert (a.library_dir / "mine.py").exists() and not (b.library_dir / "mine.py").exists()
    assert [f["name"] for f in ca.get("/api/library").json()["files"]] == ["mine.py"]
    assert cb.get("/api/library").json()["files"] == []
    assert ca.get("/api/library").json()["dir"] == str(a.library_dir)
    # Outside any request, the process-wide folder, as before.
    assert library_store.library_dir() == library_store.default_library_dir()
    with api.workspace_context(b):
        assert library_store.library_dir() == b.library_dir


def test_background_threads_keep_the_callers_context():
    var: contextvars.ContextVar[str] = contextvars.ContextVar("hosting_test", default="unset")
    var.set("the request's")
    seen = RunSlot().start_background(lambda: var.get(), wait_seconds=10)
    assert seen == "the request's"


def test_git_runs_with_the_workspaces_environment(hosted, tmp_path):
    a, b = hosted
    ca = _client(a)
    folder = a.root / "projects" / "p"
    assert ca.post("/api/project/save", json={"path": str(folder)}).status_code == 200
    assert ca.post("/api/project/git/init").status_code == 200
    assert ca.post("/api/project/git/commit", json={"message": "first"}).status_code == 200
    author = subprocess.run(
        ["git", "log", "-1", "--format=%an <%ae>"], cwd=folder, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert author == "user-a <a@example.com>"
    # Outside a request: this process's environment (None).
    gitops.set_env_resolver(lambda: None)
    try:
        assert gitops._env() is None
    finally:
        gitops.set_env_resolver(lambda: api.current_workspace().git_env)


def test_browse_load_and_save_stay_inside_the_root(hosted, tmp_path):
    a, _ = hosted
    ca = _client(a)
    root = a.root
    assert ca.get("/api/browse").json()["path"] == str(root)
    assert ca.get("/api/browse").json()["parent"] is None
    assert ca.get("/api/browse", params={"path": "projects"}).json()["path"] == str(root / "projects")
    assert ca.get("/api/browse", params={"path": str(tmp_path)}).status_code == 403
    assert ca.get("/api/browse", params={"path": "../b"}).status_code == 403
    (root / "projects" / "out").symlink_to(tmp_path / "b")
    assert ca.get("/api/browse", params={"path": str(root / "projects" / "out")}).status_code == 403

    assert ca.post("/api/project/save", json={"path": str(tmp_path / "b" / "x")}).status_code == 403
    assert ca.post("/api/project/save", json={"path": str(root / "projects" / "out" / "x")}).status_code == 403
    assert ca.post("/api/project/save", json={"path": "projects/fine"}).status_code == 200
    assert (root / "projects" / "fine" / "model.json").exists()
    assert ca.post("/api/project/load", json={"path": str(tmp_path / "b")}).status_code == 403

    # A project holding a link out of the root (as a git pull could bring)
    # is neither loaded nor saved over.
    (root / "projects" / "fine" / "files" / "leak").symlink_to(tmp_path / "b")
    assert ca.post("/api/project/load", json={"path": "projects/fine"}).status_code == 403
    assert ca.post("/api/project/save", json={"path": "projects/fine"}).status_code == 403


def test_without_a_root_nothing_is_confined(tmp_path, monkeypatch):
    ws = Workspace("local", cache_dir=tmp_path / "c", projects_dir=tmp_path / "p")
    assert ws.root is None and ws.library_dir is None and ws.git_env is None
    monkeypatch.setattr(api, "DEFAULT_WORKSPACE", ws)
    c = TestClient(api.app)
    assert c.get("/api/browse", params={"path": str(tmp_path)}).json()["path"] == str(tmp_path.resolve())


# ---- providers -----------------------------------------------------------------------------------


def test_anthropic_takes_a_base_url():
    pytest.importorskip("anthropic")
    from modelmaker.agent.loop import make_loop
    from modelmaker.llm import get_provider

    p = get_provider("anthropic", api_key="k", base_url="http://127.0.0.1:9/llm/tok/anthropic")
    assert str(p.client.base_url).startswith("http://127.0.0.1:9/llm/tok/anthropic")
    loop = make_loop("anthropic", None, api_base_url="", token="", api_key="k", base_url="http://127.0.0.1:9/x")
    assert str(loop.client.base_url).startswith("http://127.0.0.1:9/x")
    assert "127.0.0.1:9" not in str(get_provider("anthropic", api_key="k").client.base_url)
