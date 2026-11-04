"""What a block can't do to the process that runs the graph: get code executed here
through its result, or read this process's secrets from its environment.
Every block below is ordinary library block code, as a user (or an LLM)
would write it."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from modelmaker.blocks.base import PortSpec
from modelmaker.cache import CacheStore
from modelmaker.execution.local import LocalExecutor
from modelmaker.graph import Graph, Wire
from modelmaker.runner import MP_CONTEXT, Runner
from modelmaker.worker_pool import WorkerPool

from .helpers import make_block

ON_LINUX = sys.platform.startswith("linux")


def _graph_with(tmp_path, category: str, code: str, params: dict | None = None) -> Graph:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,10\n2,20\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv)})
    block = make_block(
        "b_x",
        category,
        params=params or {},
        code=code,
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out", type="any")],
        x=1,
    )
    return Graph(blocks={"b_read": read, "b_x": block}, wires={"w": Wire("w", "b_read", "out", "b_x", "df")})


def _run(graph: Graph, executor=None):
    runner = Runner(graph, CacheStore(), executor=executor)
    assert runner.refresh("b_read") == "green", runner.state["b_read"].last_error
    status = runner.run_block("b_x")
    entry = runner.cache.get(runner.compute_key("b_x"))
    return status, runner.state["b_x"].last_error, (entry.outputs["out"] if entry else None)


def _fresh_executor(launcher=None) -> LocalExecutor:
    """No spares: every task process is started after the test has set up
    its environment."""
    return LocalExecutor(WorkerPool(MP_CONTEXT, 0, launcher=launcher))


# ---- nothing executable crosses back -----------------------------------------


EVIL_RESULT = '''
def evil_result(df, marker: str = ""):
    class Evil:
        def __reduce__(self):
            return (exec, ("open(%r, 'w').write('pwned')" % marker,))
    return {"looks": "harmless", "payload": Evil()}
'''


def test_a_result_whose_unpickling_would_run_code_is_an_ordinary_block_error(tmp_path):
    marker = tmp_path / "marker"
    status, error, _ = _run(_graph_with(tmp_path, "evil_result", EVIL_RESULT, {"marker": str(marker)}))
    assert status == "red"
    assert "can't return" in error
    assert not marker.exists()


# The real attack: block code doesn't have to go through the encoder at all.
# It can find the worker's connection and write whatever bytes it likes to
# the parent -- a hand-made frame with a pickle in it, or a bare pickle the
# way results used to travel.
FORGED_FRAME = '''
def forged_frame(df, marker: str = "", raw: bool = False):
    import gc, json, os, pickle, struct

    class Evil:
        def __reduce__(self):
            return (exec, ("open(%r, 'w').write('pwned')" % marker,))

    payload = pickle.dumps(Evil())
    conn = next(o for o in gc.get_objects() if type(o).__name__ == "_ConnResult")._conn
    if raw:
        conn.send(Evil())  # pickled by multiprocessing, like the old protocol
    else:
        header = json.dumps({"fields": {"ok": True, "error": None, "modules": []},
                             "value": {"$pickle": 0}, "blobs": [len(payload)]}).encode()
        conn.send_bytes(b"MMW1" + struct.pack(">I", len(header)) + header + payload)
    os._exit(0)
'''


@pytest.mark.parametrize("raw", [False, True], ids=["forged-wire-frame", "bare-pickle"])
def test_a_block_writing_a_malicious_pickle_to_the_parent_cannot_run_code_there(tmp_path, raw):
    marker = tmp_path / "marker"
    graph = _graph_with(tmp_path, "forged_frame", FORGED_FRAME, {"marker": str(marker), "raw": raw})
    status, error, _ = _run(graph)
    assert status == "red"
    assert "rejected" in error
    assert not marker.exists()


def test_a_dataframe_and_json_result_still_comes_back(tmp_path):
    code = '''
def plain_result(df):
    return {"rows": df.height, "frame": df, "ratio": float("nan"), "pair": (1, 2)}
'''
    status, error, out = _run(_graph_with(tmp_path, "plain_result", code))
    assert status == "green", error
    assert out["rows"] == 2 and out["pair"] == (1, 2) and out["frame"].columns == ["a", "b"]


# ---- task processes see no secrets -------------------------------------------


ENV_PROBE = '''
def env_probe(df, wanted: str = ""):
    import os
    proc = None
    try:
        with open("/proc/self/environ", "rb") as f:
            proc = [item.split(b"=", 1)[0].decode() for item in f.read().split(b"\\0") if item]
    except OSError:
        pass
    return {"environ": sorted(os.environ), "proc": proc, "wanted": os.environ.get(wanted)}
'''

SECRETS = {
    "MODELMAKER_AUTH_TOKEN": "tok-123",
    "ANTHROPIC_API_KEY": "sk-ant-xyz",
    "OPENAI_API_KEY": "sk-openai",
    "MODELMAKER_OPENAI_API_KEY": "sk-openai-2",
    "GEMINI_API_KEY": "g-1",
    "MODELMAKER_GEMINI_API_KEY": "g-2",
    "GOOGLE_API_KEY": "g-3",
    "MMS_SECRET_KEY": "server-secret",
    "MMS_GATEWAY_TOKEN": "gw-token",
    "MM_TEST_ADMIN_EXTRA": "denied-by-admin",
}


@pytest.fixture
def secret_env(monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MODELMAKER_WORKER_ENV_DENY", "MM_TEST_ADMIN_*")
    # A connection-string variable a block reads by name (like Read SQL's
    # `connection_env`): not a secret to the scrubber, so it stays visible.
    monkeypatch.setenv("MM_TEST_DB_URL", "sqlite:///tmp/x.db")


@pytest.mark.parametrize("warm", [False, True], ids=["cold-start", "warm-spare"])
def test_a_task_process_sees_no_secrets_but_does_see_ordinary_variables(tmp_path, secret_env, warm):
    # The exec launcher explicitly (the default on POSIX, unless
    # MODELMAKER_WORKER_LAUNCHER forces the fallback for a test run).
    launcher = "exec" if os.name == "posix" else None
    executor = LocalExecutor(WorkerPool(MP_CONTEXT, 1 if warm else 0, launcher=launcher))
    if warm:
        executor.prewarm()
    try:
        status, error, out = _run(_graph_with(tmp_path, "env_probe", ENV_PROBE, {"wanted": "MM_TEST_DB_URL"}), executor)
    finally:
        executor.close()
    assert status == "green", error
    leaked = set(SECRETS) & set(out["environ"])
    assert not leaked, f"visible in os.environ: {leaked}"
    assert out["wanted"] == "sqlite:///tmp/x.db"
    if ON_LINUX and launcher == "exec":
        assert out["proc"] is not None
        leaked = set(SECRETS) & set(out["proc"])
        assert not leaked, f"visible in /proc/self/environ: {leaked}"
        # The exec'd environment is the scrubbed copy, not an empty one.
        assert "MM_TEST_DB_URL" in out["proc"] and "PATH" in out["proc"]


def test_a_variable_set_after_the_server_started_still_reaches_a_warm_spare(tmp_path, monkeypatch):
    # api.set_env_var writes os.environ at run time; a spare started before
    # that must still see it (each task carries the current, scrubbed env).
    executor = LocalExecutor(WorkerPool(MP_CONTEXT, 1))
    executor.prewarm()
    monkeypatch.setenv("MM_TEST_LATE_VAR", "late")
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "late-token")
    try:
        status, error, out = _run(_graph_with(tmp_path, "env_probe", ENV_PROBE, {"wanted": "MM_TEST_LATE_VAR"}), executor)
    finally:
        executor.close()
    assert status == "green", error
    assert out["wanted"] == "late"
    assert "MODELMAKER_AUTH_TOKEN" not in out["environ"]


def test_the_multiprocessing_fallback_scrubs_os_environ_too(tmp_path, secret_env):
    # The launcher used where the exec one isn't available (Windows, a
    # frozen app): os.environ is scrubbed; the original environment block
    # isn't (documented in worker_pool.py), so /proc isn't asserted here.
    executor = _fresh_executor(launcher="multiprocessing")
    try:
        status, error, out = _run(_graph_with(tmp_path, "env_probe", ENV_PROBE, {"wanted": "MM_TEST_DB_URL"}), executor)
    finally:
        executor.close()
    assert status == "green", error
    assert not set(SECRETS) & set(out["environ"])
    assert out["wanted"] == "sqlite:///tmp/x.db"


def test_the_parent_environment_is_left_alone(tmp_path, secret_env):
    executor = _fresh_executor()
    try:
        _run(_graph_with(tmp_path, "env_probe", ENV_PROBE), executor)
    finally:
        executor.close()
    for name, value in SECRETS.items():
        assert os.environ[name] == value


@pytest.mark.skipif(not ON_LINUX, reason="prctl(PR_SET_DUMPABLE) is Linux-only")
def test_opt_in_hardening_marks_the_parent_non_dumpable():
    # In a child interpreter: the flag can't be unset again in this one.
    script = textwrap.dedent(
        """
        import ctypes
        from modelmaker.execution.local import LocalExecutor
        from modelmaker.worker_pool import WorkerPool
        import multiprocessing
        LocalExecutor(WorkerPool(multiprocessing.get_context("spawn"), 0))
        print(ctypes.CDLL(None).prctl(3, 0, 0, 0, 0))
        """
    )
    for flag, expected in (("1", "0"), ("", "1")):
        env = {**os.environ, "MODELMAKER_HARDEN": flag}
        out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().splitlines()[-1] == expected
