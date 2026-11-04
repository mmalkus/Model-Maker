"""The runner agent (modelmaker/execution/agent.py) and RemoteExecutor
(remote.py) beyond the shared executor contract (test_execution_executor.py):
what a malicious agent can't do to the client, an agent dying or hanging,
cancel and time limits through it, what environment its tasks get, and
that nothing of it outlives its client."""

from __future__ import annotations

import json
import os
import pickle
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import pytest

from modelmaker.execution import Task, TaskCancelled
from modelmaker.execution import protocol, wire
from modelmaker.execution.remote import AgentLink, RemoteExecutor, agent_command, connect_unix, spawn_agent

ON_LINUX = sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="the runner agent needs POSIX sockets and process groups")


def _call(code: str, **kwargs) -> Task:
    name = code.split("def ", 1)[1].split("(", 1)[0]
    return Task("run_worker_entry", (name, code, kwargs, None))


def _wait1(ex, task, cancel=None):
    (result,) = ex.wait([ex.submit(task)], cancel or threading.Event())
    return result


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if ON_LINUX:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return False
        return state != "Z"
    return True


def _until(cond, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _children(pid: int) -> list[int]:
    out = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        out += [int(c) for c in (task / "children").read_text().split()]
    return out


@pytest.fixture
def remote():
    ex = RemoteExecutor(lambda: spawn_agent(warm_workers=0))
    yield ex
    ex.close()


# ---- a malicious agent ---------------------------------------------------------


class Evil:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (exec, (f"open({self.marker!r}, 'w').write('pwned')",))


def _forged(fields: dict, value_node, blobs: list[bytes]) -> bytes:
    header = json.dumps({"fields": fields, "value": value_node, "blobs": [len(b) for b in blobs]}).encode()
    return b"MMW1" + struct.pack(">I", len(header)) + header + b"".join(blobs)


def _raw_send(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">Q", len(data)) + data)


def _fake_agent(behaviour):
    """A RemoteExecutor whose `connect` hands it a socket to `behaviour`, a
    function run in a thread with the fake agent's end: it plays the agent."""
    errors: list[BaseException] = []

    def connect() -> AgentLink:
        mine, theirs = socket.socketpair()

        def run():
            try:
                behaviour(theirs)
            except (OSError, protocol.ProtocolError):
                pass
            except BaseException as e:  # noqa: BLE001 -- surfaced by the test
                errors.append(e)
            finally:
                theirs.close()

        threading.Thread(target=run, daemon=True).start()
        return AgentLink(mine)

    return RemoteExecutor(connect, connect_timeout=5, cancel_grace=2), errors


def _hello_then(after):
    def behaviour(sock):
        frame = protocol.recv_msg(sock, protocol.MAX_CONTROL_BYTES)
        protocol.check_hello(protocol.read_fields(frame, protocol.MAX_CONTROL_BYTES))
        protocol.send_msg(sock, protocol.message("hello", **protocol.hello_fields(pid=1)))
        task = protocol.recv_msg(sock, 1 << 30)
        after(sock, protocol.read_fields(task, 1 << 30))
        time.sleep(1)

    return behaviour


MALICIOUS = {
    "forged-result-pickle": lambda m: _hello_then(
        lambda s, t: _raw_send(
            s,
            _forged(
                {"op": "result", "task": t["task"], "ok": True, "error": None, "crashed": False},
                {"$pickle": 0},
                [pickle.dumps(Evil(m))],
            ),
        )
    ),
    "bare-pickle-result": lambda m: _hello_then(lambda s, t: _raw_send(s, pickle.dumps(Evil(m)))),
    "bare-pickle-hello": lambda m: (lambda s: (protocol.recv_msg(s, 1 << 20), _raw_send(s, pickle.dumps(Evil(m))))),
    "oversized-frame": lambda m: _hello_then(lambda s, t: s.sendall(struct.pack(">Q", 1 << 62) + b"x" * 1000)),
    "unknown-task-id": lambda m: _hello_then(
        lambda s, t: protocol.send_msg(s, protocol.message("result", task=t["task"] + 1000, ok=True, error=None, crashed=False))
    ),
    "unknown-op": lambda m: _hello_then(lambda s, t: protocol.send_msg(s, protocol.message("exec", code="1"))),
    "wrong-version": lambda m: (
        lambda s: (
            protocol.recv_msg(s, 1 << 20),
            protocol.send_msg(s, protocol.message("hello", **{**protocol.hello_fields(), "version": protocol.VERSION + 1})),
        )
    ),
    "wrong-protocol": lambda m: (
        lambda s: (
            protocol.recv_msg(s, 1 << 20),
            protocol.send_msg(s, protocol.message("hello", **{**protocol.hello_fields(), "protocol": "something-else"})),
        )
    ),
    "silent": lambda m: (lambda s: time.sleep(10)),
}


@pytest.mark.parametrize("kind", list(MALICIOUS))
def test_a_malicious_agent_cannot_run_code_in_the_client(tmp_path, kind):
    marker = tmp_path / "marker"
    ex, errors = _fake_agent(MALICIOUS[kind](str(marker)))
    if kind == "silent":
        ex.connect_timeout = 1
    start = time.monotonic()
    result = _wait1(ex, _call("def f():\n    return 1\n"))
    ex.close()
    assert time.monotonic() - start < 15
    assert not result.ok
    assert not marker.exists()
    assert not errors
    if kind == "forged-result-pickle":
        # A well-formed frame for the right task: only that result is
        # refused (safe_pickle's allowlist), the connection is fine.
        assert "rejected" in result.error and not result.crashed
    else:
        assert result.crashed, result.error
    if kind.startswith("wrong-"):
        assert "speaks" in result.error


def test_the_client_never_sends_a_denylisted_variable(monkeypatch):
    seen = {}
    monkeypatch.setenv("MODELMAKER_WORKER_ENV_DENY", "MM_TEST_DENIED_*")

    def capture(sock, task):
        seen.update(task)
        protocol.send_msg(sock, protocol.message("result", task=task["task"], ok=True, error=None, crashed=False))

    ex, errors = _fake_agent(_hello_then(capture))
    env = {"MM_TEST_OK": "1", "OPENAI_API_KEY": "sk", "MMS_SECRET_KEY": "s", "MM_TEST_DENIED_X": "d"}
    result = _wait1(ex, Task("run_worker_entry", ("f", None, {}, None), env))
    ex.close()
    assert result.ok and result.value is None, result.error
    assert not errors
    assert seen["env"] == {"MM_TEST_OK": "1"}
    assert seen["entry"] == "run_worker_entry" and seen["cwd"] is None


# ---- a real agent, driven by hand ------------------------------------------------


def _hand_session(**spawn):
    link = spawn_agent(warm_workers=0, **spawn)
    protocol.send_msg(link.sock, protocol.message("hello", **protocol.hello_fields()))
    fields = protocol.read_fields(protocol.recv_msg(link.sock, 1 << 20), 1 << 20)
    protocol.check_hello(fields)
    return link


def _hand_result(link):
    frame = protocol.recv_msg(link.sock, 1 << 30)
    return wire.decode_frame(frame)


def test_the_agent_refuses_a_client_of_another_protocol_version():
    link = spawn_agent(warm_workers=0)
    try:
        protocol.send_msg(link.sock, protocol.message("hello", **{**protocol.hello_fields(), "version": 99}))
        fields = protocol.read_fields(protocol.recv_msg(link.sock, 1 << 20), 1 << 20)
        assert fields["op"] == "error" and "version" in fields["message"]
        assert protocol.recv_msg(link.sock, 1 << 20) is None
        assert link.wait_exit(20)
    finally:
        link.kill()


def test_the_agent_runs_only_the_known_entry_points():
    link = _hand_session()
    try:
        protocol.send_msg(
            link.sock, protocol.message("task", ("echo hi",), task=1, entry="system", env=None, cwd=None, limits={})
        )
        fields, _ = _hand_result(link)
        assert fields["ok"] is False and "unknown entry point" in fields["error"]
    finally:
        link.kill()


def test_the_agent_relays_args_undecoded_and_the_task_process_refuses_a_pickle(tmp_path):
    # A client that's been subverted sends a pickle as an arg: the agent
    # doesn't decode it, the task process's decoder (safe_pickle) refuses it.
    marker = tmp_path / "marker"
    link = _hand_session()
    try:
        frame = _forged(
            {"op": "task", "task": 7, "entry": "run_worker_entry", "env": None, "cwd": None, "limits": {}},
            {"$tuple": [{"$pickle": 0}, None, {}, None]},
            [pickle.dumps(Evil(str(marker)))],
        )
        _raw_send(link.sock, frame)
        fields, _ = _hand_result(link)
    finally:
        link.kill()
    assert fields["task"] == 7 and fields["ok"] is False
    assert "task rejected" in fields["error"] and "not allowed" in fields["error"]
    assert not marker.exists()


def test_the_agent_applies_its_own_denylist_to_task_variables(monkeypatch):
    # Even if a client sent one, the agent drops a denylisted variable.
    code = "def env(names):\n    import os\n    return {n: os.environ.get(n) for n in names}\n"
    link = _hand_session(environ={**os.environ, "MODELMAKER_WORKER_ENV_DENY": "MM_TEST_DENIED_*"})
    try:
        args = ("env", code, {"names": ["MM_TEST_OK", "ANTHROPIC_API_KEY", "MM_TEST_DENIED_X"]}, None)
        env = {"MM_TEST_OK": "yes", "ANTHROPIC_API_KEY": "sk-ant", "MM_TEST_DENIED_X": "no"}
        protocol.send_msg(
            link.sock, protocol.message("task", args, task=1, entry="run_worker_entry", env=env, cwd=None, limits={})
        )
        fields, value = _hand_result(link)
    finally:
        link.kill()
    assert fields["ok"] is True, fields
    assert value == {"MM_TEST_OK": "yes", "ANTHROPIC_API_KEY": None, "MM_TEST_DENIED_X": None}


# ---- through RemoteExecutor --------------------------------------------------------


def test_env_overrides_reach_the_task_and_denylisted_ones_dont(remote):
    code = "def env(names):\n    import os\n    return {n: os.environ.get(n) for n in names}\n"
    names = ["MM_TEST_OK", "OPENAI_API_KEY", "MMS_GATEWAY_TOKEN"]
    task = Task("run_worker_entry", ("env", code, {"names": names}, None), {"MM_TEST_OK": "yes", "OPENAI_API_KEY": "sk", "MMS_GATEWAY_TOKEN": "t"})
    result = _wait1(remote, task)
    assert result.ok, result.error
    assert result.value == {"MM_TEST_OK": "yes", "OPENAI_API_KEY": None, "MMS_GATEWAY_TOKEN": None}


def test_dataframe_and_tagged_args_travel_wire_encoded(remote):
    import datetime as dt

    import numpy as np

    code = "def echo(**kw):\n    return kw\n"
    kwargs = {
        "df": pl.DataFrame({"a": [1, 2], "s": ["x", None]}),
        "series": pl.Series("v", [1.5, float("nan")]),
        "pair": (1, "a"),
        "when": dt.date(2024, 1, 2),
        "arr": np.arange(6).reshape(2, 3),
        "np_scalar": np.float64(0.25),
        "keys": {1: "int key", "$x": "dollar key"},
        "inf": float("inf"),
    }
    result = _wait1(remote, _call(code, **kwargs))
    assert result.ok, result.error
    out = result.value
    assert out["df"].equals(kwargs["df"]) and out["series"].to_list()[0] == 1.5
    assert out["pair"] == (1, "a") and out["when"] == dt.date(2024, 1, 2)
    assert (out["arr"] == kwargs["arr"]).all() and isinstance(out["np_scalar"], np.float64)
    assert out["keys"] == kwargs["keys"] and out["inf"] == float("inf")


def test_an_arg_that_isnt_data_fails_that_task_only(remote):
    code = "def f(x):\n    return 1\n"
    bad, good = remote.submit(_call(code, x=lambda: 1)), remote.submit(_call(code, x=1))
    r_bad, r_good = remote.wait([bad, good], threading.Event())
    assert not r_bad.ok and not r_bad.crashed and "can't be sent to the runner" in r_bad.error
    assert r_good.ok and r_good.value == 1


def test_a_fitted_estimator_travels_to_a_task_through_the_allowlist(remote):
    code_fit = (
        "def fit():\n    from sklearn.linear_model import LogisticRegression\n"
        "    return LogisticRegression().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])\n"
    )
    fitted = _wait1(remote, _call(code_fit))
    assert fitted.ok, fitted.error
    code_use = "def predict(model):\n    return [int(v) for v in model.predict([[0.0], [3.0]])]\n"
    used = _wait1(remote, _call(code_use, model=fitted.value))
    assert used.ok and used.value == [0, 1], used.error


@pytest.mark.skipif(not ON_LINUX, reason="reads /proc")
def test_the_agent_killed_mid_task_gives_crashed_results_then_recovers(tmp_path, remote):
    code = "def slow(path):\n    import os, time\n    open(path, 'w').write(str(os.getpid()))\n    time.sleep(60)\n"
    pidfile = tmp_path / "pid"
    handles = [remote.submit(_call(code, path=str(pidfile))), remote.submit(_call(code, path=str(tmp_path / "pid2")))]
    out: list = []
    t = threading.Thread(target=lambda: out.extend(remote.wait(handles, threading.Event(), max_concurrent=1)))
    t.start()
    assert _until(pidfile.exists)
    agent = remote.agent_pid
    task_pid = int(pidfile.read_text())
    os.kill(agent, signal.SIGKILL)
    t.join(30)
    assert not t.is_alive()
    assert all(r.crashed and "runner agent went away" in r.error for r in out), out
    # The task process went down with it (the agent's process group).
    assert _until(lambda: not _alive(task_pid), 10)
    # The next wait starts a new agent.
    result = _wait1(remote, _call("def f():\n    return 'again'\n"))
    assert result.ok and result.value == "again", result.error
    assert remote.agent_pid != agent


def test_cancel_through_the_agent_kills_a_task_that_ignores_sigterm(tmp_path, remote):
    code = (
        "def stubborn(path):\n    import os, signal, time\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    open(path, 'w').write(str(os.getpid()))\n    time.sleep(120)\n"
    )
    pidfile = tmp_path / "pid"
    cancel = threading.Event()
    threading.Thread(target=lambda: (_until(pidfile.exists), cancel.set()), daemon=True).start()
    start = time.monotonic()
    with pytest.raises(TaskCancelled):
        remote.wait([remote.submit(_call(code, path=str(pidfile)))], cancel)
    assert time.monotonic() - start < 30
    # TaskCancelled came after the process was gone, not before.
    assert not _alive(int(pidfile.read_text()))
    # The connection survived: the next task runs on the same agent.
    agent = remote.agent_pid
    assert _wait1(remote, _call("def f():\n    return 2\n")).value == 2
    assert remote.agent_pid == agent


def test_a_time_limit_is_enforced_by_the_agent(tmp_path):
    ex = RemoteExecutor(lambda: spawn_agent(warm_workers=0), task_timeout=1)
    try:
        start = time.monotonic()
        result = _wait1(ex, _call("def slow():\n    import time\n    time.sleep(60)\n"))
    finally:
        ex.close()
    assert time.monotonic() - start < 20
    assert not result.ok and not result.crashed and "time limit" in result.error


def test_an_agent_that_stops_answering_is_declared_dead_and_replaced(tmp_path):
    ex = RemoteExecutor(lambda: spawn_agent(warm_workers=0), heartbeat_interval=0.3, heartbeat_timeout=1.0)
    try:
        ex.prewarm()
        agent = ex.agent_pid
        os.kill(agent, signal.SIGSTOP)
        start = time.monotonic()
        result = _wait1(ex, _call("def f():\n    return 1\n"))
        assert time.monotonic() - start < 15
        assert result.crashed and "stopped answering" in result.error
        assert _until(lambda: not _alive(agent), 10)  # killed (a stopped process can be)
        assert _wait1(ex, _call("def f():\n    return 1\n")).value == 1
    finally:
        ex.close()


@pytest.mark.skipif(not ON_LINUX, reason="reads /proc")
@pytest.mark.parametrize("how", ["disconnect", "close"])
def test_nothing_outlives_the_client(tmp_path, how):
    ex = RemoteExecutor(lambda: spawn_agent(warm_workers=1))
    code = "def slow(path):\n    import os, time\n    open(path, 'w').write(str(os.getpid()))\n    time.sleep(60)\n"
    pidfile = tmp_path / "pid"
    out: list = []
    t = threading.Thread(target=lambda: out.extend(ex.wait([ex.submit(_call(code, path=str(pidfile)))], threading.Event())))
    t.start()
    assert _until(pidfile.exists)
    agent = ex.agent_pid
    assert _until(lambda: len(_children(agent)) >= 2)  # the task and its replacement spare
    processes = [agent, *_children(agent)]
    if how == "disconnect":
        # The client vanishes without a word: the agent sees EOF and stops
        # everything itself (not killed by the client here).
        ex.link.process = None
        ex.link.sock.shutdown(socket.SHUT_RDWR)
    else:
        ex.close()
    for pid in processes:
        assert _until(lambda pid=pid: not _alive(pid), 20), f"{pid} outlived the client"
    t.join(30)
    assert out and out[0].crashed
    ex.close()


def test_a_listening_socket_with_a_group_mode(tmp_path):
    # E.g. a sandbox's socket in a setgid directory the client's group owns.
    path = str(tmp_path / "agent.sock")
    proc = subprocess.Popen(agent_command(listen=path, warm_workers=0, socket_mode=0o660), start_new_session=True)
    try:
        assert _until(lambda: os.path.exists(path))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o660
        ex = RemoteExecutor(lambda: connect_unix(path))
        try:
            assert _wait1(ex, _call("def f():\n    return 'group'\n")).value == "group"
        finally:
            ex.close()
        assert proc.wait(20) == 0
    finally:
        proc.kill()


@pytest.mark.parametrize("mode", ["0777", "0466", "abc", "4660"])
def test_bad_socket_modes_are_refused(mode, tmp_path):
    from modelmaker.execution import agent

    with pytest.raises(SystemExit):
        agent.main(["--listen", str(tmp_path / "s"), "--socket-mode", mode])
    with pytest.raises(SystemExit):
        agent.main(["--fd", "99", "--socket-mode", "0660"])  # only with --listen
    assert not (tmp_path / "s").exists()


def test_an_agent_listening_on_a_path(tmp_path):
    path = str(tmp_path / "agent.sock")
    proc = subprocess.Popen(agent_command(listen=path, warm_workers=0), start_new_session=True)
    try:
        assert _until(lambda: os.path.exists(path))
        assert os.stat(path).st_mode & 0o077 == 0  # only its owner may connect
        ex = RemoteExecutor(lambda: connect_unix(path))
        try:
            assert _wait1(ex, _call("def f():\n    return 'listening'\n")).value == "listening"
            assert not os.path.exists(path)  # one connection, then the path is gone
        finally:
            ex.close()
        assert proc.wait(20) == 0
    finally:
        proc.kill()
