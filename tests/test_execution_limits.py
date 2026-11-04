"""Per-task memory and thread limits (modelmaker/execution/limits.py):
a block over its budget fails fast with an error naming the limit, through
LocalExecutor and through a RemoteExecutor and its runner agent; ordinary
heavy blocks pass under a realistic budget; nothing changes without limits."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from modelmaker.execution import Task, TaskLimits
from modelmaker.execution.limits import address_space_mb, annotate_error, check_limit
from modelmaker.execution.remote import RemoteExecutor, agent_command, spawn_agent

from .helpers import EXECUTOR_KINDS, make_executor

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the watchdog reads /proc (Linux)")

# What a task gets by default on a sandboxed server (a 2048 MB / 256-pids
# sandbox with one spare): the budget ordinary blocks must fit in.
SERVER_DEFAULT = TaskLimits(memory_mb=1536, threads=208)

HOG = "def hog(mb):\n    chunks = []\n    for _ in range(mb):\n        chunks.append(bytearray(b'x' * (1024 * 1024)))\n    return len(chunks)\n"
THREADS = (
    "def threads(n):\n"
    "    import threading\n"
    "    stop = threading.Event()\n"
    "    for _ in range(n):\n"
    "        threading.Thread(target=stop.wait, args=(30,), daemon=True).start()\n"
    "    stop.wait(30)\n"
    "    return n\n"
)
CHILD_HOG = (
    "def child_hog(mb):\n"
    "    import os, time\n"
    "    if os.fork() == 0:\n"
    "        chunks = [bytearray(b'x' * (1024 * 1024)) for _ in range(mb)]\n"
    "        time.sleep(30)\n"
    "        os._exit(0)\n"
    "    time.sleep(30)\n"
)
FORKS = (
    "def forks(n):\n"
    "    import os, time\n"
    "    for _ in range(n):\n"
    "        if os.fork() == 0:\n"
    "            time.sleep(30)\n"
    "            os._exit(0)\n"
    "    time.sleep(30)\n"
)
PROBE = (
    "def probe():\n"
    "    import os, resource\n"
    "    return {'pgid_is_own': os.getpgid(0) == os.getpid(),\n"
    "            'as': resource.getrlimit(resource.RLIMIT_AS)[0], 'nproc': resource.getrlimit(resource.RLIMIT_NPROC)[0]}\n"
)
HEAVY = (
    "def heavy():\n"
    "    import numpy as np\n"
    "    import polars as pl\n"
    "    import scipy.stats  # noqa: F401\n"
    "    from sklearn.linear_model import LogisticRegression\n"
    "    rng = np.random.default_rng(0)\n"
    "    n = 300_000\n"
    "    a = pl.DataFrame({'k': np.arange(n) % 5_000, 'x': rng.random(n), 'y': (rng.random(n) > 0.5).astype(int)})\n"
    "    b = pl.DataFrame({'k': np.arange(5_000), 'z': rng.random(5_000)})\n"
    "    j = a.join(b, on='k').with_columns((pl.col('x') * pl.col('z')).alias('xz'))\n"
    "    agg = j.group_by('k').agg(pl.col('xz').mean())\n"
    "    m = LogisticRegression(max_iter=300).fit(j.select('x', 'z', 'xz').to_numpy(), j['y'].to_numpy())\n"
    "    return {'rows': j.height, 'groups': agg.height, 'coef': len(m.coef_[0])}\n"
)


def _call(code: str, **kwargs) -> Task:
    name = code.split("def ", 1)[1].split("(", 1)[0]
    return Task("run_worker_entry", (name, code, kwargs, None))


def _wait1(ex, task):
    (result,) = ex.wait([ex.submit(task)], threading.Event())
    return result


def _timed(ex, task):
    start = time.monotonic()
    result = _wait1(ex, task)
    return result, time.monotonic() - start


@pytest.fixture(params=EXECUTOR_KINDS)
def kind(request):
    return request.param


# ---- units ------------------------------------------------------------------------


def test_the_stricter_limit_wins_field_by_field():
    a = TaskLimits(memory_mb=500, threads=None, total_threads=100)
    b = TaskLimits(memory_mb=800, threads=50)
    assert a.stricter(b) == TaskLimits(memory_mb=500, threads=50, total_threads=100)
    assert a.stricter(None) == a
    assert not TaskLimits() and TaskLimits(threads=1)
    assert TaskLimits(memory_mb=10, threads=5, total_threads=7).child_fields() == {"group": 1, "memory_mb": 10, "total_threads": 7}
    assert TaskLimits(threads=5).child_fields() == {"group": 1}


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_a_limit_is_a_positive_whole_number(bad):
    with pytest.raises(ValueError):
        check_limit(bad, "memory_mb")
    assert check_limit(None, "x") is None and check_limit(3, "x") == 3


def test_errors_from_the_rlimit_backstops_name_the_limit():
    limits = TaskLimits(memory_mb=300, threads=40, total_threads=60)
    assert "memory limit of 300 MB" in annotate_error("MemoryError: Unable to allocate 74.5 GiB", limits)
    assert "threads and processes: 40" in annotate_error("RuntimeError: can't start new thread", limits)
    assert "threads and processes: 40" in annotate_error("BlockingIOError: [Errno 11] Resource temporarily unavailable", limits)
    assert annotate_error("ValueError: bad", limits) == "ValueError: bad"
    assert annotate_error("MemoryError", TaskLimits()) == "MemoryError"
    # The address-space backstop is far above a normal run's virtual size.
    assert address_space_mb(160) > 4096


# ---- through the executors --------------------------------------------------------


def test_a_block_over_its_memory_limit_fails_fast_and_the_next_runs(kind):
    ex = make_executor(kind, limits=TaskLimits(memory_mb=300))
    try:
        result, took = _timed(ex, _call(HOG, mb=4000))
        assert not result.ok and result.crashed
        assert result.error == "stopped: the block used more than its memory limit of 300 MB"
        assert took < 20
        result = _wait1(ex, _call(HOG, mb=50))
        assert result.ok and result.value == 50, result.error
    finally:
        ex.close()


def test_memory_a_block_s_child_process_uses_counts(kind):
    ex = make_executor(kind, limits=TaskLimits(memory_mb=300))
    try:
        result, took = _timed(ex, _call(CHILD_HOG, mb=600))
        assert not result.ok and result.crashed and "memory limit of 300 MB" in result.error
        assert took < 20
    finally:
        ex.close()


def test_one_huge_allocation_fails_at_once_naming_the_limit(kind):
    ex = make_executor(kind, limits=TaskLimits(memory_mb=300))
    try:
        result, took = _timed(ex, _call("def big():\n    import numpy as np\n    return float(np.ones(10**11).sum())\n"))
        assert not result.ok and not result.crashed
        assert result.error.startswith("MemoryError") and "memory limit of 300 MB" in result.error
        assert took < 20
    finally:
        ex.close()


def test_too_many_threads_or_processes_fail_fast(kind):
    ex = make_executor(kind, limits=TaskLimits(threads=64))
    try:
        for task in (_call(THREADS, n=300), _call(FORKS, n=100)):
            result, took = _timed(ex, task)
            assert not result.ok and result.crashed
            assert result.error == "stopped: the block ran more than its limit of 64 threads and processes"
            assert took < 20
        result = _wait1(ex, _call(THREADS, n=0))
        assert result.ok and result.value == 0, result.error
    finally:
        ex.close()


def test_a_killed_task_leaves_no_process_behind(kind, tmp_path):
    code = (
        "def spawn(path):\n"
        "    import os, time\n"
        "    pid = os.fork()\n"
        "    if pid == 0:\n"
        "        time.sleep(60)\n"
        "        os._exit(0)\n"
        "    open(path, 'w').write(str(pid))\n"
        "    junk = [bytearray(b'x' * (1024 * 1024)) for _ in range(4000)]\n"
    )
    ex = make_executor(kind, limits=TaskLimits(memory_mb=300))
    try:
        result = _wait1(ex, _call(code, path=str(tmp_path / "pid")))
        assert result.crashed and "memory limit" in result.error
    finally:
        ex.close()
    child = int((tmp_path / "pid").read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{child}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":  # killed; its reaper (this sandbox's init, if any) is another matter
            break
        time.sleep(0.05)
    else:
        pytest.fail("the task's child outlived it")


def test_ordinary_heavy_blocks_pass_under_the_server_s_default_budget(kind):
    ex = make_executor(kind, limits=SERVER_DEFAULT)
    try:
        result = _wait1(ex, _call(HEAVY))
        assert result.ok, result.error
        assert result.value == {"rows": 300_000, "groups": 5_000, "coef": 3}
    finally:
        ex.close()


def test_the_task_process_applies_the_backstops(kind):
    limits = TaskLimits(memory_mb=1000, total_threads=5000)
    if kind == "remote":  # total_threads is the agent's own
        ex = RemoteExecutor(lambda: spawn_agent(warm_workers=0, limits=limits))
    else:
        ex = make_executor(kind, limits=limits)
    try:
        result = _wait1(ex, _call(PROBE))
        assert result.ok, result.error
        assert result.value == {"pgid_is_own": True, "as": address_space_mb(1000) * 1024 * 1024, "nproc": 5000}
    finally:
        ex.close()


def test_without_limits_nothing_changes(kind):
    import resource

    ex = make_executor(kind)
    try:
        result = _wait1(ex, _call(PROBE))
        assert result.ok, result.error
        assert result.value == {
            "pgid_is_own": False,
            "as": resource.getrlimit(resource.RLIMIT_AS)[0],
            "nproc": resource.getrlimit(resource.RLIMIT_NPROC)[0],
        }
        # No watchdog: a big (but finite) allocation and many threads run.
        assert _wait1(ex, _call(HOG, mb=400)).value == 400
    finally:
        ex.close()


# ---- the agent's own limits ---------------------------------------------------------


def test_the_agent_s_own_limits_apply_and_a_client_can_only_lower_them():
    ex = RemoteExecutor(lambda: spawn_agent(warm_workers=0, limits=TaskLimits(memory_mb=300)), limits=TaskLimits(memory_mb=4000))
    try:
        result = _wait1(ex, _call(HOG, mb=1000))
        assert result.crashed and "memory limit of 300 MB" in result.error
    finally:
        ex.close()
    ex = RemoteExecutor(lambda: spawn_agent(warm_workers=0, limits=TaskLimits(memory_mb=4000)), limits=TaskLimits(memory_mb=200))
    try:
        result = _wait1(ex, _call(HOG, mb=1000))
        assert result.crashed and "memory limit of 200 MB" in result.error
    finally:
        ex.close()


def test_agent_command_carries_the_limits():
    cmd = agent_command(3, limits=TaskLimits(memory_mb=100, threads=20, total_threads=64))
    tail = cmd[cmd.index("--fd") :]
    assert tail == ["--fd", "3", "--task-memory-mb", "100", "--task-threads", "20", "--total-threads", "64"]
    assert "--task-memory-mb" not in agent_command(3)


@pytest.mark.parametrize("bad", [{"memory_mb": 0}, {"memory_mb": 1.5}, {"threads": True}, {"threads": "8"}])
def test_the_agent_rejects_bad_limits(bad):
    from modelmaker.execution.agent import _task_fields

    with pytest.raises(ValueError, match="bad"):
        _task_fields({"entry": "run_worker_entry", "limits": bad}, frozenset({"run_worker_entry"}))
    # Absent: no limit from the client (an older client, or none set).
    assert not _task_fields({"entry": "run_worker_entry", "limits": {}}, frozenset({"run_worker_entry"}))[5]


@pytest.mark.parametrize("flag", ["--task-memory-mb", "--task-threads", "--total-threads"])
def test_the_agent_refuses_a_bad_limit_option(flag, capsys):
    from modelmaker.execution import agent

    with pytest.raises(SystemExit):
        agent.main(["--fd", "0", flag, "0"])
    assert "must be positive" in capsys.readouterr().err


def test_an_old_style_task_message_without_limits_still_runs():
    """A worker gets a 4-tuple (no limits) exactly as before."""
    import multiprocessing

    from modelmaker.worker_pool import Worker

    worker = Worker(multiprocessing.get_context("spawn"), [])
    try:
        worker.conn.send(("run_worker_entry", ("f", "def f():\n    return 7\n", {}, None), dict(os.environ), os.getcwd()))
        from modelmaker.execution import wire

        assert wire.decode_frame(worker.recv_frame(10**6))[1] == 7
    finally:
        worker.stop()
