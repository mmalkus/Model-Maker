"""The Executor seam (modelmaker/execution): LocalExecutor keeps the
Runner's old dispatch behaviour -- concurrency cap, Stop, crash detection,
the warm spare pool -- and the Runner goes through whatever executor it's
given."""

from __future__ import annotations

import sys
import threading
import time

import polars as pl
import pytest

from modelmaker.cache import CacheStore
from modelmaker.execution import Executor, Task, TaskCancelled, TaskHandle, TaskResult
from modelmaker.execution.local import LocalExecutor
from modelmaker.graph import Graph, Wire
from modelmaker.runner import MP_CONTEXT, Runner
from modelmaker.worker_pool import WorkerPool

from .helpers import make_block


def _call(code: str, **kwargs) -> Task:
    """A run_worker_entry task calling the first function in `code`."""
    name = code.split("def ", 1)[1].split("(", 1)[0]
    return Task("run_worker_entry", (name, code, kwargs, None))


@pytest.fixture
def executor():
    ex = LocalExecutor(WorkerPool(MP_CONTEXT, 0))
    yield ex
    ex.close()


def test_results_come_back_in_handle_order(executor):
    code = "def echo(x):\n    return {'x': x, 'frame': pl.DataFrame({'v': [x]})}\n"
    handles = [executor.submit(_call(code, x=i)) for i in range(4)]
    results = executor.wait(handles, threading.Event(), max_concurrent=2)
    assert [r.handle for r in results] == handles
    assert all(r.ok for r in results)
    assert [r.value["x"] for r in results] == [0, 1, 2, 3]
    assert results[2].value["frame"].equals(pl.DataFrame({"v": [2]}))


def test_a_raising_block_is_a_failed_result_not_a_crash(executor):
    code = "def boom():\n    raise ValueError('bad input')\n"
    (result,) = executor.wait([executor.submit(_call(code))], threading.Event())
    assert not result.ok and not result.crashed
    assert result.error == "ValueError: bad input"


def test_a_dying_worker_is_reported_as_a_crash(executor):
    code = "def die():\n    import os\n    os._exit(3)\n"
    start = time.monotonic()
    (result,) = executor.wait([executor.submit(_call(code))], threading.Event())
    assert time.monotonic() - start < 30
    assert not result.ok and result.crashed
    assert "unexpectedly" in result.error and "code 3" in result.error


def test_max_concurrent_caps_how_many_tasks_run_at_once(executor, tmp_path):
    code = (
        "def stamp(path):\n"
        "    import time\n"
        "    start = time.time(); time.sleep(0.6)\n"
        "    open(path, 'w').write(f'{start} {time.time()}')\n"
        "    return 1\n"
    )
    paths = [tmp_path / f"t{i}" for i in range(4)]
    results = executor.wait([executor.submit(_call(code, path=str(p))) for p in paths], threading.Event(), max_concurrent=2)
    assert all(r.ok for r in results), [r.error for r in results]
    spans = [tuple(map(float, p.read_text().split())) for p in paths]
    for t, _ in spans:
        overlapping = sum(1 for s, e in spans if s <= t < e)
        assert overlapping <= 2


def test_cancel_stops_every_task_promptly(executor, tmp_path):
    code = "def slow(path):\n    import time\n    time.sleep(20)\n    open(path, 'w').write('done')\n"
    cancel = threading.Event()
    handles = [executor.submit(_call(code, path=str(tmp_path / f"t{i}"))) for i in range(3)]
    threading.Timer(1.0, cancel.set).start()
    start = time.monotonic()
    with pytest.raises(TaskCancelled):
        executor.wait(handles, cancel, max_concurrent=3)
    assert time.monotonic() - start < 10
    time.sleep(0.5)
    assert not any((tmp_path / f"t{i}").exists() for i in range(3))


def test_a_result_over_the_cap_is_rejected_by_the_parent(tmp_path):
    ex = LocalExecutor(WorkerPool(MP_CONTEXT, 0), max_result_bytes=1024**2)
    try:
        code = "def big():\n    return pl.DataFrame({'a': list(range(500_000))})\n"
        (result,) = ex.wait([ex.submit(_call(code))], threading.Event())
    finally:
        ex.close()
    assert not result.ok and not result.crashed
    assert "over the" in result.error and "limit" in result.error


def test_a_result_over_the_cap_fails_in_the_task_process_with_the_setting_named(executor, monkeypatch):
    monkeypatch.setenv("MODELMAKER_MAX_RESULT_MB", "1")
    code = "def big():\n    return b'x' * (3 * 1024 * 1024)\n"
    (result,) = executor.wait([executor.submit(_call(code))], threading.Event())
    assert not result.ok
    assert "MODELMAKER_MAX_RESULT_MB" in result.error


def test_warm_spares_are_used_and_learn_imports():
    pool = WorkerPool(MP_CONTEXT, 1)
    ex = LocalExecutor(pool)
    try:
        ex.prewarm()
        deadline = time.monotonic() + 60
        while not pool._spares[0].is_ready() and time.monotonic() < deadline:
            time.sleep(0.05)
        spare = pool._spares[0]
        code = "def uses_csv():\n    import csv, json\n    return 'ok'\n"
        (result,) = ex.wait([ex.submit(_call(code))], threading.Event())
        assert result.ok and result.value == "ok"
        assert spare not in pool._spares  # it was taken, and replaced
        assert len(pool._spares) == 1
        assert "csv" in pool._learned
    finally:
        ex.close()


@pytest.mark.parametrize("launcher", ["exec", "multiprocessing"])
def test_both_launchers_run_tasks(launcher):
    ex = LocalExecutor(WorkerPool(MP_CONTEXT, 0, launcher=launcher))
    try:
        code = "def tup():\n    return pl.DataFrame({'a': [1]}), {'n': 1}\n"
        (result,) = ex.wait([ex.submit(_call(code))], threading.Event())
    finally:
        ex.close()
    assert result.ok, result.error
    df, meta = result.value
    assert df.equals(pl.DataFrame({"a": [1]})) and meta == {"n": 1}


class _RecordingExecutor:
    """Runs nothing in a subprocess: proves the Runner only talks to the
    executor it was given."""

    def __init__(self, inner: LocalExecutor) -> None:
        self.inner = inner
        self.entries: list[str] = []

    def submit(self, task: Task) -> TaskHandle:
        self.entries.append(task.entry)
        return self.inner.submit(task)

    def wait(self, handles, cancel, max_concurrent=None) -> list[TaskResult]:
        return self.inner.wait(handles, cancel, max_concurrent)

    def close(self) -> None:
        self.inner.close()


def test_the_runner_dispatches_through_its_executor(tmp_path, executor):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,10\n2,20\n3,30\n")
    graph = Graph(
        blocks={
            "b_read": make_block("b_read", "read_csv", params={"path": str(csv)}),
            "b_filter": make_block("b_filter", "filter", params={"expr": "a > 1"}, x=1),
        },
        wires={"w": Wire("w", "b_read", "out", "b_filter", "df")},
    )
    recording = _RecordingExecutor(executor)
    assert isinstance(recording, Executor)
    runner = Runner(graph, CacheStore(), executor=recording)
    assert runner.refresh("b_read") == "green"
    assert runner.run_block("b_filter") == "green", runner.state["b_filter"].last_error
    assert recording.entries == ["run_worker_entry", "run_worker_entry"]
    # A streaming run's fused group (read_csv -> filter) goes through it too.
    recording.entries.clear()
    streaming = Runner(graph, CacheStore(), executor=recording)
    report = streaming.run_all_streaming()
    assert recording.entries == ["run_fused_group_entry"], report


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell background job")
def test_a_crash_is_detected_even_if_the_block_left_a_child_process_running(executor):
    # A background job the block started outlives the worker; it must not
    # keep the worker's connection open and turn the crash into a hang.
    code = "def orphan():\n    import os\n    os.system('sleep 20 &')\n    os._exit(1)\n"
    start = time.monotonic()
    (result,) = executor.wait([executor.submit(_call(code))], threading.Event())
    assert time.monotonic() - start < 10
    assert result.crashed
