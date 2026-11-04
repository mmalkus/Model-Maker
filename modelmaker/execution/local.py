"""LocalExecutor: tasks run in single-use processes on this machine.

This is the behaviour the Runner always had -- a warm spare pool, one fresh
process per task, `max_concurrent` at a time, Stop that kills them, and a
worker that dies without reporting surfacing as a normal red-block error --
behind the Executor interface (base.py), with two differences that make
the local app safe against its own block code too:

- results come back as wire frames (wire.py), decoded here without
  unpickling anything the block made (F4);
- the task processes are started without this process's secrets in their
  environment (env.py, worker_pool.Worker) (F3).
"""

from __future__ import annotations

import itertools
import multiprocessing.connection
import re
import threading
import time

from ..worker_pool import Worker, WorkerPool, pool_size_from_env
from . import wire
from .base import Task, TaskCancelled, TaskHandle, TaskResult
from .env import harden, harden_requested

# How often the wait loop wakes up to check for a cancellation request or
# a worker that died without reporting a result -- small enough that Stop
# and crash detection both feel immediate, large enough not to busy-loop.
POLL_INTERVAL = 0.05

# What the result envelope may carry besides the value (see
# worker_pool._ConnResult): an error message is shown on the block, so cap
# it; learned module names are preloaded into future spares, so they must
# look like module names.
MAX_ERROR_CHARS = 20_000
MAX_LEARNED_PER_TASK = 5_000
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


class LocalExecutor:
    """Runs each Task in a single-use worker process from `pool` (see
    worker_pool.py). Thread-safe: the API server runs several dispatches at
    once, from different threads, through the one shared executor."""

    def __init__(
        self, pool: WorkerPool | None = None, max_result_bytes: int | None = None, task_timeout: float | None = None
    ) -> None:
        """`task_timeout`: wall-clock seconds a task may run once started;
        one still running then is stopped and fails with a message saying
        so (None or 0: no limit, the local app's default)."""
        if pool is None:
            import multiprocessing

            pool = WorkerPool(multiprocessing.get_context("spawn"), pool_size_from_env())
        self.pool = pool
        # None: read MODELMAKER_MAX_RESULT_MB at each wait (see wire.py).
        self.max_result_bytes = max_result_bytes
        self.task_timeout = task_timeout or None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        if harden_requested():
            harden()

    @property
    def base_environ(self):
        """The environment this executor's tasks start from, before their
        own variables (None: this process's, minus the secrets) -- so that
        what runs outside a task on a task's behalf, like a source's
        change probe, sees what the task would."""
        return self.pool.launch.environ

    def prewarm(self) -> None:
        self.pool.prewarm()

    def submit(self, task: Task) -> TaskHandle:
        with self._lock:
            return TaskHandle(next(self._ids), task)

    def wait(
        self, handles: list[TaskHandle], cancel: threading.Event, max_concurrent: int | None = None
    ) -> list[TaskResult]:
        limit = len(handles) if max_concurrent is None else max(1, max_concurrent)
        max_bytes = self.max_result_bytes if self.max_result_bytes is not None else wire.max_bytes_from_env()
        pending = list(handles)
        running: dict[TaskHandle, Worker] = {}
        started: dict[TaskHandle, float] = {}
        results: dict[TaskHandle, TaskResult] = {}
        timeout = self.task_timeout

        def _start_next() -> None:
            # Acquire every free slot's worker before sending any task: a
            # send can block until its worker has finished importing, and
            # sending one at a time would serialize the cold starts.
            batch = []
            while pending and len(running) < limit:
                handle = pending.pop(0)
                running[handle] = self.pool.acquire()
                batch.append(handle)
            for handle in batch:
                started[handle] = time.monotonic()
                try:
                    running[handle].send_task(handle.task.entry, handle.task.args, handle.task.env)
                except (BrokenPipeError, EOFError, OSError):
                    pass  # died before taking the task -- reported as a crash by the loop below

        try:
            _start_next()
            while running:
                if cancel.is_set():
                    raise TaskCancelled("cancelled")
                by_conn = {w.conn: h for h, w in running.items()}
                # A worker that dies closes its end of the connection, which
                # also makes it ready -- reading then raises EOFError.
                ready = multiprocessing.connection.wait(list(by_conn), timeout=POLL_INTERVAL)
                for conn in ready:
                    handle = by_conn[conn]
                    worker = running.pop(handle)
                    results[handle] = self._receive(handle, worker, max_bytes)
                    worker.stop()
                # Belt and braces: a process that has exited with nothing
                # left to read is a crash even if something it started
                # still holds its end of the connection open.
                for handle, worker in list(running.items()):
                    if not worker.process.is_alive() and not worker.conn.poll():
                        running.pop(handle)
                        results[handle] = TaskResult(handle, ok=False, error=worker.crash_message(), crashed=True)
                        worker.stop()
                if timeout is not None:
                    now = time.monotonic()
                    for handle, worker in list(running.items()):
                        if now - started[handle] > timeout:
                            running.pop(handle)
                            worker.stop()
                            results[handle] = TaskResult(
                                handle, ok=False, error=f"stopped: the block ran longer than its {timeout:g} s time limit"
                            )
                _start_next()
        finally:
            for worker in running.values():
                worker.stop()
        return [results[h] for h in handles]

    def _receive(self, handle: TaskHandle, worker: Worker, max_bytes: int) -> TaskResult:
        try:
            frame = worker.recv_frame(max_bytes)
        except EOFError:
            return TaskResult(handle, ok=False, error=worker.crash_message(), crashed=True)
        except wire.WireError as e:
            return TaskResult(handle, ok=False, error=f"result rejected: {e}")
        try:
            fields, value = wire.decode_frame(frame, max_bytes=max_bytes)
        except wire.WireError as e:
            return TaskResult(handle, ok=False, error=f"result rejected: {e}")
        except Exception as e:  # noqa: BLE001 -- a decoder bug must still be a red block, not a dead dispatch
            return TaskResult(handle, ok=False, error=f"result rejected: {type(e).__name__}: {e}")
        modules = fields.get("modules")
        if isinstance(modules, list):
            self.pool.learn(
                [m for m in modules[:MAX_LEARNED_PER_TASK] if isinstance(m, str) and len(m) < 200 and _MODULE_NAME.match(m)]
            )
        if fields.get("ok") is True:
            return TaskResult(handle, ok=True, value=value)
        error = fields.get("error")
        error = error if isinstance(error, str) and error else "the block failed without an error message"
        if len(error) > MAX_ERROR_CHARS:
            error = error[:MAX_ERROR_CHARS] + "…"
        return TaskResult(handle, ok=False, error=error)

    def close(self) -> None:
        self.pool.shutdown()
