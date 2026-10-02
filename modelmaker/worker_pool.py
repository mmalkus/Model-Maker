"""Pre-started, single-use worker processes for block runs (see
Runner._run_tasks in runner.py).

Every task still runs in a process of its own that is used for exactly one
task and then exits -- the isolation contract described in run_worker.py
is unchanged, and nothing a block does can leak into another block's run.
What changes is only *when* that process starts. A "spawn" child (see
runner.MP_CONTEXT) spends almost all of its life importing polars, and
sklearn/scipy for the modelling and test blocks, before doing a few
milliseconds of actual work: ~0.6s for a filter, ~3s for a logistic
regression. So a few spares are started, and their imports done, ahead of
time, and a dispatch hands its task to one that is already warm instead
of paying that start-up in line. Each spare taken is replaced in the
background.

A spare is started before it is known what it will run, so it can't rely
on what it inherited at spawn time: the parent's *current* os.environ and
cwd travel with each task and are applied before the task runs (e.g.
api.set_env_var relies on a block seeing an env var set after server
start). The one thing that can't be carried over is a module that reads
the environment at import time -- polars' POLARS_MAX_THREADS, say -- which
in a spare sees the environment as of the spare's start.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from typing import Any

# Imported by every worker before it takes a task: what nearly every block
# run needs (the block modules themselves register the registry blocks --
# see run_worker_entry).
BASE_PRELOAD = (
    "polars",
    "modelmaker.blocks.binning",
    "modelmaker.blocks.data_quality",
    "modelmaker.blocks.feature_analysis",
    "modelmaker.blocks.library",
    "modelmaker.blocks.modelling",
    "modelmaker.blocks.stat_tests",
    "modelmaker.blocks.stochastic",
)

# Upper bound on the learned preload list (module names, not packages: a
# single `from sklearn.linear_model import ...` pulls in hundreds).
MAX_LEARNED_MODULES = 5000


def _preload(names: list[str]) -> None:
    for name in names:
        try:
            importlib.import_module(name)
        except BaseException:  # noqa: BLE001 -- a preload is only an optimization; the task imports for real
            pass


class _ConnResult:
    """Stands in for the result queue the run_worker entry points report
    to, sending `(task_key, ok, payload)` back over this worker's pipe
    instead, plus every module the task imported that the worker hadn't
    preloaded -- the parent preloads those into future spares (see
    WorkerPool.learn), which is how a session that keeps running logistic
    regressions stops paying for the sklearn import after the first."""

    def __init__(self, conn: Any, modules_before: set[str]) -> None:
        self._conn = conn
        self._before = modules_before

    def put(self, item: tuple[Any, bool, Any]) -> None:
        learned = [m for m in sys.modules if m not in self._before and not m.startswith("__")]
        self._conn.send((*item, learned))


# What a spare sends once its preload is done (see WorkerPool.acquire).
READY = "ready"


def spare_main(conn: Any, preload: list[str], announce_ready: bool) -> None:
    """A worker process's whole life: import `preload`, wait for exactly
    one task, run it, report, exit. A parent that discards a spare unused
    just closes its end of the pipe, which ends this at recv()."""
    _preload(preload)
    if announce_ready:
        conn.send(READY)
    modules_before = set(sys.modules)
    try:
        entry_name, args, task_key, env, cwd = conn.recv()
    except (EOFError, OSError):
        return
    os.environ.clear()
    os.environ.update(env)
    os.chdir(cwd)
    from . import run_worker

    getattr(run_worker, entry_name)(*args, result_queue=_ConnResult(conn, modules_before), task_key=task_key)


class Worker:
    """The parent's handle on one (possibly still preloading) worker."""

    def __init__(self, ctx: Any, preload: list[str], spare: bool = False) -> None:
        self.conn, child_conn = ctx.Pipe(duplex=True)
        self.process = ctx.Process(target=spare_main, args=(child_conn, preload, spare), daemon=True)
        self._ready = False
        self.process.start()
        # The parent must drop its copy of the child's end, or the pipe
        # never reports EOF when the child dies and a crash would read as
        # a hang.
        child_conn.close()

    def is_ready(self) -> bool:
        """For a spare: whether it has finished its preload (never blocks)."""
        if not self._ready:
            try:
                self._ready = self.conn.poll() and self.conn.recv() == READY
            except (EOFError, OSError):
                return False
        return self._ready

    def send_task(self, entry_name: str, args: tuple[Any, ...], task_key: Any) -> None:
        self.conn.send((entry_name, args, task_key, dict(os.environ), os.getcwd()))

    def crash_message(self) -> str:
        self.process.join(timeout=5)
        return f"worker process exited unexpectedly (code {self.process.exitcode}) -- likely out of memory or a crash"

    def stop(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        self.conn.close()


class WorkerPool:
    """Keeps `size` spares started ahead of use. Thread-safe: the API
    server dispatches from several background threads at once."""

    def __init__(self, ctx: Any, size: int) -> None:
        self._ctx = ctx
        self.size = size
        self._spares: list[Worker] = []
        self._learned: dict[str, None] = {}  # insertion-ordered set: parents before submodules
        self._lock = threading.Lock()

    def acquire(self) -> Worker:
        """A worker to send one task to: a spare that has finished its
        preload if there is one, else a freshly started one (which costs
        what every dispatch used to). Never hands out a spare still mid-
        preload: the learned preload can be far heavier than the task needs
        (sklearn for a filter), and a burst of acquires -- a fan-out --
        would otherwise just pay it in line, worker after worker."""
        with self._lock:
            worker = None
            for candidate in list(self._spares):  # oldest first
                if not candidate.process.is_alive():
                    self._spares.remove(candidate)
                    candidate.stop()
                elif worker is None and candidate.is_ready():
                    self._spares.remove(candidate)
                    worker = candidate
            self._top_up_locked()
        # A cold start only preloads the base set: it's about to be told
        # what it runs, and the learned set may well be more than that needs.
        return worker or Worker(self._ctx, list(BASE_PRELOAD))

    def prewarm(self) -> None:
        """Start the spares now rather than at the first dispatch, so even
        the first block run of a fresh server finds one ready."""
        with self._lock:
            self._top_up_locked()

    def learn(self, modules: list[str]) -> None:
        with self._lock:
            for name in modules:
                if len(self._learned) >= MAX_LEARNED_MODULES:
                    break
                self._learned.setdefault(name)

    def _top_up_locked(self) -> None:
        preload = [*BASE_PRELOAD, *self._learned]
        while len(self._spares) < self.size:
            self._spares.append(Worker(self._ctx, preload, spare=True))

    def shutdown(self) -> None:
        with self._lock:
            spares, self._spares = self._spares, []
        for worker in spares:
            worker.stop()


def pool_size_from_env() -> int:
    """MODELMAKER_WARM_WORKERS: how many spares to keep started (0 turns
    pre-starting off -- every task then starts its own process in line)."""
    try:
        return max(0, int(os.environ.get("MODELMAKER_WARM_WORKERS", "2")))
    except ValueError:
        return 2
