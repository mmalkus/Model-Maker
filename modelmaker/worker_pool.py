"""Pre-started, single-use worker processes for block runs (see
execution/local.py's LocalExecutor, which Runner._run_tasks in runner.py
goes through).

Every task still runs in a process of its own that is used for exactly one
task and then exits -- the isolation contract described in run_worker.py
is unchanged, and nothing a block does can leak into another block's run.
What changes is only *when* that process starts. A fresh interpreter
spends almost all of its life importing polars, and sklearn/scipy for the
modelling and test blocks, before doing a few milliseconds of actual work:
~0.6s for a filter, ~3s for a logistic regression. So a few spares are
started, and their imports done, ahead of time, and a dispatch hands its
task to one that is already warm instead of paying that start-up in line.
Each spare taken is replaced in the background.

How a worker process is started (see Worker): on POSIX it is exec'd
directly -- `subprocess.Popen` of this interpreter with a small bootstrap,
one end of a socketpair passed as its only extra fd -- with the secrets
already removed from its environment (execution/env.py). That is the point
of not using multiprocessing's "spawn" here: spawn execs the child with
this process's *current* environment, and deleting variables inside the
child afterwards leaves them readable in /proc/self/environ, which keeps
whatever the process was exec'd with. Popen takes the environment as an
argument, so nothing has to mutate this process's os.environ around a
start either (which wouldn't be thread-safe: the API dispatches from
several threads). Elsewhere -- Windows, a frozen app -- it falls back to
multiprocessing's spawn (`runner.MP_CONTEXT`) and removes the secrets from
os.environ first thing in the child: good against a block reading
os.environ, not against one reading the process's original environment
block. Either way, every message the parent *receives* is bytes it decodes
itself (execution/wire.py), never a pickle.

A spare is started before it is known what it will run, so it can't rely
on what it inherited at start: the parent's *current* (scrubbed)
os.environ and cwd travel with each task, together with the task's own
variables (the ones the user set from the UI for their workspace, see
execution.base.Task.env), and are applied before the task runs. The one thing that can't be carried over is a module that
reads the environment at import time -- polars' POLARS_MAX_THREADS, say --
which in a spare sees the environment as of the spare's start.

An application that runs tasks on someone else's behalf -- a server hosting
several users' workspaces in one process -- can change three things about
how its workers start, with a `WorkerLaunch` (exec launcher only): a
command prefix the interpreter is exec'd through (e.g. `setpriv` to run as
another uid), the base environment used *instead of* this process's
os.environ (still scrubbed of secrets), and the working directory each task
runs in instead of this process's cwd. A pool has one launch, so spares
started for one launch never serve another.
"""

from __future__ import annotations

import atexit
import importlib
import os
import socket
import subprocess
import sys
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .execution.env import is_secret, scrubbed_environ, task_environ

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

Launcher = Literal["exec", "multiprocessing"]


@dataclass(frozen=True)
class WorkerLaunch:
    """How a pool's workers are started, when not the default (see the
    module docstring). Every field is optional:

    - `command_prefix`: argv the interpreter command is appended to, e.g.
      ("setpriv", "--reuid=20001", ..., "--"). The prefix must exec the
      command it's given with the inherited file descriptors open.
    - `environ`: the base environment for the worker process and each of
      its tasks, used instead of this process's os.environ. Secrets are
      still dropped from it (env.scrubbed_environ), and a task's own
      variables still go on top.
    - `cwd`: the directory each task runs in, instead of this process's
      cwd at send time. It's entered by the worker itself, after any
      prefix took effect, so it only needs to be accessible to the worker.

    Needs the exec launcher: multiprocessing's spawn can't do any of it."""

    command_prefix: tuple[str, ...] = ()
    environ: Mapping[str, str] | None = None
    cwd: str | None = None

    @property
    def is_default(self) -> bool:
        return not self.command_prefix and self.environ is None and self.cwd is None


def default_launcher() -> Launcher:
    """The exec launcher wherever Popen can pass the socket's fd (POSIX),
    unless this is a frozen app, where sys.executable isn't a Python that
    runs `-c`; multiprocessing's spawn everywhere else.
    MODELMAKER_WORKER_LAUNCHER=multiprocessing forces the fallback (an
    escape hatch, and how the fallback gets tested on POSIX)."""
    if os.environ.get("MODELMAKER_WORKER_LAUNCHER") == "multiprocessing":
        return "multiprocessing"
    if os.name == "posix" and sys.executable and not getattr(sys, "frozen", False):
        return "exec"
    return "multiprocessing"


# ---- the worker process's side ----------------------------------------------


def _preload(names: list[str]) -> None:
    for name in names:
        try:
            importlib.import_module(name)
        except BaseException:  # noqa: BLE001 -- a preload is only an optimization; the task imports for real
            pass


class _ConnResult:
    """Stands in for the result queue the run_worker entry points report
    to: encodes `(task_key, ok, payload)` into one wire frame (see
    execution/wire.py) and sends it back over this worker's socket as raw
    bytes. The frame also lists every module the task imported that the
    worker hadn't preloaded -- the parent preloads those into future spares
    (see WorkerPool.learn), which is how a session that keeps running
    logistic regressions stops paying for the sklearn import after the
    first.

    A result that can't be encoded (an object that isn't data, or one over
    the size cap) becomes the task's error here, in the task process, with
    a message saying why -- it never reaches the parent at all."""

    def __init__(self, conn: Any, modules_before: set[str]) -> None:
        self._conn = conn
        self._before = modules_before

    def put(self, item: tuple[Any, bool, Any]) -> None:
        from .execution import wire

        _key, ok, payload = item
        learned = [m for m in sys.modules if m not in self._before and not m.startswith("__")]
        fields = {"ok": ok, "error": None if ok else str(payload), "modules": learned}
        try:
            frame = wire.encode_frame(fields, payload if ok else None)
        except (wire.WireError, RecursionError) as e:
            fields.update(ok=False, error=str(e) if isinstance(e, wire.WireError) else "RecursionError: result nested too deeply")
            frame = wire.encode_frame(fields, None)
        self._conn.send_bytes(frame)


# What a spare sends once its preload is done (see WorkerPool.acquire).
READY = b"ready"


def worker_main(conn: Any, preload: list[str], announce_ready: bool) -> None:
    """A worker process's whole life: import `preload`, wait for exactly
    one task, run it, report, exit. A parent that discards a spare unused
    just closes its end of the connection, which ends this at recv()."""
    _preload(preload)
    if announce_ready:
        conn.send_bytes(READY)
    modules_before = set(sys.modules)
    try:
        entry_name, args, env, cwd = conn.recv()  # from the parent: trusted
    except (EOFError, OSError):
        return
    os.environ.clear()
    os.environ.update(env)
    os.chdir(cwd)
    from . import block_library, run_worker

    # This process exists to run block code; the API process never does
    # (see block_library.exec_block_source).
    block_library.enter_task_process()

    result_queue = _ConnResult(conn, modules_before)
    try:
        from .execution import wire

        if isinstance(args, wire.EncodedArgs):
            # From a relay (the runner agent, execution/agent.py) that
            # never decodes them itself: decoded here, by the same
            # validating decoder results go through.
            try:
                args = wire.decode(args.data)
            except (wire.WireError, RecursionError) as e:
                result_queue.put((None, False, f"task rejected: {e}"))
                return
            if not isinstance(args, tuple):
                result_queue.put((None, False, "task rejected: its args aren't a tuple"))
                return
        getattr(run_worker, entry_name)(*args, result_queue=result_queue, task_key=None)
    except (BrokenPipeError, EOFError, OSError):
        pass  # the parent stopped waiting (cancelled, or gone)


def _mp_child_main(conn: Any, preload: list[str], announce_ready: bool) -> None:
    """The multiprocessing fallback's target: this process was started with
    the parent's whole environment, so drop the secrets from os.environ
    before anything else runs. Best effort -- see the module docstring."""
    for name in [n for n in os.environ if is_secret(n)]:
        os.environ.pop(name, None)
    worker_main(conn, preload, announce_ready)


# The exec launcher's `-c` program: the fd of its end of the socketpair is
# argv[1]; the first message on it (pickled -- it's from the parent) is
# this process's sys.path, so modelmaker imports exactly as in the parent,
# plus what to preload. Runs under -P so nothing in the cwd can shadow the
# stdlib modules imported before sys.path is replaced. The fd is made
# non-inheritable again (pass_fds made it inheritable), so a program the
# block starts doesn't hold the socket open after this process dies.
_BOOTSTRAP = """\
import os, sys
from multiprocessing.connection import Connection
os.set_inheritable(int(sys.argv[1]), False)
conn = Connection(int(sys.argv[1]))
try:
    sys.path[:], preload, announce_ready = conn.recv()
except (EOFError, OSError):
    raise SystemExit(0)
from modelmaker.worker_pool import worker_main
worker_main(conn, preload, announce_ready)
"""


# ---- the parent's side --------------------------------------------------------


class _PopenProcess:
    """A Popen with the slice of multiprocessing.Process's interface the
    pool and the executor use."""

    def __init__(self, popen: subprocess.Popen) -> None:
        self._popen = popen
        self.pid = popen.pid

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    @property
    def exitcode(self) -> int | None:
        return self._popen.poll()

    def terminate(self) -> None:
        try:
            self._popen.terminate()
        except OSError:
            pass

    def kill(self) -> None:
        try:
            self._popen.kill()
        except OSError:
            pass

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout)
        except subprocess.TimeoutExpired:
            pass


# Every worker not yet stopped, so none outlives this process (the
# multiprocessing launcher's daemon flag did this for free; a Popen child
# doesn't get it). A spare would exit on its own once its socket closes,
# but a task still running wouldn't, until it finished.
_LIVE: weakref.WeakSet[Worker] = weakref.WeakSet()
_LIVE_LOCK = threading.Lock()


@atexit.register
def _stop_live_workers() -> None:
    with _LIVE_LOCK:
        workers = list(_LIVE)
    for worker in workers:
        worker.stop()


def _interpreter_flags() -> list[str]:
    flags = list(getattr(subprocess, "_args_from_interpreter_flags", list)())
    return flags if "-P" in flags else [*flags, "-P"]


class Worker:
    """The parent's handle on one (possibly still preloading) worker:
    `conn` (send to it pickled, receive from it bytes only) and `process`."""

    def __init__(
        self,
        ctx: Any,
        preload: list[str],
        spare: bool = False,
        launcher: Launcher | None = None,
        launch: WorkerLaunch | None = None,
    ) -> None:
        self._ready = False
        self.launch = launch or WorkerLaunch()
        self.launcher: Launcher = launcher or default_launcher()
        if not self.launch.is_default and self.launcher != "exec":
            raise ValueError("a WorkerLaunch needs the exec launcher (POSIX, not a frozen app)")
        if self.launcher == "exec":
            from multiprocessing.connection import Connection

            parent_sock, child_sock = socket.socketpair()
            try:
                popen = subprocess.Popen(
                    [
                        *self.launch.command_prefix,
                        sys.executable,
                        *_interpreter_flags(),
                        "-c",
                        _BOOTSTRAP,
                        str(child_sock.fileno()),
                    ],
                    env=scrubbed_environ(self.launch.environ),
                    pass_fds=(child_sock.fileno(),),
                    stdin=subprocess.DEVNULL,
                )
            except BaseException:
                parent_sock.close()
                raise
            finally:
                # The parent must drop its copy of the child's end, or the
                # socket never reports EOF when the child dies and a crash
                # would read as a hang.
                child_sock.close()
            self.conn = Connection(parent_sock.detach())
            self.process: Any = _PopenProcess(popen)
            try:
                self.conn.send((list(sys.path), list(preload), spare))
            except OSError:
                pass  # died at start -- reported as a crash when its task is waited on
        else:
            self.conn, child_conn = ctx.Pipe(duplex=True)
            self.process = ctx.Process(target=_mp_child_main, args=(child_conn, preload, spare), daemon=True)
            self.process.start()
            child_conn.close()  # see above
        with _LIVE_LOCK:
            _LIVE.add(self)

    def is_ready(self) -> bool:
        """For a spare: whether it has finished its preload (never blocks)."""
        if not self._ready:
            try:
                self._ready = self.conn.poll() and self.conn.recv_bytes(len(READY)) == READY
            except (EOFError, OSError):
                return False
        return self._ready

    def send_task(
        self, entry_name: str, args: Any, env: Mapping[str, str] | None = None, cwd: str | None = None
    ) -> None:
        """Hand this worker its one task, with the environment it runs in:
        this process's current one minus the secrets, plus the task's own
        variables (`env`, see execution.base.Task) -- also minus the
        secrets. With a WorkerLaunch, its `environ` and `cwd` stand in for
        this process's; an explicit `cwd` stands in for both. `args` is the
        args tuple, or an execution.wire.EncodedArgs the worker decodes."""
        if cwd is None:
            cwd = self.launch.cwd if self.launch.cwd is not None else os.getcwd()
        self.conn.send((entry_name, args, task_environ(env, self.launch.environ), cwd))

    def recv_frame(self, max_bytes: int) -> bytes:
        """The task's result frame (raw bytes, for execution/wire.py to
        decode). Raises EOFError when the worker died without reporting,
        and wire.ValueTooLarge when it announces more than `max_bytes`
        (checked before a byte of the payload is read)."""
        from .execution.wire import ValueTooLarge

        try:
            return self.conn.recv_bytes(max_bytes)
        except OSError as e:
            if str(e) == "bad message length":
                raise ValueTooLarge(f"block result is over the {max_bytes / 1024**2:,.1f} MB limit") from None
            raise EOFError(str(e)) from None

    def crash_message(self) -> str:
        self.process.join(timeout=5)
        return f"worker process exited unexpectedly (code {self.process.exitcode}) -- likely out of memory or a crash"

    def stop(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive() and hasattr(self.process, "kill"):
            self.process.kill()
            self.process.join(timeout=5)
        self.conn.close()
        with _LIVE_LOCK:
            _LIVE.discard(self)


class WorkerPool:
    """Keeps `size` spares started ahead of use. Thread-safe: the API
    server dispatches from several background threads at once."""

    def __init__(self, ctx: Any, size: int, launcher: Launcher | None = None, launch: WorkerLaunch | None = None) -> None:
        self._ctx = ctx
        self.size = size
        self.launcher: Launcher = launcher or default_launcher()
        # How every worker of this pool starts (see WorkerLaunch); spares
        # are per pool, so they never cross from one launch to another.
        self.launch = launch or WorkerLaunch()
        if not self.launch.is_default and self.launcher != "exec":
            raise ValueError("a WorkerLaunch needs the exec launcher (POSIX, not a frozen app)")
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
        return worker or Worker(self._ctx, list(BASE_PRELOAD), launcher=self.launcher, launch=self.launch)

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
            self._spares.append(Worker(self._ctx, preload, spare=True, launcher=self.launcher, launch=self.launch))

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
