"""The runner agent: runs block tasks for one client over one socket.

    python -m modelmaker.execution.agent --fd N          # an inherited, connected socket
    python -m modelmaker.execution.agent --listen PATH   # accept one connection on PATH
        [--socket-mode 0660]                             # who may connect (default 0600)
        [--task-memory-mb N] [--task-threads N] [--total-threads N]   # limits (default: none)
    modelmaker-runner ...                                 # the same, as a console script

A long-lived process on the far side of the execution boundary. It serves
exactly one client connection (a RemoteExecutor, remote.py) speaking the
runner protocol (protocol.py), and runs each task it is sent in a
single-use task process of its own -- the same warm spare pool, crash
detection, Stop and time limit as the local app, because inside it is
just a LocalExecutor (local.py) over a WorkerPool (worker_pool.py).

What it holds: nothing secret. Its environment is whatever it was started
with (a deployment starts it with a minimal one), each task's own
variables go on top minus the secret denylist (env.task_environ), and its
task processes are started from that, scrubbed again.

What it interprets: as little as it can. It reads each message's fields
(wire.read_header), never a value: a task's args go to the task process
still encoded (wire.EncodedArgs) and are decoded there, and a task's
result frame goes back to the client with only its fields replaced
(wire.reframe), to be decoded -- and validated -- by the client. So the
agent never unpickles, and never parses a dataframe.

Lifecycle:
- the client says hello, the agent answers (or refuses a protocol or
  version it doesn't speak, see protocol.check_hello);
- every task starts as soon as it arrives -- the client decides how many
  run at once -- in a thread of its own waiting on its task process;
- `cancel` stops that task's process (LocalExecutor's Stop: terminate,
  then kill) and answers with a cancelled result once it's gone;
- a task process that dies without reporting is a crashed result; one
  over its `wall_seconds` is stopped and fails saying so, and one over its
  `memory_mb` or `threads` (limits.py) is killed and crashes saying so --
  the stricter of the client's limit and the agent's own (--task-memory-mb,
  --task-threads), which the client can lower but not raise;
- `ping` is answered with `pong` (the client's heartbeat);
- `bye`, the client closing the socket, SIGTERM or SIGINT end the agent:
  every running task process is stopped and the spares discarded first.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import multiprocessing
import os
import signal
import socket
import sys
import threading
from typing import Any

from . import protocol, wire
from .base import Task, TaskCancelled
from .limits import TaskLimits, check_limit
from .local import LocalExecutor
from .protocol import ProtocolError

# How long shutdown waits for the task threads to stop their processes.
SHUTDOWN_GRACE = 10.0


class Agent:
    """One client connection, served until it ends (see the module
    docstring). `executor` must have `raw_results=True`."""

    def __init__(self, sock: socket.socket, executor: LocalExecutor, max_bytes: int | None = None) -> None:
        if not executor.raw_results:
            raise ValueError("the agent relays result frames: its executor needs raw_results=True")
        self.sock = sock
        self.executor = executor
        self.max_bytes = max_bytes if max_bytes is not None else wire.max_bytes_from_env()
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._tasks: dict[int, tuple[threading.Event, threading.Thread]] = {}

    # ---- sending

    def _send(self, frame: bytes) -> bool:
        with self._send_lock:
            try:
                protocol.send_msg(self.sock, frame)
                return True
            except OSError:
                return False  # the client is gone; the reader loop sees it too

    def _send_result(self, task_id: int, ok: bool, error: str | None = None, crashed: bool = False, cancelled: bool = False) -> None:
        if error is not None and len(error) > protocol.MAX_ERROR_CHARS:
            error = error[: protocol.MAX_ERROR_CHARS] + "…"
        self._send(protocol.message("result", task=task_id, ok=ok, error=error, crashed=crashed, cancelled=cancelled))

    # ---- serving

    def serve(self) -> None:
        try:
            if not self._handshake():
                return
            while True:
                frame = protocol.recv_msg(self.sock, self.max_bytes + protocol.ENVELOPE_SLACK)
                if frame is None:
                    return  # the client went away
                fields = protocol.read_fields(frame, self.max_bytes + protocol.ENVELOPE_SLACK)
                op = fields["op"]
                if op == "task":
                    self._start(fields, frame)
                elif op == "cancel":
                    self._cancel(fields.get("task"))
                elif op == "ping":
                    seq = fields.get("seq")
                    self._send(protocol.message("pong", seq=seq if type(seq) is int else None))
                elif op == "bye":
                    return
                else:
                    raise ProtocolError(f"unknown op {op!r}")
        except ProtocolError as e:
            self._send(protocol.message("error", message=str(e)[: protocol.MAX_ERROR_CHARS]))
        except OSError:
            pass
        finally:
            self.shutdown()

    def _handshake(self) -> bool:
        frame = protocol.recv_msg(self.sock, protocol.MAX_CONTROL_BYTES)
        if frame is None:
            return False
        protocol.check_hello(protocol.read_fields(frame, protocol.MAX_CONTROL_BYTES))
        return self._send(protocol.message("hello", **protocol.hello_fields(pid=os.getpid())))

    def _start(self, fields: dict[str, Any], frame: bytearray) -> None:
        from ..run_worker import ENTRY_POINTS

        task_id = fields.get("task")
        if type(task_id) is not int:
            raise ProtocolError("task without an integer id")
        with self._lock:
            if task_id in self._tasks:
                raise ProtocolError(f"task {task_id} is already running")
        try:
            entry, env, cwd, wall, result_bytes, limits = _task_fields(fields, ENTRY_POINTS)
        except ValueError as e:
            self._send_result(task_id, False, f"task rejected: {e}")
            return
        cancel = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(task_id, Task(entry, wire.EncodedArgs(bytes(frame)), env), cancel, cwd, wall, result_bytes, limits),
            name=f"task-{task_id}",
            daemon=True,
        )
        with self._lock:
            self._tasks[task_id] = (cancel, thread)
        thread.start()

    def _run(
        self,
        task_id: int,
        task: Task,
        cancel: threading.Event,
        cwd: str | None,
        wall: float | None,
        result_bytes: int | None,
        limits: TaskLimits | None = None,
    ) -> None:
        try:
            handle = self.executor.submit(task)
            try:
                (result,) = self.executor.wait(
                    [handle], cancel, 1, max_result_bytes=result_bytes, task_timeout=wall, cwd=cwd, limits=limits
                )
            except TaskCancelled:
                self._send_result(task_id, False, "cancelled", cancelled=True)
                return
            except Exception as e:  # noqa: BLE001 -- e.g. a worker that couldn't be started
                self._send_result(task_id, False, f"the runner couldn't run the task: {type(e).__name__}: {e}", crashed=True)
                return
            fields = {"op": "result", "task": task_id, "ok": result.ok, "error": result.error, "crashed": result.crashed, "cancelled": False}
            if result.ok:
                try:
                    out = wire.reframe(result.value, fields, max_bytes=result_bytes or self.max_bytes)
                except wire.WireError as e:
                    self._send_result(task_id, False, f"result rejected: {e}")
                    return
                self._send(out)
            else:
                self._send_result(task_id, False, result.error, crashed=result.crashed)
        finally:
            with self._lock:
                self._tasks.pop(task_id, None)

    def _cancel(self, task_id: Any) -> None:
        if type(task_id) is not int:
            raise ProtocolError("cancel without an integer id")
        with self._lock:
            entry = self._tasks.get(task_id)
        if entry is not None:
            entry[0].set()
        # An unknown id is a task that has just finished: its result is
        # already on its way, which the client takes as the answer.

    def shutdown(self) -> None:
        """Stop every running task process, discard the spares, close."""
        with self._lock:
            tasks = list(self._tasks.values())
        for cancel, _ in tasks:
            cancel.set()
        for _, thread in tasks:
            thread.join(timeout=SHUTDOWN_GRACE)
        self.executor.close()
        with contextlib.suppress(OSError):
            self.sock.close()


def _task_fields(
    fields: dict[str, Any], entry_points: frozenset[str]
) -> tuple[str, dict[str, str] | None, str | None, float | None, int | None, TaskLimits]:
    entry = fields.get("entry")
    if entry not in entry_points:
        raise ValueError(f"unknown entry point {entry!r}")
    env = fields.get("env")
    if env is not None:
        if not isinstance(env, dict) or len(env) > protocol.MAX_ENV_VARS:
            raise ValueError("bad env")
        size = 0
        for k, v in env.items():
            if not isinstance(v, str) or not k or "=" in k or "\0" in k or "\0" in v:
                raise ValueError(f"bad environment variable {k!r}")
            size += len(k) + len(v)
        if size > protocol.MAX_ENV_CHARS:
            raise ValueError("environment too large")
    cwd = fields.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or "\0" in cwd):
        raise ValueError("bad cwd")
    limits = fields.get("limits") or {}
    if not isinstance(limits, dict):
        raise ValueError("bad limits")
    wall = limits.get("wall_seconds")
    if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0):
        raise ValueError("bad wall_seconds")
    result_bytes = limits.get("result_bytes")
    if result_bytes is not None and (type(result_bytes) is not int or result_bytes <= 0):
        raise ValueError("bad result_bytes")
    task_limits = TaskLimits(
        memory_mb=check_limit(limits.get("memory_mb"), "memory_mb"), threads=check_limit(limits.get("threads"), "threads")
    )
    return entry, env, cwd, (float(wall) if wall is not None else None), result_bytes, task_limits


# ---- the process ---------------------------------------------------------


def make_executor(warm_workers: int | None = None, limits: TaskLimits | None = None) -> LocalExecutor:
    """The agent's own executor: this process's environment and cwd are the
    base of every task (no WorkerLaunch: the agent is already what a
    deployment started as the right user, in the right place). `limits`:
    every task's, whatever its client asks (a client can only lower them)."""
    from ..worker_pool import WorkerPool, pool_size_from_env

    size = pool_size_from_env() if warm_workers is None else warm_workers
    return LocalExecutor(WorkerPool(multiprocessing.get_context("spawn"), size), raw_results=True, limits=limits)


def _accept_one(path: str, mode: int = 0o600) -> socket.socket:
    """Accept exactly one connection on a new socket at `path`, created with
    `mode` (connecting needs write permission on it: the default lets only
    its owner in; e.g. 0o660 lets its group in too, for a socket in a
    setgid directory shared with the client)."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Created with no permissions beyond `mode` from the start (umask),
        # then set to exactly `mode` before anyone can connect.
        old = os.umask(0o777 & ~mode)
        try:
            listener.bind(path)
        finally:
            os.umask(old)
        os.chmod(path, mode)
        listener.listen(1)
        conn, _ = listener.accept()
    finally:
        listener.close()
        with contextlib.suppress(OSError):
            os.unlink(path)
    return conn


def _socket_mode(text: str) -> int:
    try:
        mode = int(text, 8)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an octal mode: {text!r}") from None
    if mode & ~0o666:
        # Execute and the special bits mean nothing on a socket.
        raise argparse.ArgumentTypeError(f"a socket mode has only read and write bits: {text!r}")
    if not mode & 0o200:
        raise argparse.ArgumentTypeError(f"the owner must be able to connect: {text!r}")
    return mode


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text!r}")
    return value


def _on_signal(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelmaker-runner", description="Serve one runner-protocol client.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--fd", type=int, help="an inherited file descriptor of a connected stream socket")
    where.add_argument("--listen", metavar="PATH", help="a Unix socket path to accept exactly one connection on")
    parser.add_argument("--warm-workers", type=int, default=None, help="spare task processes (default: MODELMAKER_WARM_WORKERS or 2)")
    parser.add_argument("--cwd", default=None, help="the directory to run in (entered once started)")
    parser.add_argument(
        "--socket-mode",
        type=_socket_mode,
        default=0o600,
        help="with --listen: the socket's permissions, in octal (default 0600: only this user may connect)",
    )
    parser.add_argument(
        "--task-memory-mb",
        type=_positive_int,
        default=None,
        help="each task's memory limit: private resident memory of its processes, in MB (default: none)",
    )
    parser.add_argument(
        "--task-threads",
        type=_positive_int,
        default=None,
        help="each task's limit on threads and processes, together (default: none)",
    )
    parser.add_argument(
        "--total-threads",
        type=_positive_int,
        default=None,
        help="RLIMIT_NPROC for every task process: threads and processes of this user, all together (default: none)",
    )
    args = parser.parse_args(argv)
    if args.socket_mode != 0o600 and args.listen is None:
        parser.error("--socket-mode needs --listen")
    if args.cwd is not None:
        os.chdir(args.cwd)

    if args.fd is not None:
        os.set_inheritable(args.fd, False)  # task processes never hold the client's socket
        sock = socket.socket(fileno=args.fd)
    else:
        sock = _accept_one(args.listen, args.socket_mode)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_signal)
    limits = TaskLimits(memory_mb=args.task_memory_mb, threads=args.task_threads, total_threads=args.total_threads)
    agent = Agent(sock, make_executor(args.warm_workers, limits))
    try:
        agent.serve()
    except SystemExit:
        agent.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
