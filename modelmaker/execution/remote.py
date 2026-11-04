"""RemoteExecutor: tasks run by a runner agent at the other end of a socket.

The same contract as LocalExecutor (base.py) -- `submit` only queues,
`wait` runs a batch `max_concurrent` at a time and returns results in
handle order, raises TaskCancelled once every task of the batch has been
stopped, and reports a task (or agent) that died as `crashed=True` -- but
the task processes belong to a runner agent (agent.py) that this process
talks to over one stream socket, in the runner protocol (protocol.py).
Where the agent runs is the caller's business: a local child process
(`spawn_agent`, the default), the same as another user (`spawn_agent`
with a command prefix), or a sandbox that only shares a socket with this
process (`connect_unix`, or any `connect` callable returning an AgentLink).

What this process trusts the agent with: nothing. Everything it sends
back is decoded here by wire.py's validating decoder -- the same one
LocalExecutor uses on its task processes -- and its envelope fields are
checked; a frame that's malformed, oversized, or breaks the protocol ends
the connection.

When the connection ends -- the agent died, was killed, or broke the
protocol -- every task in flight on it comes back crashed, and the agent
is killed (AgentLink.kill) to be sure nothing of it is left running. The
next `wait` connects again through `connect`: a crashed or destroyed agent
is replaced on the next run, like a crashed task process is.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import protocol, wire
from .base import Task, TaskCancelled, TaskHandle, TaskResult
from .env import _secret_test, scrubbed_environ, task_environ
from .protocol import ProtocolError

POLL_INTERVAL = 0.05
# A task the agent hasn't confirmed stopped this long after a cancel (its
# own Stop takes up to ~10 s for a process that ignores SIGTERM) gets the
# whole agent killed instead: a runner that won't stop a task is destroyed,
# its task processes with it.
CANCEL_GRACE = 15.0
# Heartbeat: while a wait is in progress, an agent silent this long is
# pinged; one silent for HEARTBEAT_TIMEOUT more is taken for dead.
HEARTBEAT_INTERVAL = 10.0
HEARTBEAT_TIMEOUT = 30.0
# How long connecting (spawn, handshake) may take.
CONNECT_TIMEOUT = 60.0
# How long close() waits for the agent to stop its tasks and exit.
CLOSE_GRACE = 15.0


# ---- the far side: an agent process, or just a socket ---------------------


class AgentLink:
    """A connected socket to a runner agent, and how to make sure it's gone.
    `process` is the agent's process when this side started it (killed as
    a whole process group: the agent and every task process it started);
    None for an agent someone else runs, where `on_kill` (if any) is how to
    have it destroyed."""

    def __init__(
        self, sock: socket.socket, process: subprocess.Popen | None = None, on_kill: Callable[[], None] | None = None
    ) -> None:
        self.sock = sock
        self.process = process
        self.on_kill = on_kill

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def wait_exit(self, timeout: float) -> bool:
        if self.process is None:
            return True
        try:
            self.process.wait(timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()
        if self.process is not None:
            # The agent is its own session leader (spawn_agent), so its
            # process group holds its task processes too.
            with contextlib.suppress(OSError):
                os.killpg(self.process.pid, signal.SIGKILL)
            with contextlib.suppress(OSError):
                self.process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(10)
        if self.on_kill is not None:
            self.on_kill()


# The agent's `-c` program: sys.path from argv (so modelmaker imports as
# in this process), then the agent's own main. Under -P, so nothing in the
# cwd shadows the stdlib.
_BOOTSTRAP = "import json, sys\nsys.path[:] = json.loads(sys.argv[1])\nfrom modelmaker.execution.agent import main\nsys.exit(main(sys.argv[2:]))\n"


def agent_command(
    fd: int | None = None,
    *,
    listen: str | None = None,
    warm_workers: int | None = None,
    cwd: str | None = None,
    python: str | None = None,
    socket_mode: int | None = None,
) -> list[str]:
    """argv that runs a runner agent serving the socket on `fd`, or the
    one connection it accepts on `listen` (created with `socket_mode`,
    default 0o600)."""
    flags = list(getattr(subprocess, "_args_from_interpreter_flags", list)())
    if "-P" not in flags:
        flags.append("-P")
    where = ["--fd", str(fd)] if listen is None else ["--listen", listen]
    cmd = [python or sys.executable, *flags, "-c", _BOOTSTRAP, json.dumps(sys.path), *where]
    if warm_workers is not None:
        cmd += ["--warm-workers", str(warm_workers)]
    if cwd is not None:
        cmd += ["--cwd", cwd]
    if socket_mode is not None:
        cmd += ["--socket-mode", f"{socket_mode:04o}"]
    return cmd


def spawn_agent(
    *,
    command_prefix: tuple[str, ...] = (),
    environ: Mapping[str, str] | None = None,
    cwd: str | None = None,
    warm_workers: int | None = None,
    python: str | None = None,
) -> AgentLink:
    """Start a runner agent as a child process over a socketpair -- no
    socket in the filesystem for anything else to connect to. The agent
    runs `command_prefix` + the interpreter (e.g. setpriv, to run it as
    another uid), with `environ` (default: this process's) scrubbed of
    secrets, in its own session (so killing it kills its task processes
    too), and enters `cwd` itself once started (after the prefix took
    effect)."""
    parent, child = socket.socketpair()
    try:
        process = subprocess.Popen(
            [*command_prefix, *agent_command(child.fileno(), warm_workers=warm_workers, cwd=cwd, python=python)],
            env=scrubbed_environ(environ),
            pass_fds=(child.fileno(),),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except BaseException:
        parent.close()
        raise
    finally:
        child.close()  # or EOF never comes when the agent dies
    return AgentLink(parent, process)


def connect_unix(path: str, on_kill: Callable[[], None] | None = None) -> AgentLink:
    """Connect to an agent listening on a Unix socket (`--listen PATH`)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except BaseException:
        sock.close()
        raise
    return AgentLink(sock, None, on_kill)


# ---- one connection ----------------------------------------------------------


class _Connection:
    """A handshaken connection and its reader thread, which routes each
    result to the queue of the wait that sent its task."""

    def __init__(self, link: AgentLink, max_bytes: Callable[[], int], connect_timeout: float) -> None:
        self.link = link
        self.sock = link.sock
        self._max_bytes = max_bytes
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._waiters: dict[int, queue.Queue] = {}
        self._ids = itertools.count(1)
        self._issued = 0
        self._seq = itertools.count(1)
        self.dead: str | None = None
        self.last_heard = time.monotonic()
        self._pinged_at: float | None = None
        self.sock.settimeout(connect_timeout)
        try:
            protocol.send_msg(self.sock, protocol.message("hello", **protocol.hello_fields()))
            frame = protocol.recv_msg(self.sock, protocol.MAX_CONTROL_BYTES)
            if frame is None:
                raise ProtocolError("the runner agent closed the connection during the handshake")
            fields = protocol.read_fields(frame, protocol.MAX_CONTROL_BYTES)
            if fields["op"] == "error":
                raise ProtocolError(f"the runner agent refused the connection: {str(fields.get('message'))[:500]}")
            protocol.check_hello(fields)
        except socket.timeout:
            link.kill()
            raise ProtocolError("the runner agent didn't answer the handshake") from None
        except BaseException:
            link.kill()
            raise
        self.agent_pid = fields.get("pid") if type(fields.get("pid")) is int else None
        self.sock.settimeout(None)
        self._reader = threading.Thread(target=self._read_loop, name="runner-agent-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            while True:
                cap = self._max_bytes() + protocol.ENVELOPE_SLACK
                frame = protocol.recv_msg(self.sock, cap)
                if frame is None:
                    raise ProtocolError("the runner agent closed the connection")
                fields = protocol.read_fields(frame, cap)
                self.last_heard = time.monotonic()
                self._pinged_at = None
                op = fields["op"]
                if op == "result":
                    tid = fields.get("task")
                    with self._lock:
                        q = self._waiters.get(tid) if type(tid) is int else None
                        issued = self._issued
                    if q is not None:
                        q.put(("result", fields, frame))
                    elif not (type(tid) is int and 0 < tid <= issued):
                        raise ProtocolError(f"the runner agent sent a result for an unknown task {tid!r}")
                    # else: a task whose wait gave up on it (cancelled, interrupted)
                elif op == "pong":
                    pass
                elif op == "error":
                    raise ProtocolError(f"the runner agent reported: {str(fields.get('message'))[:500]}")
                else:
                    raise ProtocolError(f"the runner agent sent an unknown message {op!r}")
        except (ProtocolError, OSError) as e:
            self.fail(str(e) or type(e).__name__)

    def register(self, q: queue.Queue) -> int:
        with self._lock:
            tid = self._issued = next(self._ids)
            self._waiters[tid] = q
            dead = self.dead
        if dead is not None:
            q.put(("dead", dead, None))
        return tid

    def unregister(self, tid: int) -> None:
        with self._lock:
            self._waiters.pop(tid, None)

    def send(self, frame: bytes) -> None:
        with self._send_lock:
            try:
                protocol.send_msg(self.sock, frame)
            except OSError as e:
                send_error = e
            else:
                return
        self.fail(f"the connection to the runner agent broke ({send_error})")

    def heartbeat(self, interval: float, timeout: float) -> None:
        """Called while a wait is in progress: ping an agent silent for
        `interval`; one that leaves a ping unanswered for `timeout` is
        dead. (Silence between waits doesn't count: only a ping does.)"""
        now = time.monotonic()
        pinged = self._pinged_at
        if pinged is not None:
            if now - pinged > timeout:
                self.fail(f"the runner agent stopped answering (no reply to a ping in {now - pinged:.0f} s)")
        elif now - self.last_heard > interval:
            self._pinged_at = now
            self.send(protocol.message("ping", seq=next(self._seq)))

    def fail(self, reason: str) -> None:
        """End the connection: every waiting task hears it, the agent is
        killed. Idempotent."""
        with self._lock:
            if self.dead is not None:
                return
            self.dead = reason
            waiters = list(self._waiters.values())
        for q in waiters:
            q.put(("dead", reason, None))
        self.link.kill()

    def close(self, grace: float) -> None:
        """Orderly: say bye, let the agent stop its tasks and exit, then
        make sure (kill its process group) -- the agent's exit also ends
        any task process that outlived it."""
        if self.dead is None:
            with self._send_lock, contextlib.suppress(OSError):
                protocol.send_msg(self.sock, protocol.message("bye"))
            with contextlib.suppress(OSError):
                self.sock.shutdown(socket.SHUT_WR)
            self.link.wait_exit(grace)
        self.fail("the executor was closed")


# ---- the executor --------------------------------------------------------------


class RemoteExecutor:
    """Runs each Task through a runner agent (see the module docstring).
    Thread-safe: several waits can share one connection.

    - `connect`: returns a connected AgentLink, called on first use and
      again after a connection ended (default: a local agent, spawn_agent()).
    - `max_result_bytes`, `task_timeout`: as for LocalExecutor; both travel
      with each task (the agent enforces them) and the cap is enforced here
      again on receipt. Read at each use, so they can be changed.
    - `base_environ`: the environment the agent's tasks start from, if
      known (see LocalExecutor.base_environ); also supplies the secret
      denylist applied to each task's variables before they're sent.
    - `local_context`: send this process's current environment (scrubbed)
      and cwd with every task, the way LocalExecutor's tasks get them --
      for an agent on this machine standing in for a LocalExecutor. Off for
      a sandbox, which has its own."""

    def __init__(
        self,
        connect: Callable[[], AgentLink] | None = None,
        *,
        max_result_bytes: int | None = None,
        task_timeout: float | None = None,
        base_environ: Mapping[str, str] | None = None,
        local_context: bool = False,
        connect_timeout: float = CONNECT_TIMEOUT,
        cancel_grace: float = CANCEL_GRACE,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
    ) -> None:
        self._connect = connect or spawn_agent
        self.max_result_bytes = max_result_bytes
        self.task_timeout = task_timeout or None
        self.base_environ = dict(base_environ) if base_environ is not None else None
        self.local_context = local_context
        self.connect_timeout = connect_timeout
        self.cancel_grace = cancel_grace
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._conn: _Connection | None = None

    # ---- connection

    def _limit(self) -> int:
        return self.max_result_bytes if self.max_result_bytes is not None else wire.max_bytes_from_env()

    def connection(self) -> _Connection:
        """The live connection, connecting (again) if there is none."""
        with self._lock:
            conn = self._conn
            if conn is not None and conn.dead is None:
                return conn
            self._conn = None
            conn = _Connection(self._connect(), self._limit, self.connect_timeout)
            self._conn = conn
            return conn

    @property
    def agent_pid(self) -> int | None:
        """The current agent's pid as it reported it (None: not connected)."""
        conn = self._conn
        return conn.agent_pid if conn is not None and conn.dead is None else None

    @property
    def link(self) -> AgentLink | None:
        conn = self._conn
        return conn.link if conn is not None and conn.dead is None else None

    def prewarm(self) -> None:
        """Connect now (starting the agent) rather than at the first wait."""
        self.connection()

    def ping(self, timeout: float = 10.0) -> bool:
        """Whether the agent answers within `timeout` (connecting if needed)."""
        conn = self.connection()
        before = conn.last_heard
        conn.send(protocol.message("ping", seq=0))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and conn.dead is None:
            if conn.last_heard > before:
                return True
            time.sleep(0.01)
        return False

    # ---- the Executor interface

    def submit(self, task: Task) -> TaskHandle:
        with self._lock:
            return TaskHandle(next(self._ids), task)

    def wait(
        self, handles: list[TaskHandle], cancel: threading.Event, max_concurrent: int | None = None
    ) -> list[TaskResult]:
        if not handles:
            return []
        limit = len(handles) if max_concurrent is None else max(1, max_concurrent)
        max_bytes = self._limit()
        results: dict[TaskHandle, TaskResult] = {}
        try:
            conn = self.connection()
        except Exception as e:  # noqa: BLE001 -- no agent: every task is a crashed one, not a dead dispatch
            error = f"the runner agent couldn't be reached: {type(e).__name__}: {e}"
            return [TaskResult(h, ok=False, error=error, crashed=True) for h in handles]
        q: queue.Queue = queue.Queue()
        pending = list(handles)
        running: dict[int, TaskHandle] = {}

        def _start_next() -> None:
            while pending and len(running) < limit:
                handle = pending.pop(0)
                try:
                    tid, frame = self._encode(conn, q, handle.task, max_bytes)
                except (wire.WireError, RecursionError) as e:
                    msg = str(e) if isinstance(e, wire.WireError) else "nested too deeply"
                    results[handle] = TaskResult(handle, ok=False, error=f"the task can't be sent to the runner: {msg}")
                    continue
                running[tid] = handle
                conn.send(frame)

        try:
            _start_next()
            while running:
                if cancel.is_set():
                    self._cancel_all(conn, q, running)
                    raise TaskCancelled("cancelled")
                try:
                    kind, a, b = q.get(timeout=POLL_INTERVAL)
                except queue.Empty:
                    conn.heartbeat(self.heartbeat_interval, self.heartbeat_timeout)
                    continue
                if kind == "dead":
                    error = f"the runner agent went away ({a}) -- likely out of memory or a crash"
                    for handle in [*running.values(), *pending]:
                        results[handle] = TaskResult(handle, ok=False, error=error, crashed=True)
                    running.clear()
                    pending.clear()
                    break
                handle = running.pop(a["task"], None)
                if handle is None:
                    continue  # a straggler for a task this wait gave up on
                conn.unregister(a["task"])
                results[handle] = self._result(handle, a, b, max_bytes)
                _start_next()
        finally:
            for tid in running:
                conn.unregister(tid)
        return [results[h] for h in handles]

    def _encode(self, conn: _Connection, q: queue.Queue, task: Task, max_bytes: int) -> tuple[int, bytes]:
        if self.local_context:
            env: dict[str, str] | None = task_environ(task.env, None)
            cwd: str | None = os.getcwd()
        else:
            secret = _secret_test(self.base_environ if self.base_environ is not None else os.environ)
            env = {k: v for k, v in task.env.items() if not secret(k)} if task.env else None
            cwd = None
        tid = conn.register(q)
        fields = {
            "op": "task",
            "task": tid,
            "entry": task.entry,
            "env": env,
            "cwd": cwd,
            "limits": {"wall_seconds": self.task_timeout or None, "result_bytes": max_bytes},
        }
        try:
            return tid, wire.encode_frame(fields, tuple(task.args), max_bytes=wire.DEFAULT_MAX_BYTES)
        except BaseException:
            conn.unregister(tid)
            raise

    def _cancel_all(self, conn: _Connection, q: queue.Queue, running: dict[int, TaskHandle]) -> None:
        """Ask the agent to stop every task in `running` and wait until it
        says each is stopped; an agent that doesn't in time is killed."""
        for tid in list(running):
            conn.send(protocol.message("cancel", task=tid))
        deadline = time.monotonic() + self.cancel_grace
        while running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                conn.fail(f"the runner agent didn't stop a cancelled task within {self.cancel_grace:g} s")
                break
            try:
                kind, a, _ = q.get(timeout=min(remaining, POLL_INTERVAL * 4))
            except queue.Empty:
                continue
            if kind == "dead":
                break  # the agent is gone, and its tasks with it
            if running.pop(a["task"], None) is not None:
                conn.unregister(a["task"])
        running.clear()

    def _result(self, handle: TaskHandle, fields: dict[str, Any], frame: bytearray, max_bytes: int) -> TaskResult:
        if fields.get("ok") is True:
            try:
                _, value = wire.decode_frame(frame, max_bytes=max_bytes + protocol.ENVELOPE_SLACK)
            except wire.WireError as e:
                return TaskResult(handle, ok=False, error=f"result rejected: {e}")
            except Exception as e:  # noqa: BLE001 -- a decoder bug must still be a red block, not a dead dispatch
                return TaskResult(handle, ok=False, error=f"result rejected: {type(e).__name__}: {e}")
            return TaskResult(handle, ok=True, value=value)
        if fields.get("cancelled") is True:
            # Not asked for here (a cancel this side sent is answered in
            # _cancel_all): the agent stopped it on its own, shutting down.
            return TaskResult(handle, ok=False, error="the runner agent stopped the task (it was shutting down)", crashed=True)
        error = fields.get("error")
        error = error if isinstance(error, str) and error else "the block failed without an error message"
        if len(error) > protocol.MAX_ERROR_CHARS:
            error = error[: protocol.MAX_ERROR_CHARS] + "…"
        return TaskResult(handle, ok=False, error=error, crashed=fields.get("crashed") is True)

    def close(self) -> None:
        """End the connection: the agent stops its task processes and exits
        (and is killed if it doesn't in time). A later wait connects anew."""
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            conn.close(CLOSE_GRACE)
