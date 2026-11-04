"""The contract between the Runner and whatever actually runs block code.

`Runner._run_tasks` is the only caller: it turns a block run (one call, a
group_by fan-out, a fused streaming group, a stochastic region's
iterations) into `Task`s, hands them to an `Executor`, and gets plain
values back. Everything about *where* a task runs -- a local single-use
process today (`LocalExecutor`), a session sandbox later (a
`RemoteExecutor`) -- lives behind
this interface, so the run plan, staleness, fusion and fan-out logic above
it never has to know.

The shapes follow S6.1 of the architecture doc, adapted to what the Runner
actually needs:

- `wait` takes the concurrency cap (`max_concurrent`). A group_by fan-out
  can be hundreds of tasks run a few at a time, so "how many at once" is a
  property of one batch, not of the executor or of a single task; with
  `submit` only queuing, the executor (not the caller) decides when each
  queued task starts.
- `wait` raises `TaskCancelled` when the caller's cancel event is set,
  after stopping every task of the batch, rather than returning partial
  results: a cancelled block run is discarded as a whole.
- A task process that dies without reporting is a normal failed
  `TaskResult` (`crashed=True`), never an exception and never a hang.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Task:
    """One unit of block work: the name of an entry point in
    modelmaker/run_worker.py and the positional args it is called with
    (everything but `result_queue`/`task_key`, which the executor supplies).

    The args travel *to* the task process, which is the trusted direction:
    they come from this process (the graph, params, upstream outputs), so a
    local executor may pickle them. Only what comes back is untrusted -- see
    wire.py."""

    entry: str
    args: tuple[Any, ...]


@dataclass(frozen=True, eq=False)
class TaskHandle:
    """What `submit` returns: identifies one submitted task to `wait`.
    Compared by identity -- two submissions of an equal Task are two
    tasks."""

    id: int
    task: Task


@dataclass(frozen=True)
class TaskResult:
    """The outcome of one task. `value` is the entry point's result decoded
    from the wire format (only meaningful when `ok`); `error` is a one-line
    message shown on the block (`"ValueError: ..."`, or a crash/rejection
    description). `crashed` marks a process that died without reporting --
    a segfault, an OS OOM-kill -- as opposed to block code that raised."""

    handle: TaskHandle
    ok: bool
    value: Any = None
    error: str | None = None
    crashed: bool = False


class TaskCancelled(RuntimeError):
    """Raised by `Executor.wait` once the cancel event is set and every task
    of the batch has been stopped."""


@runtime_checkable
class Executor(Protocol):
    def submit(self, task: Task) -> TaskHandle:
        """Queue a task. It need not start until `wait` is called on it."""
        ...

    def wait(
        self, handles: list[TaskHandle], cancel: threading.Event, max_concurrent: int | None = None
    ) -> list[TaskResult]:
        """Run `handles` to completion, at most `max_concurrent` at a time
        (None: no cap), and return one result per handle, in order. Blocks
        the calling thread. Raises TaskCancelled -- having stopped every
        task in `handles` -- as soon as `cancel` is set."""
        ...

    def close(self) -> None:
        """Release whatever the executor holds (warm processes, sockets)."""
        ...
