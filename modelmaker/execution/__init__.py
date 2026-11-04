"""The execution seam: how the Runner gets block code run somewhere else.

- base.py: the `Executor` protocol and its `Task` / `TaskHandle` /
  `TaskResult` types -- the only thing Runner._run_tasks talks to.
- local.py: `LocalExecutor`, single-use local processes (the default).
- remote.py: `RemoteExecutor`, the same contract through a runner agent
  at the other end of a socket (agent.py), speaking protocol.py.
- agent.py: the runner agent (`modelmaker-runner`): serves one client,
  runs each task in a single-use process of its own, holds no secrets.
- protocol.py: the runner protocol -- framing, messages, the handshake.
- wire.py: the data-only format tasks and results travel in (never pickle).
- safe_pickle.py: the interim allowlisting unpickler for the few non-data
  results wire.py still admits (to be replaced by sandbox handles).
- env.py: which environment variables a task process never sees.

The point of the agent: whatever runs it -- another uid, a container with
no network -- the process holding the secrets only ever exchanges data
with it over one socket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Executor, Task, TaskCancelled, TaskHandle, TaskResult

if TYPE_CHECKING:
    from .local import LocalExecutor
    from .remote import RemoteExecutor

__all__ = ["Executor", "LocalExecutor", "RemoteExecutor", "Task", "TaskCancelled", "TaskHandle", "TaskResult"]


def __getattr__(name: str) -> Any:
    # Lazy: local.py imports worker_pool, which imports env.py from this
    # package -- an eager import here would make that a cycle.
    if name == "LocalExecutor":
        from .local import LocalExecutor

        return LocalExecutor
    if name == "RemoteExecutor":
        from .remote import RemoteExecutor

        return RemoteExecutor
    raise AttributeError(name)
