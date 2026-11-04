"""The execution seam: how the Runner gets block code run somewhere else.

- base.py: the `Executor` protocol and its `Task` / `TaskHandle` /
  `TaskResult` types -- the only thing Runner._run_tasks talks to.
- local.py: `LocalExecutor`, single-use local processes (the default).
- wire.py: the data-only format results come back in (never pickle).
- safe_pickle.py: the interim allowlisting unpickler for the few non-data
  results wire.py still admits (to be replaced by sandbox handles).
- env.py: which environment variables a task process never sees.

Where this is going: a `RemoteExecutor` speaking the same contract to a
sandboxed runner that holds no secrets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Executor, Task, TaskCancelled, TaskHandle, TaskResult

if TYPE_CHECKING:
    from .local import LocalExecutor

__all__ = ["Executor", "LocalExecutor", "Task", "TaskCancelled", "TaskHandle", "TaskResult"]


def __getattr__(name: str) -> Any:
    # Lazy: local.py imports worker_pool, which imports env.py from this
    # package -- an eager import here would make that a cycle.
    if name == "LocalExecutor":
        from .local import LocalExecutor

        return LocalExecutor
    raise AttributeError(name)
