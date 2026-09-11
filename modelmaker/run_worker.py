"""Subprocess entry point for executing a single block call in isolation
(see Runner._dispatch in runner.py). Kept in its own module, importing as
little as possible at module level, because this is the function every
worker process's `multiprocessing.Process(target=...)` points at: it must
be picklable by reference (importable by dotted path) so it works whether
the process was created via "fork" or "spawn" (see runner.MP_CONTEXT).

Running every block call -- grouped or not -- in its own process is what
makes a runaway or misconfigured block (e.g. group_by on a near-unique
column producing far more work than expected) a crashed *worker*, caught
and reported as a normal red-block error, instead of a crashed server: a
segfault, an OS OOM-kill, or a runaway allocation in the child never
touches this process or any other block's run.
"""

from __future__ import annotations

from typing import Any


def run_worker_entry(
    category: str,
    is_custom: bool,
    code: str | None,
    kwargs: dict[str, Any],
    result_queue: Any,
    task_key: Any,
) -> None:
    """Resolve and call one block's function, reporting success or failure
    back over `result_queue` as `(task_key, ok, payload)`. Never raises into
    the parent process -- multiprocessing.Process has no return channel of
    its own, so any escaping exception here would simply vanish and leave
    the parent's `_dispatch` waiting until it notices the process died."""
    try:
        if is_custom:
            from .graph import _compile_code_to_fn

            fn = _compile_code_to_fn(code or "")
        else:
            # A freshly spawned interpreter has none of the registry blocks
            # registered yet -- modelmaker/__init__.py only imports
            # library.py itself, so modelling.py/stat_tests.py (and any
            # other block module the running app has loaded) must be
            # imported explicitly here too, mirroring what api.py does at
            # startup, or BLOCK_REGISTRY simply won't have `category` yet.
            import modelmaker.blocks.library  # noqa: F401
            import modelmaker.blocks.modelling  # noqa: F401
            import modelmaker.blocks.stat_tests  # noqa: F401
            from .blocks.base import BLOCK_REGISTRY

            fn = BLOCK_REGISTRY[category].fn
        result = fn(**kwargs)
        result_queue.put((task_key, True, result))
    except BaseException as e:  # noqa: BLE001 -- must reach the queue, not crash the worker silently
        result_queue.put((task_key, False, f"{type(e).__name__}: {e}"))
