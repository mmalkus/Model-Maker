"""Subprocess entry points for executing block calls in isolation (see
Runner._dispatch and Runner._dispatch_fused_group in runner.py). Kept in
its own module, importing as little as possible at module level, because
these are the functions every worker process's
`multiprocessing.Process(target=...)` points at: each must be picklable by
reference (importable by dotted path) so it works whether the process was
created via "fork" or "spawn" (see runner.MP_CONTEXT).

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


def run_fused_group_entry(
    steps: list[dict[str, Any]],
    result_queue: Any,
    task_key: Any,
) -> None:
    """Entry point for a *streaming run*'s fused chain (see
    Runner._build_fusion_groups/_dispatch_fused_group in runner.py) --
    runs an ordered run of compatible registry blocks (filter/select/
    groupby_agg/join and read_csv today; see BlockSpec.lazy_fn) as one
    unbroken polars lazy plan in this single subprocess, and only collects
    once, at the very end, with the streaming engine -- the one place in
    this app where a source larger than memory doesn't fully materialize
    at every block boundary.

    Each `steps[i]` is `{"category", "kwargs", "chain_port"}`: `kwargs` is
    every param this step's lazy_fn needs *except* the chained one --
    including any already-materialized pl.DataFrame values read from cache
    for a non-chained dataframe input (the caller's job, see
    Runner._real_df_and_meta) -- lazily wrapped here so it joins the same
    unbroken plan rather than anchoring a fresh eager sub-result partway
    through. `chain_port`, when not None, names the one kwarg replaced with
    the previous step's still-uncollected LazyFrame output.

    Reports `(task_key, True, (per_step_schema, collected_df))` on success
    -- `per_step_schema` is a plain `[{col: dtype_str}, ...]`, one per step,
    from `LazyFrame.collect_schema()` (free -- no data touched) so the
    parent can chain each block's own metadata_transform through the fused
    stretch without having materialized any of it -- or
    `(task_key, False, "step <i> (<category>): <error>")` naming which
    block in the chain actually failed. Never raises into the parent, same
    contract as run_worker_entry above."""
    import polars as pl

    import modelmaker.blocks.library  # noqa: F401 -- see run_worker_entry's comment on this import
    from .blocks.base import BLOCK_REGISTRY

    idx = -1
    schemas: list[dict[str, str]] = []
    try:
        chain_value: pl.LazyFrame | None = None
        for idx, step in enumerate(steps):
            spec = BLOCK_REGISTRY[step["category"]]
            kwargs = {k: (v.lazy() if isinstance(v, pl.DataFrame) else v) for k, v in step["kwargs"].items()}
            chain_port = step.get("chain_port")
            if chain_port is not None:
                kwargs[chain_port] = chain_value
            chain_value = spec.lazy_fn(**kwargs)
            schemas.append({name: str(dtype) for name, dtype in chain_value.collect_schema().items()})
        final = chain_value.collect(engine="streaming")
        result_queue.put((task_key, True, (schemas, final)))
    except BaseException as e:  # noqa: BLE001 -- must reach the queue, not crash the worker silently
        category = steps[idx]["category"] if 0 <= idx < len(steps) else "?"
        result_queue.put((task_key, False, f"step {idx} ({category}): {type(e).__name__}: {e}"))
