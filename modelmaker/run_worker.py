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
            import modelmaker.blocks.feature_analysis  # noqa: F401
            import modelmaker.blocks.library  # noqa: F401
            import modelmaker.blocks.modelling  # noqa: F401
            import modelmaker.blocks.stat_tests  # noqa: F401
            import modelmaker.blocks.stochastic  # noqa: F401
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
    """Entry point for a *streaming run*'s fused group (see
    Runner._build_fusion_groups/_dispatch_fused_group in runner.py) --
    runs a whole connected DAG of compatible registry blocks (filter/
    select/groupby_agg/join and read_csv today; see BlockSpec.lazy_fn) as
    one unbroken polars lazy plan in this single subprocess, and only
    collects at the very end, with the streaming engine -- the one place
    in this app where a source larger than memory doesn't fully
    materialize at every block boundary. A group is no longer required to
    be a straight-line chain: fan-out (one block feeding several others in
    the same group) and fan-in (e.g. a join whose both sides are still
    lazy) are both single lazy plans here, built up in `steps` order (a
    topological order over the group, guaranteed by the caller) and
    resolved by block id rather than by "the previous step".

    Each `steps[i]` is `{"id", "category", "kwargs", "chain_inputs",
    "is_exit", "is_sink"}`: `kwargs` is every param this step's lazy_fn (or
    lazy_sink_fn, for a sink step) needs *except* the chained ones --
    including any already-materialized pl.DataFrame values read from cache
    for a non-chained dataframe input (the caller's job, see
    Runner._real_df_and_meta) -- lazily wrapped here so it joins the same
    unbroken plan rather than anchoring a fresh eager sub-result partway
    through. `chain_inputs` maps a kwarg name to the block id (some earlier
    step) whose still-uncollected LazyFrame output fills it -- one entry
    per internal edge, so a step can have more than one (a join with both
    sides fused) or none (a group's head).

    A step with `is_exit` True needs its LazyFrame actually collected and
    handed back as a real DataFrame; all such exits in the group are
    collected together via `pl.collect_all(..., engine="streaming")`,
    which applies common-subplan elimination across them, so a shared
    upstream stretch (fan-out) is computed once even though it feeds two
    different exits. A step with `is_sink` True is a terminal write (e.g.
    write_csv, see BlockSpec.lazy_sink_fn): its lazy_fn is never called --
    instead its `lazy_sink_fn` is called directly on its one chained input
    and writes straight to disk via the streaming engine, so the block(s)
    feeding it never need a collected DataFrame at all, in memory or in
    the cache.

    Reports `(task_key, True, (schemas, results))` on success -- `schemas`
    is `{block_id: {col: dtype_str}}` for every step (from
    `LazyFrame.collect_schema()`, free -- no data touched), so the parent
    can chain each block's own metadata_transform through the fused DAG
    without having materialized any of it; `results` is `{block_id: pl.DataFrame}`
    for exit steps only (empty for a sink, which produces no DataFrame) --
    or `(task_key, False, "step <i> (<category>, block <id>): <error>")`
    naming which block in the group actually failed. Never raises into the
    parent, same contract as run_worker_entry above."""
    import polars as pl

    import modelmaker.blocks.library  # noqa: F401 -- see run_worker_entry's comment on this import
    from .blocks.base import BLOCK_REGISTRY

    idx = -1
    schemas: dict[str, dict[str, str]] = {}
    lazy_by_id: dict[str, pl.LazyFrame] = {}
    try:
        exit_ids: list[str] = []
        for idx, step in enumerate(steps):
            spec = BLOCK_REGISTRY[step["category"]]
            kwargs = {k: (v.lazy() if isinstance(v, pl.DataFrame) else v) for k, v in step["kwargs"].items()}
            for port, src_id in step["chain_inputs"].items():
                kwargs[port] = lazy_by_id[src_id]
            if step.get("is_sink"):
                spec.lazy_sink_fn(**kwargs)
                continue
            value = spec.lazy_fn(**kwargs)
            lazy_by_id[step["id"]] = value
            schemas[step["id"]] = {name: str(dtype) for name, dtype in value.collect_schema().items()}
            if step.get("is_exit"):
                exit_ids.append(step["id"])

        results: dict[str, pl.DataFrame] = {}
        if exit_ids:
            collected = pl.collect_all([lazy_by_id[eid] for eid in exit_ids], engine="streaming")
            results = dict(zip(exit_ids, collected))
        result_queue.put((task_key, True, (schemas, results)))
    except BaseException as e:  # noqa: BLE001 -- must reach the queue, not crash the worker silently
        step = steps[idx] if 0 <= idx < len(steps) else None
        where = f"step {idx} ({step['category']}, block {step['id']!r})" if step else "step ?"
        result_queue.put((task_key, False, f"{where}: {type(e).__name__}: {e}"))


def run_iteration_entry(
    steps: list[dict[str, Any]],
    iteration_index: int,
    collect_source_id: str,
    collect_source_port_index: int,
    result_queue: Any,
    task_key: Any,
) -> None:
    """Entry point for one iteration of a fan-out region (see
    Runner._run_region_iterations/_dispatch_iterations in runner.py) --
    runs every step in `steps` (topologically ordered, one per region
    block) as an *ordinary* block call in this one subprocess, unlike
    run_fused_group_entry above, which is specifically for the handful of
    Polars-lazy-fusable blocks (filter/select/join/...). Each step's own
    return value is threaded into whichever later step's `chain_inputs`
    names it, by block id -- there's no lazy plan here, just plain
    eager calls in order, exactly like run_worker_entry's single call,
    repeated for every block in the region.

    Each `steps[i]` is `{"id", "category", "is_custom", "code", "kwargs",
    "chain_inputs", "n_outputs", "wants_iteration_index"}`: `kwargs` is
    every param and already-resolved external (outside-the-region) input
    this step's fn needs; `chain_inputs` maps a kwarg name to
    `(source_block_id, source_output_index)` for a value produced by an
    earlier step in this same iteration. `wants_iteration_index` mirrors
    the block_id/output_dir injection convention in runner.run_block --
    true only for a block whose fn accepts `_iteration_index` (the
    'iterate' block itself, and any other region block that wants to draw
    its own per-iteration randomness).

    Reports back only the single value the 'collect' block is wired from
    -- `outputs_by_id[collect_source_id][collect_source_port_index]` --
    not the whole region's output, since concatenating that one value
    across iterations is all the parent (_run_region_iterations) needs.
    Never raises into the parent, same contract as the entry points above."""
    import modelmaker.blocks.feature_analysis  # noqa: F401
    import modelmaker.blocks.library  # noqa: F401
    import modelmaker.blocks.modelling  # noqa: F401
    import modelmaker.blocks.stat_tests  # noqa: F401
    import modelmaker.blocks.stochastic  # noqa: F401

    from .blocks.base import BLOCK_REGISTRY
    from .graph import _compile_code_to_fn

    idx = -1
    outputs_by_id: dict[str, tuple[Any, ...]] = {}
    try:
        for idx, step in enumerate(steps):
            kwargs = dict(step["kwargs"])
            for port, (src_id, src_port_index) in step["chain_inputs"].items():
                kwargs[port] = outputs_by_id[src_id][src_port_index]
            if step["wants_iteration_index"]:
                kwargs["_iteration_index"] = iteration_index
            fn = _compile_code_to_fn(step["code"] or "") if step["is_custom"] else BLOCK_REGISTRY[step["category"]].fn
            raw = fn(**kwargs)
            n_out = step["n_outputs"]
            outputs_by_id[step["id"]] = (raw,) if n_out <= 1 else tuple(raw)
        result_queue.put((task_key, True, outputs_by_id[collect_source_id][collect_source_port_index]))
    except BaseException as e:  # noqa: BLE001 -- must reach the queue, not crash the worker silently
        step = steps[idx] if 0 <= idx < len(steps) else None
        where = f"step {idx} ({step['category']}, block {step['id']!r})" if step else "step ?"
        result_queue.put((task_key, False, f"iteration {iteration_index}, {where}: {type(e).__name__}: {e}"))
