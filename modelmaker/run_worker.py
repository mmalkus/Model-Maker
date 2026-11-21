"""Subprocess entry points for executing block calls in isolation (see
Runner._run_tasks in runner.py and execution/local.py). Kept in its own
module, importing as little as possible at module level, because these are
what every worker process runs: a single-use worker
(worker_pool.worker_main) looks its task's entry point up here by name and
calls it with the task's args, plus a `result_queue` to report to and the
`task_key` to report under. What an entry point puts on `result_queue` is
encoded into the data-only wire format (execution/wire.py) before it
leaves this process, so a payload here must be a dataframe, JSON-shaped,
bytes or a numpy array -- see wire.py for the full list.

Running every block call -- grouped or not -- in its own process is what
makes a runaway or misconfigured block (e.g. group_by on a near-unique
column producing far more work than expected) a crashed *worker*, caught
and reported as a normal red-block error, instead of a crashed server: a
segfault, an OS OOM-kill, or a runaway allocation in the child never
touches this process or any other block's run.
"""

from __future__ import annotations

from typing import Any

# The entry points a task may name. A local executor looks a task's entry
# up by name in this module; a runner agent (execution/agent.py), whose
# tasks come over a socket, refuses any other name.
ENTRY_POINTS = frozenset({"run_worker_entry", "run_fused_group_entry", "run_iteration_entry", "run_examples_entry"})


def resolve_fn(category: str, code: str | None):
    """A block's function inside a worker: a built-in's from the package
    (`import modelmaker` loads every built-in), a library block's defined
    from the code the task carries -- the only place a library block's
    module-level code runs (block_library.compile_fn refuses outside a
    task process)."""
    if code is not None:
        from .block_library import compile_fn

        return compile_fn(category, code)
    from .blocks.base import BLOCK_REGISTRY

    return BLOCK_REGISTRY[category].fn


# Guarding a LazyFrame's collect against running the machine out of memory
# (see _collect_lazy). MODELMAKER_COLLECT_LIMIT_MB sets the limit outright;
# 0 turns the guard off. Otherwise it's this share of the memory that's
# free when the task starts: the result is copied again on its way to the
# app process and into the cache, so a result can't take it all.
COLLECT_LIMIT_ENV = "MODELMAKER_COLLECT_LIMIT_MB"
COLLECT_FREE_SHARE = 0.4
# Rows per streamed batch: large enough that the per-batch overhead is
# noise, small enough that the guard stops close to the limit.
COLLECT_BATCH_ROWS = 1_000_000
# The phrase the assistants' run outcome keys its `next` hint on (see
# assistant_mcp.run_outcome, vscode/src/assistant.ts).
TOO_BIG_PHRASE = "too big to hold in memory"


# Bytes per value of a String/Binary column that polars' estimated_size()
# leaves out: each value has a 16-byte view (length, prefix, buffer offset)
# besides its characters, and values of up to 12 bytes live in the view
# alone. estimated_size() counts only the characters, so a column of short
# codes -- the MMYYYY dates, status codes and flags of a loan tape read as
# text -- holds several times what it reports.
STRING_VIEW_BYTES = 16


def frame_bytes(df: Any) -> int:
    """A polars DataFrame's in-memory size: polars' estimated_size() plus
    the 16-byte view per String/Binary value it leaves out (see
    STRING_VIEW_BYTES), nested ones (in List, Array and Struct columns)
    included. Errs high by up to 12 bytes per short value, never low."""
    return df.estimated_size() + STRING_VIEW_BYTES * sum(_view_count(s) for s in df.get_columns())


def _has_views(dtype: Any) -> bool:
    import polars as pl

    if dtype in (pl.String, pl.Binary):
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return _has_views(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_has_views(f.dtype) for f in dtype.fields)
    return False


def _view_count(s: Any) -> int:
    """How many String/Binary values (null slots included) `s` holds."""
    import polars as pl

    dtype = s.dtype
    if not _has_views(dtype):
        return 0
    if dtype in (pl.String, pl.Binary):
        return s.len()
    if isinstance(dtype, pl.Struct):
        return sum(_view_count(f) for f in s.struct.unnest().get_columns())
    inner = s.list.explode() if isinstance(dtype, pl.List) else s.arr.explode()
    return _view_count(inner)


class OutputTooBig(MemoryError):
    """A block's output was stopped while collecting, before it could run
    the machine out of memory."""


def memory_bytes() -> tuple[int | None, int | None]:
    """(physical memory free right now -- what the OS could hand this
    process without swapping --, the machine's total physical memory);
    either is None when it can't be told."""
    import sys

    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None, None
        return int(status.ullAvailPhys), int(status.ullTotalPhys)
    free = total = None
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    free = int(line.split()[1]) * 1024
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    # Each read on its own, so one name the platform lacks (macOS has no
    # SC_AVPHYS_PAGES) doesn't lose the other.
    if free is None:
        free = _sysconf_bytes("SC_AVPHYS_PAGES")
    if total is None:
        total = _sysconf_bytes("SC_PHYS_PAGES")
    return free, total


def _sysconf_bytes(pages_name: str) -> int | None:
    """os.sysconf(pages_name) in bytes, or None where it isn't supported."""
    import os

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf(pages_name)
    except (AttributeError, ValueError, OSError):
        return None


def available_memory_bytes() -> int | None:
    """Physical memory free right now, or None when it can't be told."""
    return memory_bytes()[0]


def collect_limit() -> tuple[int, str] | None:
    """How big a collected output may grow before it's stopped, and what
    that limit is based on (said in OutputTooBig's message, so 2.1 GB on a
    16 GB machine doesn't read as Model-Maker thinking it has 2 GB);
    None for no limit (turned off, or the free memory can't be told)."""
    import os

    raw = os.environ.get(COLLECT_LIMIT_ENV, "").strip()
    if raw:
        try:
            mb = int(raw)
        except ValueError:
            mb = None
        if mb is not None:
            return None if mb <= 0 else (mb * 1024**2, f"set by {COLLECT_LIMIT_ENV}={mb}")
    free, total = memory_bytes()
    if free is None:
        return None
    of_total = f", of {_gb(total)} in total" if total else ""
    basis = (
        f"{COLLECT_FREE_SHARE:.0%} of the {_gb(free)} of memory free when the block started{of_total}; other "
        "programs use the rest, and the result is copied again on its way to the app"
    )
    return int(free * COLLECT_FREE_SHARE), basis


def collect_limit_bytes() -> int | None:
    """collect_limit's size alone."""
    found = collect_limit()
    return None if found is None else found[0]


def _gb(n: float) -> str:
    return f"{n / 1024**3:,.1f} GB" if n >= 1024**3 else f"{n / 1024**2:,.0f} MB"


def _collect_lazy(lf: Any) -> Any:
    """`lf` collected -- streamed in batches, adding up their size as they
    come, and stopped with OutputTooBig as soon as it passes
    collect_limit(). A plain collect would hand the whole table to
    the OS in one go and, when it doesn't fit, take the worker (and on a
    Windows machine with little headroom, other programs) down with it.
    Stopping is safe: each worker process runs one task, so raising ends
    the query with the process. Without a limit, or on a polars without
    collect_batches, it's a plain collect."""
    import polars as pl

    found = collect_limit()
    if found is None or not hasattr(lf, "collect_batches"):
        return lf.collect()
    limit, basis = found
    batches: list[pl.DataFrame] = []
    rows = size = 0
    for batch in lf.collect_batches(chunk_size=COLLECT_BATCH_ROWS):
        batches.append(batch)
        rows += batch.height
        size += frame_bytes(batch)
        if size > limit:
            raise OutputTooBig(
                f"this block's output is {TOO_BIG_PHRASE}: it passed {rows:,} rows and {_gb(size)} and was still "
                f"growing, past its {_gb(limit)} limit ({basis}), so it was stopped before it ran out of "
                "memory. Reduce it inside the block before it is handed on -- select only the columns needed, "
                "filter to the population, sample whole IDs or aggregate (to one row per loan per snapshot, say) "
                "-- or mark the next block's dataframe input \"lazy\": true and use Run all (streaming) so the two "
                "run as one streamed query; sample mode keeps builds small meanwhile. "
                f"(Closing other programs frees memory for it; {COLLECT_LIMIT_ENV} sets the limit in MB, 0 turns "
                "it off -- but an output bigger than the machine's memory still won't fit.)"
            )
    if not batches:
        return pl.DataFrame(schema=lf.collect_schema())
    return pl.concat(batches, how="vertical", rechunk=False)


def _sample_by_key(value: Any, rows: int, key: str) -> Any:
    """Sample mode keyed on an ID column: keeps every row of a pseudo-random
    subset of IDs, sized so that about `rows` rows come back. All of an
    entity's rows stay together -- a loan's whole monthly history, say --
    so per-entity work downstream (forward-looking default flags, lags,
    splits by ID) sees complete histories, unlike a plain first-N-rows cut
    of a file sorted by ID and date, which keeps a few entities' histories
    and nothing else.

    The subset is the IDs whose hash falls below a threshold: deterministic
    (the same file and size give the same sample) and computed inside the
    lazy plan, so a scan never loads more than it keeps. Counting the rows
    first costs one pass over the source (cheap for Parquet: its metadata).
    `value` is a LazyFrame or a DataFrame; the result is a DataFrame."""
    import polars as pl

    lf = value.lazy() if isinstance(value, pl.DataFrame) else value
    total = lf.select(pl.len()).collect().item()
    if total <= rows:
        return _collect_lazy(lf)
    threshold = int(rows / total * 2**64)
    return _collect_lazy(lf.filter(pl.col(key).hash(seed=0) < pl.lit(threshold, dtype=pl.UInt64)))


def _collect(value: Any, sample: int | dict[str, Any] | None) -> Any:
    """A LazyFrame result collected (with sample mode's row cap pushed down
    first); a DataFrame sampled the same way; anything else as is.

    `sample` is sample mode's setting for an input block: None (off), a
    row count (the first that many rows), or {"rows": n, "key": column}
    (whole IDs, about n rows -- see _sample_by_key; used when the block
    tags a column as the ID). A key the output doesn't have falls back to
    the first n rows."""
    import polars as pl

    if sample is None or not isinstance(value, (pl.LazyFrame, pl.DataFrame)):
        return _collect_lazy(value) if isinstance(value, pl.LazyFrame) else value
    if isinstance(sample, dict):
        rows, key = int(sample["rows"]), sample.get("key")
        names = value.collect_schema().names() if isinstance(value, pl.LazyFrame) else value.columns
        if key and key in names:
            return _sample_by_key(value, rows, key)
    else:
        rows = int(sample)
    if isinstance(value, pl.LazyFrame):
        return value.head(rows).collect()
    return value.head(rows)


def run_worker_entry(
    category: str,
    code: str | None,
    kwargs: dict[str, Any],
    sample_rows: int | dict[str, Any] | None,
    result_queue: Any,
    task_key: Any,
) -> None:
    """Resolve and call one block's function, reporting success or failure
    back over `result_queue` as `(task_key, ok, payload)`. Never raises into
    the parent process -- multiprocessing.Process has no return channel of
    its own, so any escaping exception here would simply vanish and leave
    the parent's `_dispatch` waiting until it notices the process died.

    LazyFrame results are collected here (a LazyFrame can't cross the
    process boundary as data), after sample mode's cap -- set only for an
    input block, see _collect -- so the row cap reaches the scan."""
    try:
        fn = resolve_fn(category, code)
        result = fn(**kwargs)
        if isinstance(result, tuple):
            result = tuple(_collect(v, sample_rows) for v in result)
        else:
            result = _collect(result, sample_rows)
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
    runs a whole connected DAG of fusable blocks (lazy port flags -- see
    runner._fusable_block) as
    one unbroken polars lazy plan in this single subprocess, and only
    collects at the very end, with the streaming engine -- the one place
    in this app where a source larger than memory doesn't fully
    materialize at every block boundary. A group is no longer required to
    be a straight-line chain: fan-out (one block feeding several others in
    the same group) and fan-in (e.g. a join whose both sides are still
    lazy) are both single lazy plans here, built up in `steps` order (a
    topological order over the group, guaranteed by the caller) and
    resolved by block id rather than by "the previous step".

    Each `steps[i]` is `{"id", "category", "code", "kwargs", "chain_inputs",
    "is_exit", "is_sink"}`: `kwargs` is every param this step's function
    needs *except* the chained ones --
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
    write_csv, see runner._sink_block): its function is called on its one
    chained, still-lazy input and writes straight to disk via the streaming
    engine, so the block(s)
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

    idx = -1
    schemas: dict[str, dict[str, str]] = {}
    lazy_by_id: dict[str, pl.LazyFrame] = {}
    try:
        exit_ids: list[str] = []
        for idx, step in enumerate(steps):
            fn = resolve_fn(step["category"], step.get("code"))
            kwargs = {k: (v.lazy() if isinstance(v, pl.DataFrame) else v) for k, v in step["kwargs"].items()}
            for port, src_id in step["chain_inputs"].items():
                kwargs[port] = lazy_by_id[src_id]
            if step.get("is_sink"):
                fn(**kwargs)
                continue
            value = fn(**kwargs)
            if isinstance(value, pl.DataFrame):
                value = value.lazy()
            lazy_by_id[step["id"]] = value
            schemas[step["id"]] = {name: str(dtype) for name, dtype in value.collect_schema().items()}
            if step.get("is_exit"):
                exit_ids.append(step["id"])

        results: dict[str, pl.DataFrame] = {}
        if len(exit_ids) == 1:
            # One exit: streamed and guarded like any block's output (see
            # _collect_lazy). Several share subplans that collect_all
            # computes once, so they're collected together, unguarded.
            results = {exit_ids[0]: _collect_lazy(lazy_by_id[exit_ids[0]])}
        elif exit_ids:
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

    Each `steps[i]` is `{"id", "category", "code", "kwargs",
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
    idx = -1
    outputs_by_id: dict[str, tuple[Any, ...]] = {}
    try:
        for idx, step in enumerate(steps):
            kwargs = dict(step["kwargs"])
            for port, (src_id, src_port_index) in step["chain_inputs"].items():
                kwargs[port] = outputs_by_id[src_id][src_port_index]
            if step["wants_iteration_index"]:
                kwargs["_iteration_index"] = iteration_index
            fn = resolve_fn(step["category"], step.get("code"))
            raw = fn(**kwargs)
            raw = tuple(_collect(v, None) for v in raw) if isinstance(raw, tuple) else _collect(raw, None)
            n_out = step["n_outputs"]
            outputs_by_id[step["id"]] = (raw,) if n_out <= 1 else tuple(raw)
        result_queue.put((task_key, True, outputs_by_id[collect_source_id][collect_source_port_index]))
    except BaseException as e:  # noqa: BLE001 -- must reach the queue, not crash the worker silently
        step = steps[idx] if 0 <= idx < len(steps) else None
        where = f"step {idx} ({step['category']}, block {step['id']!r})" if step else "step ?"
        result_queue.put((task_key, False, f"iteration {iteration_index}, {where}: {type(e).__name__}: {e}"))


def run_examples_entry(
    source: str,
    origin: str,
    ids: list[str] | None,
    result_queue: Any,
    task_key: Any,
) -> None:
    """Run a block file's `_example_<id>` self-checks ("Try it" with no
    canvas input, `modelmaker check-blocks`; see block_library.run_examples).
    Executes the whole file -- examples call its blocks and helpers by name
    -- then each requested example, and reports `{block id: {"ok": True |
    False | None (no example), "error", "line"}}`, `line` being the file's
    own line the failure surfaced at (the editor jumps to it). A file that
    fails to run at all reports under "__file__". Never raises into the
    parent, same contract as the entry points above."""
    import traceback

    from .block_library import exec_block_source, parse_source

    out: dict[str, dict[str, Any]] = {}
    try:
        ns = exec_block_source(source, origin, "modelmaker_block_examples")
        parsed = parse_source(source, origin)
        for d in parsed.definitions:
            if ids is not None and d.id not in ids:
                continue
            name = parsed.examples.get(d.id)
            if name is None:
                out[d.id] = {"ok": None, "error": "no _example_ for this block"}
                continue
            try:
                ns[name]()
                out[d.id] = {"ok": True, "error": None}
            except BaseException as e:  # noqa: BLE001 -- reported per example
                tb = traceback.extract_tb(e.__traceback__)
                line = next((f.lineno for f in reversed(tb) if f.filename == origin), None)
                out[d.id] = {"ok": False, "error": f"{type(e).__name__}: {e}", "line": line}
    except BaseException as e:  # noqa: BLE001 -- the file itself failed to run
        out["__file__"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result_queue.put((task_key, True, out))
