from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue as queue_mod
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

import polars as pl

from .blocks.base import BLOCK_REGISTRY
from .cache import CacheStore
from .graph import BlockInstance, Graph
from .metadata_transforms import resolve_metadata_transform
from .packet import ColumnMeta, ColumnRole, DataFramePacket, find_duplicate_unique_role, resolve_role_column
from .run_worker import run_worker_entry
from .util import ROLE_PARAM_NAMES, accepts_param, find_role_param

Status = Literal["grey", "green", "orange", "red", "running"]

# Always "spawn", deliberately, even though "fork" is cheaper and available
# on Linux: this Runner is driven from a multi-threaded process (the API
# server runs each dispatch from its own background thread -- see
# api.py -- and the process may have other threads besides, e.g. from
# uvicorn/starlette or an HTTP client library). Forking a multi-threaded
# process only copies the calling thread; any lock another thread happened
# to hold at that instant (malloc arena, an import lock, a connection
# pool's lock, ...) is copied in its *locked* state with no thread left to
# release it, so the forked child can deadlock the moment it touches
# whatever that lock guards. That's not a theoretical risk here -- it's
# reproducible under this project's own multi-threaded test suite. "spawn"
# starts a genuinely fresh interpreter instead, at the cost of a slower
# process start (re-importing polars/sklearn/etc.), which is why
# run_worker_entry rebuilds a block's function from (category, code)
# rather than relying on inherited memory.
MP_CONTEXT = multiprocessing.get_context("spawn")

# How often the dispatch loop wakes up to check for a cancellation request
# or a worker that died without reporting a result -- small enough that
# Stop and crash-detection both feel immediate, large enough not to busy-loop.
POLL_INTERVAL = 0.05


class RunCancelled(RuntimeError):
    """Raised inside Runner._dispatch when the in-flight run is stopped via
    Runner.cancel() -- caught by run_block's normal exception handling, so a
    cancelled block just goes red with an informative error like any other
    failure, rather than needing separate state."""


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(value: Any, limit: int = 24) -> str:
    """A param value rendered small enough to sit inside a one-line staleness
    message (see Runner._describe_local_change)."""
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


@dataclass
class RunState:
    last_successful_key: str | None = None
    last_successful_read_at: str | None = None  # input blocks only
    last_attempt_key: str | None = None
    last_attempt_at: str | None = None
    failed: bool = False
    last_error: str | None = None
    read_counter: int = 0  # bumped by Refresh; part of an input block's key
    last_probe_value: Any = None
    # The full key basis (see Runner._basis) as it stood at the last
    # successful run, kept so staleness can be *explained* rather than just
    # detected: the key alone is a hash, so comparing it to the current one
    # says that something changed but never what. Diffed against the current
    # basis by Runner.stale_reason.
    last_successful_basis: dict[str, Any] | None = None


class Runner:
    # Safety ceiling on how many distinct groups a group_by block may fan
    # out to. Grouping on an effectively-unique column (an id, a raw
    # timestamp) would otherwise try to spawn one subprocess per row --
    # checked before any work starts (see _run_grouped), so it fails fast
    # with a clear error instead of exhausting memory/process handles.
    MAX_GROUPS = 500
    # Ceiling on concurrent worker processes for one grouped run, regardless
    # of what an individual block's max_workers asks for.
    MAX_GROUP_WORKERS = 8

    def __init__(
        self,
        graph: Graph,
        cache: CacheStore | None = None,
        output_dir: str = "./output",
        sample_rows: int | None = None,
    ):
        self.graph = graph
        self.cache = cache or CacheStore()
        self.output_dir = output_dir
        # Sample mode: when set, every input block's dataframe outputs are
        # truncated to this many rows, so the whole pipeline can be iterated
        # on cheaply. Folded into input blocks' key basis (see _basis), so
        # toggling it invalidates the graph the same way any other change
        # does -- sampled and full results are never mixed in one cache.
        self.sample_rows = sample_rows
        self.state: dict[str, RunState] = {}
        # block_id -> {"cancel_event": ...} for whichever block is currently
        # mid-dispatch (see _dispatch) -- read by status()/is_running(), and
        # is how cancel() (called from a different thread than the one
        # blocked inside run_block) reaches an in-flight run.
        self._active: dict[str, dict[str, Any]] = {}
        self._active_lock = threading.Lock()
        # Set by cancel() in addition to whatever block is currently
        # mid-dispatch -- the *in-flight* block stops via its own
        # cancel_event (checked inside _dispatch), but a cascade
        # (run_all/force_run_all/run_to_here) is just a plain Python loop
        # calling run_block one block at a time, with real gaps between
        # calls where nothing is in self._active for cancel() to reach. This
        # flag is what a cascade's loop checks in that gap to stop issuing
        # further blocks; each cascade entry point clears it first, so a
        # previous cancellation never leaks into the next run.
        self._cancel_requested = threading.Event()

    def _st(self, block_id: str) -> RunState:
        return self.state.setdefault(block_id, RunState())

    def is_running(self, block_id: str | None = None) -> bool:
        with self._active_lock:
            return (block_id in self._active) if block_id is not None else bool(self._active)

    def cancel(self, block_id: str | None = None) -> bool:
        """Stop whichever run is currently in flight. With no block_id,
        cancels whatever's active (there is only ever at most one run in
        progress per Runner -- see api.py's single-flight guard) -- the
        common case for a single global Stop button. Returns whether
        anything was actually cancelled."""
        with self._active_lock:
            targets = [block_id] if block_id is not None else list(self._active)
            handles = [self._active[b] for b in targets if b in self._active]
        self._cancel_requested.set()
        for h in handles:
            h["cancel_event"].set()
        return bool(handles)

    def _basis(self, block_id: str, _stack: frozenset[str] = frozenset()) -> dict[str, Any]:
        """Everything a block's cache key is derived from, as a plain dict --
        hashed into the key by compute_key, and kept verbatim on a successful
        run (RunState.last_successful_basis) so stale_reason can diff it and
        say *what* changed.

        `_stack` guards against a cyclic graph recursing forever (and
        crashing the process with a RecursionError): wires that would
        introduce a cycle are rejected at creation time (see
        Graph.creates_cycle / session.add_wire), but a project file loaded
        from disk could still contain one, and every read of block status
        -- including a plain GET /api/graph -- calls this, so it must fail
        cleanly rather than blow the stack."""
        if block_id in _stack:
            raise RuntimeError(f"cycle detected in graph at block '{block_id}'")
        block = self.graph.blocks[block_id]
        if block.block_type == "input":
            basis: dict[str, Any] = {
                "category": block.category,
                "code_version": block.code_version,
                "params": block.params,
                "column_role_overrides": block.column_role_overrides,
                "read_counter": self._st(block_id).read_counter,
            }
            # Only present when sample mode is actually on, so ordinary
            # full-data keys are exactly what they were before sample mode
            # existed -- a cache built by an older version stays valid, and
            # turning sample mode on and back off returns to those same keys.
            if self.sample_rows is not None:
                basis["sample_rows"] = self.sample_rows
            return basis
        next_stack = _stack | {block_id}
        upstream = {
            port: f"{self.compute_key(wire.from_block, next_stack)}:{wire.from_port}"
            for port, wire in sorted(self.graph.input_wires(block_id).items())
        }
        return {
            "category": block.category,
            "code_version": block.code_version,
            "params": block.params,
            "column_role_overrides": block.column_role_overrides,
            "code": block.code if block.is_custom else None,
            "upstream": upstream,
            "group_by": block.group_by,
            "max_workers": block.max_workers,
        }

    def compute_key(self, block_id: str, _stack: frozenset[str] = frozenset()) -> str:
        """Lineage-based cache key: hashes block identity/config plus the
        upstream blocks' *keys*, never the underlying DataFrame content, so
        computing it is cheap no matter how large the data is."""
        return _hash(self._basis(block_id, _stack))

    def status(self, block_id: str) -> Status:
        if self.is_running(block_id):
            return "running"
        st = self._st(block_id)
        try:
            current = self.compute_key(block_id)
        except RuntimeError as e:
            # e.g. a cyclic graph loaded from disk (see compute_key) -- surface
            # it as a normal red/failed block instead of raising out of a
            # plain status check, which every graph read goes through.
            st.failed = True
            st.last_error = str(e)
            return "red"
        if st.failed and st.last_attempt_key == current:
            return "red"
        if st.last_successful_key == current:
            return "green"
        if st.last_successful_key is not None:
            return "orange"
        return "grey"

    def stale_reason(self, block_id: str) -> str | None:
        """Why this block's cached output is no longer current, in one short
        phrase -- the answer to "why did this go orange?", which the key on
        its own can't give (it's a hash: it says *that* something changed,
        never what). None when there's nothing to explain (never run, or
        still current).

        Local changes are reported in preference to upstream ones, and an
        upstream change is followed to the block that actually caused it, so
        a long cascade names its origin rather than just "the block before
        me changed"."""
        st = self._st(block_id)
        if st.last_successful_basis is None:
            return None
        try:
            current = self._basis(block_id)
        except RuntimeError:
            return None
        if current == st.last_successful_basis:
            return None
        return self._describe_change(block_id, st.last_successful_basis, current)

    def _describe_change(self, block_id: str, old: dict[str, Any], new: dict[str, Any], depth: int = 0) -> str | None:
        local = self._describe_local_change(old, new)
        if local:
            return "; ".join(local)

        old_up: dict[str, str] = old.get("upstream", {})
        new_up: dict[str, str] = new.get("upstream", {})
        wires = self.graph.input_wires(block_id)
        reasons: list[str] = []
        for port in sorted(set(old_up) | set(new_up)):
            if old_up.get(port) == new_up.get(port):
                continue
            if port not in old_up:
                reasons.append(f"input '{port}' was connected")
                continue
            wire = wires.get(port)
            if wire is None or port not in new_up:
                reasons.append(f"input '{port}' was disconnected")
                continue
            # Values are "<upstream key>:<upstream port>" -- a differing port
            # with an identical key means the wire was moved to a different
            # output of the same block, which is a rewire, not a re-run.
            if old_up[port].split(":", 1)[0] == new_up[port].split(":", 1)[0]:
                reasons.append(f"input '{port}' was rewired to another output of '{self._block_name(wire.from_block)}'")
                continue
            reasons.append(self._describe_upstream(wire.from_block, depth))
        return "; ".join(r for r in reasons if r) or None

    @staticmethod
    def _describe_local_change(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if old.get("category") != new.get("category"):
            reasons.append("the block was replaced with a different kind")
        if old.get("code_version") != new.get("code_version") or old.get("code") != new.get("code"):
            reasons.append("its code changed")
        old_params, new_params = old.get("params", {}) or {}, new.get("params", {}) or {}
        if old_params != new_params:
            changed = sorted(k for k in set(old_params) | set(new_params) if old_params.get(k) != new_params.get(k))
            shown = ", ".join(f"'{k}'" for k in changed[:3]) + (f" and {len(changed) - 3} more" if len(changed) > 3 else "")
            if len(changed) == 1:
                k = changed[0]
                reasons.append(f"param {shown} changed ({_short(old_params.get(k))} → {_short(new_params.get(k))})")
            else:
                reasons.append(f"params {shown} changed")
        if old.get("column_role_overrides") != new.get("column_role_overrides"):
            reasons.append("a column role was retagged")
        if old.get("group_by") != new.get("group_by") or old.get("max_workers") != new.get("max_workers"):
            reasons.append("its grouping changed")
        if old.get("read_counter") != new.get("read_counter"):
            reasons.append("its source was re-read")
        if old.get("sample_rows") != new.get("sample_rows"):
            reasons.append("sample mode changed")
        return reasons

    def _describe_upstream(self, up_id: str, depth: int) -> str:
        """Why an upstream block's output differs, phrased from this block's
        point of view. Recurses (bounded) so a cascade points at its origin."""
        name = self._block_name(up_id)
        up_st = self._st(up_id)
        if depth < 3 and up_st.last_successful_basis is not None:
            try:
                sub = self._describe_change(up_id, up_st.last_successful_basis, self._basis(up_id), depth + 1)
            except RuntimeError:
                sub = None
            if sub:
                return f"upstream '{name}' changed: {sub}"
        return f"upstream '{name}' changed"

    def _block_name(self, block_id: str) -> str:
        block = self.graph.blocks.get(block_id)
        return block.name if block else block_id

    def _gather_inputs(self, block_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for port, wire in self.graph.input_wires(block_id).items():
            if self.status(wire.from_block) != "green":
                raise RuntimeError(f"input block '{wire.from_block}' is not green")
            pred_st = self._st(wire.from_block)
            entry = self.cache.get(pred_st.last_successful_key)  # type: ignore[arg-type]
            if entry is None:
                raise RuntimeError(f"missing cached output for '{wire.from_block}'")
            result[port] = entry.outputs[wire.from_port]
        return result

    def run_block(self, block_id: str) -> Status:
        """Requires all of this block's inputs to currently be green."""
        block = self.graph.blocks[block_id]
        st = self._st(block_id)
        basis = self._basis(block_id)
        key = _hash(basis)
        st.last_attempt_key = key
        st.last_attempt_at = _now()
        try:
            input_packets = self._gather_inputs(block_id)
            plain_inputs = {
                port: (p.data if isinstance(p, DataFramePacket) else p) for port, p in input_packets.items()
            }
            fn = block.resolved_fn()
            call_kwargs = dict(plain_inputs, **block.params)
            # Dynamic defaults: for every role a block's signature opts into
            # (target via target/target_col, predicted via
            # score_col/predicted_col, ...), resolve it fresh on every run
            # from whichever upstream column currently carries that role,
            # never written back into block.params -- so retagging the role
            # elsewhere in the graph propagates here automatically instead of
            # leaving a stale copy behind. An explicit value in block.params
            # always wins (see the `not in` check below).
            input_schema_metas = [p.schema_meta for p in input_packets.values() if isinstance(p, DataFramePacket)]
            for role in ROLE_PARAM_NAMES:
                role_param = find_role_param(fn, role)
                if role_param is not None and role_param not in block.params:
                    resolved = resolve_role_column(input_schema_metas, role)
                    if resolved is not None:
                        call_kwargs[role_param] = resolved
            if block.block_type == "output" and accepts_param(fn, "output_dir"):
                call_kwargs["output_dir"] = self.output_dir
            if block.block_type == "output" and accepts_param(fn, "block_id"):
                call_kwargs["block_id"] = block_id

            group_col = block.group_by
            if group_col:
                raw = self._run_grouped(block_id, block, group_col, call_kwargs)
            else:
                results = self._dispatch(block_id, {None: call_kwargs}, max_workers=1)
                raw = results[None]

            out_names = [p.name for p in block.outputs]
            if len(out_names) == 0:
                raw_outputs: dict[str, Any] = {}
            elif len(out_names) == 1:
                raw_outputs = {out_names[0]: raw}
            else:
                raw_outputs = dict(zip(out_names, raw))

            if block.block_type == "input" and self.sample_rows is not None:
                # Sample mode truncates at the source, so every downstream
                # block sees the sample without needing to know it exists.
                raw_outputs = {
                    k: (v.head(self.sample_rows) if isinstance(v, pl.DataFrame) else v)
                    for k, v in raw_outputs.items()
                }

            data_outputs = {k: v for k, v in raw_outputs.items() if isinstance(v, pl.DataFrame)}
            transform = block.resolved_metadata_transform()
            metas: dict[str, dict] = {}
            if transform is not None and data_outputs:
                metas = resolve_metadata_transform(transform)(
                    {port: p.schema_meta for port, p in input_packets.items() if isinstance(p, DataFramePacket)},
                    data_outputs,
                    block.params,
                )
                if group_col:
                    # A grouped run's combined dataframe often didn't exist
                    # as a single frame anywhere the block's own transform
                    # could describe (e.g. auc_gini's dict output became a
                    # per-group table -- see _combine_group_results): fall
                    # back to plain dtype inference for any port the
                    # transform left undescribed, and always tag the
                    # grouping column itself as a segment.
                    for port, df in data_outputs.items():
                        port_meta = metas.setdefault(port, {})
                        for name in df.columns:
                            if name not in port_meta:
                                port_meta[name] = ColumnMeta(dtype=str(df.schema[name]))
                        if group_col in port_meta:
                            port_meta[group_col] = replace(port_meta[group_col], role=ColumnRole.SEGMENT)
                if block.column_role_overrides:
                    for port_meta in metas.values():
                        for name, role_value in block.column_role_overrides.items():
                            if name in port_meta:
                                port_meta[name] = replace(port_meta[name], role=ColumnRole(role_value))
                # The backstop half of the uniqueness rule (see
                # packet.find_duplicate_unique_role): a hand-tagged column
                # can only collide with what was already on *this* block's
                # own schema (checked immediately in session.set_column_role
                # instead), but two upstream branches independently tagged
                # target/id/weight/... can still collide the moment
                # something -- chiefly join -- merges their schemas into
                # one. That can only be caught here, once the merge actually
                # happens, so it's a hard failure rather than picking a
                # winner silently.
                for port_meta in metas.values():
                    dup = find_duplicate_unique_role(port_meta)
                    if dup:
                        role, cols = dup
                        raise ValueError(f"role '{role.value}' is set on more than one column: {', '.join(cols)}")

            packets: dict[str, Any] = {}
            for name, value in raw_outputs.items():
                if isinstance(value, pl.DataFrame):
                    packets[name] = DataFramePacket(data=value, schema_meta=metas.get(name, {})).with_lineage(
                        block_id
                    )
                else:
                    packets[name] = value

            self.cache.set(key, packets)
            st.last_successful_key = key
            st.last_successful_basis = basis
            st.failed = False
            st.last_error = None
            if block.block_type == "input":
                st.last_successful_read_at = st.last_attempt_at
        except Exception as e:  # noqa: BLE001 -- captured as block state, not propagated
            st.failed = True
            st.last_error = f"{type(e).__name__}: {e}"
        return self.status(block_id)

    def _dispatch(self, block_id: str, tasks: dict[Any, dict[str, Any]], max_workers: int) -> dict[Any, Any]:
        """Run each of `tasks` (an arbitrary key -> call kwargs) in its own
        subprocess, up to `max_workers` at a time, and return {key: result}.
        Blocks the calling thread until every task finishes, the run is
        cancelled (see cancel(), raises RunCancelled), or a worker dies
        without reporting a result -- treated as a crash (segfault, OS
        OOM-kill) and surfaced as a normal RuntimeError, never as a hang.

        This is the isolation boundary: whatever a block's code does --
        including a group_by fan-out with far more groups or memory use
        than expected -- happens in a child process, so it can only take
        itself down, never this one."""
        block = self.graph.blocks[block_id]
        result_queue = MP_CONTEXT.Queue()
        cancel_event = MP_CONTEXT.Event()
        pending = list(tasks.items())
        running: dict[Any, Any] = {}  # task key -> Process
        results: dict[Any, Any] = {}
        errors: dict[Any, str] = {}

        with self._active_lock:
            self._active[block_id] = {"cancel_event": cancel_event}

        def _start_next() -> None:
            while pending and len(running) < max(1, max_workers):
                key, kwargs = pending.pop(0)
                p = MP_CONTEXT.Process(
                    target=run_worker_entry,
                    args=(block.category, block.is_custom, block.code, kwargs, result_queue, key),
                    daemon=True,
                )
                p.start()
                running[key] = p

        try:
            _start_next()
            while running:
                if cancel_event.is_set():
                    raise RunCancelled("cancelled by user")
                try:
                    key, ok, payload = result_queue.get(timeout=POLL_INTERVAL)
                except queue_mod.Empty:
                    dead = [k for k, p in running.items() if not p.is_alive()]
                    for k in dead:
                        p = running.pop(k)
                        errors[k] = f"worker process exited unexpectedly (code {p.exitcode}) -- likely out of memory or a crash"
                    continue
                proc = running.pop(key, None)
                if proc is not None:
                    proc.join(timeout=5)
                if ok:
                    results[key] = payload
                else:
                    errors[key] = payload
                _start_next()
        finally:
            for p in running.values():
                if p.is_alive():
                    p.terminate()
            for p in running.values():
                p.join(timeout=5)
            with self._active_lock:
                self._active.pop(block_id, None)

        if errors:
            first_key, first_err = next(iter(errors.items()))
            suffix = "" if len(tasks) == 1 else f" (group {first_key!r}; {len(errors)}/{len(tasks)} group(s) failed)"
            raise RuntimeError(f"{first_err}{suffix}")
        return results

    def _run_grouped(
        self, block_id: str, block: BlockInstance, group_col: str, call_kwargs: dict[str, Any]
    ) -> Any:
        """Partition every dataframe kwarg that actually has `group_col`,
        run the block's function once per distinct value (each in its own
        subprocess, see _dispatch), and recombine -- see
        _combine_group_results. Guards against grouping on an effectively
        unique column (MAX_GROUPS) before spawning anything."""
        df_kwargs = {k: v for k, v in call_kwargs.items() if isinstance(v, pl.DataFrame)}
        groupable_ports = [k for k, v in df_kwargs.items() if group_col in v.columns]
        if not groupable_ports:
            raise ValueError(f"group_by column {group_col!r} not found in any dataframe input")

        primary_port = groupable_ports[0]
        primary_df = df_kwargs[primary_port]
        n_groups = primary_df.select(pl.col(group_col).n_unique()).item() or 0
        if n_groups == 0:
            raise ValueError(f"no groups found for group_by column {group_col!r}")
        if n_groups > self.MAX_GROUPS:
            raise ValueError(
                f"grouping by {group_col!r} would produce {n_groups} groups, over the {self.MAX_GROUPS}-group "
                "safety limit -- pick a lower-cardinality column (this usually means the column isn't really "
                "categorical, e.g. an id or a raw timestamp)"
            )

        partitions: dict[str, dict[Any, pl.DataFrame]] = {}
        for port in groupable_ports:
            parts = df_kwargs[port].partition_by(group_col, as_dict=True)
            partitions[port] = {key[0]: sub for key, sub in parts.items()}
        group_values = list(partitions[primary_port].keys())

        tasks: dict[Any, dict[str, Any]] = {}
        for gval in group_values:
            gkwargs = dict(call_kwargs)
            for port in groupable_ports:
                sub = partitions[port].get(gval)
                if sub is not None:
                    gkwargs[port] = sub
            tasks[gval] = gkwargs

        max_workers = max(1, min(block.max_workers or self.MAX_GROUP_WORKERS, self.MAX_GROUP_WORKERS, n_groups))
        results = self._dispatch(block_id, tasks, max_workers)
        return self._combine_group_results(block, group_col, group_values, results)

    @staticmethod
    def _combine_group_results(
        block: BlockInstance, group_col: str, group_values: list[Any], results: dict[Any, Any]
    ) -> Any:
        """Reshape {group_value: this block's normal return value} into the
        same single/tuple shape run_block expects from an ungrouped call --
        a dataframe output gets every group's rows concatenated (the group
        column added back if the block's own output dropped it, e.g. an
        aggregation); a dict output (a scalar_metric block like auc_gini)
        becomes one small dataframe, one row per group, so "Gini per
        region" is a single table rather than N disconnected numbers.
        Anything else (a model artifact, an image, ...) has no obvious
        per-group merge, so it's kept as a {group_value: value} dict rather
        than silently discarding every group but one."""
        out_names = [p.name for p in block.outputs]
        if not out_names:
            return None
        per_group: dict[Any, tuple] = {}
        for gval in group_values:
            raw = results[gval]
            per_group[gval] = (raw,) if len(out_names) == 1 else tuple(raw)

        combined: list[Any] = []
        for i in range(len(out_names)):
            values = [per_group[g][i] for g in group_values]
            if all(isinstance(v, pl.DataFrame) for v in values):
                parts = [
                    v if group_col in v.columns else v.with_columns(pl.lit(gval).alias(group_col))
                    for gval, v in zip(group_values, values)
                ]
                combined.append(pl.concat(parts, how="diagonal_relaxed"))
            elif all(isinstance(v, dict) for v in values):
                combined.append(pl.DataFrame([{group_col: gval, **v} for gval, v in zip(group_values, values)]))
            else:
                combined.append(dict(zip(group_values, values)))
        return combined[0] if len(combined) == 1 else tuple(combined)

    def _ordered_ancestors(self, block_id: str) -> list[str]:
        anc = self.graph.ancestors([block_id])
        anc.discard(block_id)
        order = self.graph.topo_order()
        return [b for b in order if b in anc]

    def run_to_here(self, block_id: str) -> Status:
        """Cascades: runs whatever upstream chain isn't green, then this block."""
        self._cancel_requested.clear()
        for pred in self._ordered_ancestors(block_id):
            if self._cancel_requested.is_set():
                return self.status(block_id)
            if self.graph.blocks[pred].block_type == "input":
                continue
            if self.status(pred) != "green":
                self.run_block(pred)
        if self._cancel_requested.is_set():
            return self.status(block_id)
        return self.run_block(block_id)

    def _blocked_on_ungread_input(self, block_id: str) -> bool:
        preds = self.graph.ancestors([block_id]) - {block_id}
        return any(
            self.graph.blocks[p].block_type == "input" and self.status(p) == "grey" for p in preds
        )

    def run_all(self) -> dict[str, str]:
        """Topological order; skips already-green blocks; input blocks are
        never (re)run here — see refresh()/refresh_all()."""
        self._cancel_requested.clear()
        report: dict[str, str] = {}
        for bid in self.graph.topo_order():
            if self._cancel_requested.is_set():
                report[bid] = "cancelled"
                continue
            block = self.graph.blocks[bid]
            if block.block_type == "input":
                continue
            if self._blocked_on_ungread_input(bid):
                report[bid] = "blocked: upstream input block has never been read"
                continue
            if self.status(bid) != "green":
                self.run_block(bid)
            report[bid] = self.status(bid)
        return report

    def force_run_all(self) -> dict[str, str]:
        """As run_all, but ignores cache entirely for non-input blocks."""
        self._cancel_requested.clear()
        report: dict[str, str] = {}
        for bid in self.graph.topo_order():
            if self._cancel_requested.is_set():
                report[bid] = "cancelled"
                continue
            block = self.graph.blocks[bid]
            if block.block_type == "input":
                continue
            if self._blocked_on_ungread_input(bid):
                report[bid] = "blocked: upstream input block has never been read"
                continue
            self.run_block(bid)
            report[bid] = self.status(bid)
        return report

    def refresh(self, block_id: str) -> Status:
        """Re-reads an input block's source and swaps the cached packet. A
        failed refresh flips the block red but keeps the last-known-good
        packet and its 'last successful read' timestamp intact."""
        block = self.graph.blocks[block_id]
        assert block.block_type == "input", "refresh() is only valid for input blocks"
        self._st(block_id).read_counter += 1
        return self.run_block(block_id)

    def refresh_all(self) -> dict[str, str]:
        return {bid: self.refresh(bid) for bid, b in self.graph.blocks.items() if b.block_type == "input"}

    def check_for_changes(self, block_id: str) -> bool:
        """Cheap read-only probe (e.g. file mtime) — does not touch the
        cached packet or cascade anything."""
        block = self.graph.blocks[block_id]
        spec = BLOCK_REGISTRY.get(block.category)
        if not spec or not spec.probe:
            return False
        st = self._st(block_id)
        current = spec.probe(block.params)
        changed = st.last_probe_value is not None and current != st.last_probe_value
        st.last_probe_value = current
        return changed

    def check_all_sources(self) -> dict[str, bool]:
        return {
            bid: self.check_for_changes(bid) for bid, b in self.graph.blocks.items() if b.block_type == "input"
        }
