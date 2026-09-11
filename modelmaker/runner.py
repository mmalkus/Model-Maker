from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import queue as queue_mod
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

import polars as pl

from .blocks.base import BLOCK_REGISTRY, BlockSpec
from .cache import CacheStore
from .graph import BlockInstance, Graph
from .metadata_transforms import resolve_metadata_transform
from .packet import ColumnMeta, ColumnRole, DataFramePacket, find_duplicate_unique_role, resolve_role_column
from .run_worker import run_fused_group_entry, run_worker_entry
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


@dataclass(frozen=True)
class RunPlan:
    """An immutable view of everything a block's cache key is derived from:
    the graph, each input block's read counter, and the sample-row cap.

    Runs execute against a pinned plan (see Runner.pin) while status reads
    answer against a live one, which is what lets the graph be edited while
    a run is in flight without the run and the canvas contradicting each
    other."""

    graph: Graph
    read_counters: dict[str, int]
    sample_rows: int | None

    def basis(self, block_id: str, _stack: frozenset[str] = frozenset()) -> dict[str, Any]:
        """The key basis as a plain dict -- hashed into the key by `key`, and
        kept verbatim on a successful run (RunState.last_successful_basis) so
        stale_reason can diff it and say *what* changed.

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
                "read_counter": self.read_counters.get(block_id, 0),
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
            port: f"{_hash(self.basis(wire.from_block, next_stack))}:{wire.from_port}"
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

    def key(self, block_id: str) -> str:
        return _hash(self.basis(block_id))


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


class _SchemaView:
    """Duck-types just enough of pl.DataFrame (`.columns`, `.schema[name]`)
    for a metadata_transform to run against a plain `{name: dtype_str}`
    schema snapshot (see run_worker.run_fused_group_entry's
    `LazyFrame.collect_schema()`) instead of a real, materialized
    DataFrame -- used only by Runner._run_fused_group, so a streaming
    run's fused, uncached interior blocks still get full metadata_transform
    treatment (role propagation, dtype inference) without ever touching
    their actual data. Every registry metadata_transform only ever reads
    these two attributes (confirmed against metadata_transforms.py and
    library.py's _groupby_meta/_join_meta), so no transform code needs to
    know this isn't a real DataFrame."""

    def __init__(self, schema: dict[str, str]):
        self.columns = list(schema.keys())
        self.schema = schema


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

    def _live_plan(self) -> RunPlan:
        """A view over the graph as it is *right now* -- what status reads
        answer against, so the canvas always reflects the current graph even
        while a run built on an older one is still executing."""
        return RunPlan(
            graph=self.graph,
            read_counters={bid: st.read_counter for bid, st in self.state.items()},
            sample_rows=self.sample_rows,
        )

    def pin(self) -> RunPlan:
        """Freeze everything a run's cache keys depend on, for the duration
        of that run. The graph is deep-copied, so edits made while the run is
        in flight -- including an undo, which replaces the graph object
        wholesale -- can't reach the blocks being executed, and a block's
        code and params can never be read from two different versions of it.

        A run therefore always describes one coherent version of the graph.
        Results land under that version's keys; a block edited mid-run simply
        computes a different key afterwards and shows as stale, with
        stale_reason naming the edit -- which is the truth, and costs no
        extra bookkeeping because the cache is keyed by configuration
        rather than by when it ran."""
        return RunPlan(
            graph=copy.deepcopy(self.graph),
            read_counters={bid: st.read_counter for bid, st in self.state.items()},
            sample_rows=self.sample_rows,
        )

    def _basis(self, block_id: str) -> dict[str, Any]:
        return self._live_plan().basis(block_id)

    def compute_key(self, block_id: str) -> str:
        """Lineage-based cache key for the block as it stands now. The
        execution path uses a pinned plan's key instead (see pin)."""
        return self._live_plan().key(block_id)

    def _is_current(self, block_id: str, plan: RunPlan) -> bool:
        """Whether this block's cached output is the one `plan` asks for --
        the execution path's equivalent of `status() == "green"`, asked
        against the run's own pinned version of the graph rather than
        against whatever the canvas currently shows."""
        return self._st(block_id).last_successful_key == plan.key(block_id)

    def status(self, block_id: str) -> Status:
        # A block can be absent from the live graph while a run pinned to an
        # older version of it is still executing (see pin) -- it has no
        # current status, rather than being an error to look one up.
        if block_id not in self.graph.blocks:
            return "grey"
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
        # Two inputs fed by the same disturbance (a split block's train and
        # test, say) would otherwise say the same thing twice.
        return "; ".join(dict.fromkeys(r for r in reasons if r)) or None

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

    # How far up a chain of stale blocks to look for the edit that started
    # it. The message doesn't grow with depth (see _describe_upstream), so
    # this only bounds work, not readability.
    MAX_STALE_DEPTH = 8

    def _describe_upstream(self, up_id: str, depth: int) -> str:
        """Why an upstream block's output differs, phrased from this block's
        point of view. Recurses (bounded) so a cascade names the edit that
        started it rather than just the neighbour that passed it on."""
        name = self._block_name(up_id)
        up_st = self._st(up_id)
        if depth < self.MAX_STALE_DEPTH and up_st.last_successful_basis is not None:
            try:
                sub = self._describe_change(up_id, up_st.last_successful_basis, self._basis(up_id), depth + 1)
            except RuntimeError:
                sub = None
            if sub:
                # A reason that is itself about an upstream block already
                # names the origin -- pass it through rather than nesting
                # another "upstream X changed:" in front of it, which is how
                # a five-block chain turns into an unreadable sentence.
                return sub if sub.startswith("upstream ") else f"upstream '{name}' changed: {sub}"
        return f"upstream '{name}' changed"

    def _block_name(self, block_id: str) -> str:
        block = self.graph.blocks.get(block_id)
        return block.name if block else block_id

    def _gather_inputs(self, block_id: str, plan: RunPlan) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for port, wire in plan.graph.input_wires(block_id).items():
            # Asked against the run's own version of the graph: an upstream
            # block the user has edited since this run started is stale on
            # the canvas, but the output this run computed for it is still
            # exactly what this run should consume.
            if not self._is_current(wire.from_block, plan):
                raise RuntimeError(f"input block '{wire.from_block}' is not green")
            entry = self.cache.get(plan.key(wire.from_block))
            if entry is None:
                raise RuntimeError(f"missing cached output for '{wire.from_block}'")
            result[port] = entry.outputs[wire.from_port]
        return result

    def _real_df_and_meta(self, wire, plan: RunPlan) -> tuple[pl.DataFrame, dict[str, ColumnMeta]]:
        """Like a single entry of _gather_inputs, but unwrapped to the real
        pl.DataFrame plus its schema_meta -- used by _run_fused_group to
        source a fused chain's non-chained dataframe inputs (a checkpoint
        predecessor's cached output) since the worker needs a plain
        DataFrame it can `.lazy()`, not a DataFramePacket."""
        if not self._is_current(wire.from_block, plan):
            raise RuntimeError(f"input block '{wire.from_block}' is not green")
        entry = self.cache.get(plan.key(wire.from_block))
        if entry is None:
            raise RuntimeError(f"missing cached output for '{wire.from_block}'")
        value = entry.outputs[wire.from_port]
        if isinstance(value, DataFramePacket):
            return value.data, value.schema_meta
        return value, {}

    def _fusable_spec(self, block: BlockInstance) -> BlockSpec | None:
        """The registry spec for `block` if it's eligible to join a
        streaming run's fusion (see _build_fusion_groups), else None.
        Fusable means: a registry block (never custom/AI-authored code --
        the LLM contract in llm/prompts.py keeps assuming plain eager
        pl.DataFrame) with a lazy_fn (see BlockSpec.lazy_fn), no group_by
        (grouping needs materialized data to partition_by), and no
        role/output_dir/block_id auto-fill -- fusable blocks (filter/
        select/groupby_agg/join/read_csv today) only ever take plain
        params, and this also guards against a future lazy_fn'd block that
        actually needs one of these, which _run_fused_group doesn't
        resolve."""
        if block.is_custom or block.group_by:
            return None
        spec = BLOCK_REGISTRY.get(block.category)
        if spec is None or spec.lazy_fn is None:
            return None
        fn = spec.fn
        if accepts_param(fn, "output_dir") or accepts_param(fn, "block_id"):
            return None
        if any(find_role_param(fn, role) for role in ROLE_PARAM_NAMES):
            return None
        return spec

    def run_block(self, block_id: str, plan: RunPlan | None = None) -> Status:
        """Requires all of this block's inputs to currently be green.

        Executes against `plan` -- a frozen view of the graph (see pin) --
        so that everything this run reads about the block, from its params
        to the code handed to the worker, comes from one coherent version of
        it. A single-block run pins its own."""
        plan = plan or self.pin()
        block = plan.graph.blocks[block_id]
        st = self._st(block_id)
        basis = plan.basis(block_id)
        key = _hash(basis)
        st.last_attempt_key = key
        st.last_attempt_at = _now()
        try:
            input_packets = self._gather_inputs(block_id, plan)
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
            # Same injection pattern as output_dir/block_id above: an input
            # block that names a `sample_rows` parameter (e.g. read_csv, via
            # a lazy scan) gets sample mode's row cap pushed to its source
            # instead of reading everything and truncating after the fact
            # (the fallback below, for input blocks that don't opt in).
            if block.block_type == "input" and plan.sample_rows is not None and accepts_param(fn, "sample_rows"):
                call_kwargs["sample_rows"] = plan.sample_rows

            group_col = block.group_by
            if group_col:
                raw = self._run_grouped(block_id, block, group_col, call_kwargs)
            else:
                results = self._dispatch(block_id, block, {None: call_kwargs}, max_workers=1)
                raw = results[None]

            out_names = [p.name for p in block.outputs]
            if len(out_names) == 0:
                raw_outputs: dict[str, Any] = {}
            elif len(out_names) == 1:
                raw_outputs = {out_names[0]: raw}
            else:
                raw_outputs = dict(zip(out_names, raw))

            if block.block_type == "input" and plan.sample_rows is not None:
                # Sample mode truncates at the source, so every downstream
                # block sees the sample without needing to know it exists.
                # A no-op for a block that already honored `sample_rows`
                # above (its output is at most this many rows already) --
                # this stays the backstop for any input block that doesn't
                # accept the param, custom AI-authored ones included.
                raw_outputs = {
                    k: (v.head(plan.sample_rows) if isinstance(v, pl.DataFrame) else v)
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

    def _dispatch(
        self, block_id: str, block: BlockInstance, tasks: dict[Any, dict[str, Any]], max_workers: int
    ) -> dict[Any, Any]:
        """Run each of `tasks` (an arbitrary key -> call kwargs) in its own
        subprocess, up to `max_workers` at a time, and return {key: result}.

        `block` is the caller's pinned copy, never a fresh lookup in the live
        graph: the kwargs were built from that same copy, and re-reading here
        is how a code edit landing mid-run used to send *new* code to the
        worker alongside *old* params -- producing output that matched
        neither, cached under a key describing the old version.
        Blocks the calling thread until every task finishes, the run is
        cancelled (see cancel(), raises RunCancelled), or a worker dies
        without reporting a result -- treated as a crash (segfault, OS
        OOM-kill) and surfaced as a normal RuntimeError, never as a hang.

        This is the isolation boundary: whatever a block's code does --
        including a group_by fan-out with far more groups or memory use
        than expected -- happens in a child process, so it can only take
        itself down, never this one."""
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

    def _dispatch_fused_group(
        self, group_key: str, steps: list[dict[str, Any]]
    ) -> tuple[list[dict[str, str]], pl.DataFrame]:
        """Runs one streaming run's fused chain (see _build_fusion_groups)
        to completion in its own subprocess -- same isolation rationale as
        _dispatch above (a crash or runaway allocation only takes down this
        worker), trimmed to a single task since a fused group is inherently
        one unit of work, not a pool of independent ones, so there's no
        worker-pool bookkeeping to do. `group_key` is the group's exit
        block id -- used the same way `_dispatch`'s `block_id` is, so
        cancel()/is_running() work identically for a fused group as for any
        other in-flight run. Returns (per_step_schema, collected_result) on
        success; raises RunCancelled or RuntimeError exactly like _dispatch
        does on cancellation or a worker crash/error."""
        result_queue = MP_CONTEXT.Queue()
        cancel_event = MP_CONTEXT.Event()

        with self._active_lock:
            self._active[group_key] = {"cancel_event": cancel_event}

        p = MP_CONTEXT.Process(target=run_fused_group_entry, args=(steps, result_queue, group_key), daemon=True)
        p.start()
        try:
            while True:
                if cancel_event.is_set():
                    raise RunCancelled("cancelled by user")
                try:
                    _key, ok, payload = result_queue.get(timeout=POLL_INTERVAL)
                    break
                except queue_mod.Empty:
                    if not p.is_alive():
                        raise RuntimeError(
                            f"worker process exited unexpectedly (code {p.exitcode}) -- likely out of memory or a crash"
                        )
                    continue
        finally:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
            with self._active_lock:
                self._active.pop(group_key, None)

        if not ok:
            raise RuntimeError(payload)
        return payload

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
        results = self._dispatch(block_id, block, tasks, max_workers)
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

    def _run_fused_group(self, group: list[tuple[str, str | None]], plan: RunPlan, report: dict[str, str]) -> None:
        """Runs one fusion group (see _build_fusion_groups) end to end and
        writes `report` in place -- the streaming-run counterpart of
        run_block for a chain of 2+ blocks fused into a single polars
        query. Only the group's exit block (its last member) gets a real
        cache entry/RunState update; interior members are marked "fused"
        and left otherwise untouched, exactly as documented on
        run_all_streaming."""
        steps: list[dict[str, Any]] = []
        input_metas_per_step: list[dict[str, dict[str, ColumnMeta]]] = []

        for bid, chain_port in group:
            block = plan.graph.blocks[bid]
            spec = self._fusable_spec(block)
            assert spec is not None, f"non-fusable block {bid!r} in a fusion group"
            kwargs: dict[str, Any] = dict(block.params)
            input_metas: dict[str, dict[str, ColumnMeta]] = {}
            for port, wire in plan.graph.input_wires(bid).items():
                if port == chain_port:
                    continue  # filled in below, chained from the previous step's real output schema
                df, meta = self._real_df_and_meta(wire, plan)
                kwargs[port] = df
                input_metas[port] = meta
            steps.append({"category": block.category, "kwargs": kwargs, "chain_port": chain_port})
            input_metas_per_step.append(input_metas)

        exit_id, _ = group[-1]
        try:
            schemas, final_df = self._dispatch_fused_group(exit_id, steps)
        except Exception as e:  # noqa: BLE001 -- captured as block state, same contract as run_block
            st = self._st(exit_id)
            st.last_attempt_key = plan.key(exit_id)
            st.last_attempt_at = _now()
            st.failed = True
            st.last_error = f"{type(e).__name__}: {e}"
            report[exit_id] = "red"
            for bid, _ in group[:-1]:
                report[bid] = "fused (group failed)"
            return

        # Chain each member's own metadata_transform forward using the
        # schemas the worker returned -- collect_schema() never touched
        # data, so this reconstructs full role/dtype tracking (including
        # column_role_overrides and the duplicate-unique-role check) through
        # the fused stretch without having materialized any interior
        # result, not just a bare dtype fallback at the exit.
        out_meta: dict[str, ColumnMeta] = {}
        for (bid, chain_port), input_metas, schema in zip(group, input_metas_per_step, schemas):
            block = plan.graph.blocks[bid]
            if chain_port is not None:
                input_metas[chain_port] = out_meta
            spec = BLOCK_REGISTRY[block.category]
            metas = spec.metadata_transform(input_metas, {"out": _SchemaView(schema)}, block.params)
            out_meta = dict(metas.get("out", {}))
            if block.column_role_overrides:
                for name, role_value in block.column_role_overrides.items():
                    if name in out_meta:
                        out_meta[name] = replace(out_meta[name], role=ColumnRole(role_value))
            dup = find_duplicate_unique_role(out_meta)
            if dup:
                role, cols = dup
                st = self._st(exit_id)
                st.last_attempt_key = plan.key(exit_id)
                st.last_attempt_at = _now()
                st.failed = True
                st.last_error = f"role '{role.value}' is set on more than one column: {', '.join(cols)}"
                report[exit_id] = "red"
                for other_bid, _ in group[:-1]:
                    report[other_bid] = "fused (group failed)"
                return
            if bid != exit_id:
                report[bid] = "fused"

        key = plan.key(exit_id)
        packet = DataFramePacket(data=final_df, schema_meta=out_meta).with_lineage(exit_id)
        self.cache.set(key, {"out": packet})
        st = self._st(exit_id)
        st.last_attempt_key = key
        st.last_attempt_at = _now()
        st.last_successful_key = key
        st.last_successful_basis = plan.basis(exit_id)
        st.failed = False
        st.last_error = None
        report[exit_id] = self.status(exit_id) if exit_id in self.graph.blocks else "deleted during run"

    def _ordered_ancestors(self, block_id: str, plan: RunPlan) -> list[str]:
        anc = plan.graph.ancestors([block_id])
        anc.discard(block_id)
        order = plan.graph.topo_order()
        return [b for b in order if b in anc]

    def run_to_here(self, block_id: str) -> Status:
        """Cascades: runs whatever upstream chain isn't green, then this block."""
        self._cancel_requested.clear()
        plan = self.pin()
        for pred in self._ordered_ancestors(block_id, plan):
            if self._cancel_requested.is_set():
                return self.status(block_id)
            if plan.graph.blocks[pred].block_type == "input":
                continue
            if not self._is_current(pred, plan):
                self.run_block(pred, plan)
        if self._cancel_requested.is_set():
            return self.status(block_id)
        return self.run_block(block_id, plan)

    def _blocked_on_unread_input(self, block_id: str, plan: RunPlan) -> bool:
        preds = plan.graph.ancestors([block_id]) - {block_id}
        return any(
            plan.graph.blocks[p].block_type == "input" and self._st(p).last_successful_key is None for p in preds
        )

    def _sweep(self, plan: RunPlan, force: bool) -> dict[str, str]:
        report: dict[str, str] = {}
        for bid in plan.graph.topo_order():
            if self._cancel_requested.is_set():
                report[bid] = "cancelled"
                continue
            if plan.graph.blocks[bid].block_type == "input":
                continue
            if self._blocked_on_unread_input(bid, plan):
                report[bid] = "blocked: upstream input block has never been read"
                continue
            if force or not self._is_current(bid, plan):
                self.run_block(bid, plan)
            # Reported against the live graph, which is what the user is
            # looking at -- a block they edited mid-sweep really is stale
            # now, however well its run went.
            report[bid] = self.status(bid) if bid in self.graph.blocks else "deleted during run"
        return report

    def run_all(self) -> dict[str, str]:
        """Topological order; skips blocks already current under this run's
        pinned graph; input blocks are never (re)run here — see
        refresh()/refresh_all()."""
        self._cancel_requested.clear()
        return self._sweep(self.pin(), force=False)

    def force_run_all(self) -> dict[str, str]:
        """As run_all, but ignores cache entirely for non-input blocks."""
        self._cancel_requested.clear()
        return self._sweep(self.pin(), force=True)

    def _streaming_blocked(self, block_id: str, plan: RunPlan) -> bool:
        """Like _blocked_on_unread_input, but a fusable input ancestor (its
        source can be re-scanned live -- see _fusable_spec/BlockSpec.lazy_fn)
        never blocks, even if it's never been explicitly refreshed: forcing
        an eager refresh() first would defeat the point of streaming for
        exactly the large-source case it exists for. A non-fusable input
        ancestor still needs a prior refresh(), same as an ordinary run."""
        preds = plan.graph.ancestors([block_id]) - {block_id}
        return any(
            plan.graph.blocks[p].block_type == "input"
            and self._fusable_spec(plan.graph.blocks[p]) is None
            and not self._is_current(p, plan)
            for p in preds
        )

    def _streaming_to_run(self, plan: RunPlan) -> tuple[list[str], dict[str, str]]:
        """The topo-ordered set of blocks a streaming run needs to (re)
        compute, plus a report entry for anything it can't run yet --
        mirrors _sweep, except a fusable input block is included (as a
        potential fusion-group head, scanned live) even when it's never
        been read, rather than always skipped the way _sweep skips every
        input block (see _streaming_blocked above for why)."""
        to_run: list[str] = []
        report: dict[str, str] = {}
        for bid in plan.graph.topo_order():
            block = plan.graph.blocks[bid]
            if block.block_type == "input":
                if self._is_current(bid, plan):
                    continue  # already cached and valid -- read from cache like any checkpoint
                if self._fusable_spec(block) is None:
                    report[bid] = "blocked: upstream input block has never been read"
                    continue
                to_run.append(bid)
                continue
            if self._streaming_blocked(bid, plan):
                report[bid] = "blocked: upstream input block has never been read"
                continue
            if not self._is_current(bid, plan):
                to_run.append(bid)
        return to_run, report

    def _build_fusion_groups(self, to_run: list[str], plan: RunPlan) -> list[list[tuple[str, str | None]]]:
        """Partitions `to_run` (topo-ordered) into fusion groups: a maximal
        straight-line chain of fusable blocks per group, each entry a
        (block_id, chain_port) pair -- chain_port is None for a group's
        head (all of whose dataframe inputs, if any, come from cache or,
        for an input block, from its own params) and names the one port
        that receives the previous member's still-lazy output for every
        later member.

        Block `b` extends predecessor `p`'s group only if `p` is fusable,
        already placed in a group, and `b` is the *only* not-yet-current
        dataframe consumer of `p` anywhere in the graph -- multiple
        qualifying predecessors (fan-in) or multiple such consumers of one
        predecessor (fan-out) both fall back to starting a new group
        instead, which just means less fusion, never wrong results (see
        the MVP scope note on this in the streaming-run design). A
        non-fusable block is always its own singleton group and can never
        be chained onto by a later block."""
        to_run_set = set(to_run)
        groups: list[list[tuple[str, str | None]]] = []
        group_index: dict[str, int] = {}

        for bid in to_run:
            block = plan.graph.blocks[bid]
            spec = self._fusable_spec(block)
            if spec is None:
                groups.append([(bid, None)])
                group_index[bid] = len(groups) - 1
                continue

            port_types = {p.name: p.type for p in block.inputs}
            chainable: list[tuple[str, str]] = []
            for port, wire in plan.graph.input_wires(bid).items():
                if port_types.get(port) != "dataframe":
                    continue
                pred = wire.from_block
                if pred not in to_run_set or pred not in group_index:
                    continue
                if self._fusable_spec(plan.graph.blocks[pred]) is None:
                    continue
                consumers = {
                    w.to_block
                    for w in plan.graph.wires.values()
                    if w.from_block == pred
                    and w.to_block in to_run_set
                    and {p.name: p.type for p in plan.graph.blocks[w.to_block].inputs}.get(w.to_port) == "dataframe"
                }
                if consumers != {bid}:
                    continue  # fan-out: pred must stay a checkpoint
                chainable.append((pred, port))

            if len(chainable) == 1:
                pred, port = chainable[0]
                gi = group_index[pred]
                groups[gi].append((bid, port))
                group_index[bid] = gi
            else:
                groups.append([(bid, None)])
                group_index[bid] = len(groups) - 1

        return groups

    def run_all_streaming(self) -> dict[str, str]:
        """Like run_all, but fuses whatever contiguous stretch of the
        pipeline it safely can (see _build_fusion_groups) into a single
        polars query per group, executed with a streaming collect -- the
        one path in this engine where a source larger than memory doesn't
        have to fully materialize at every block boundary. An explicit,
        opt-in action: ordinary run_all/run_to_here/single-block runs are
        completely unaffected by this method existing.

        A fusion group's interior members (everything but the group's
        last/exit block) get no independent cache entry or RunState update
        -- the report marks them "fused" rather than green/red/orange, and
        clicking them individually still shows whatever they last were.
        That's the real cost of streaming mode, in exchange for the fused
        stretch never fully materializing at every step.

        Mutually exclusive with sample mode: sample mode already makes
        interactive iteration on a huge source cheap (see read_csv's
        sample_rows), and streaming is for the opposite case -- a full,
        production-scale run."""
        if self.sample_rows is not None:
            raise ValueError("streaming runs are for full data -- turn off sample mode first")
        self._cancel_requested.clear()
        plan = self.pin()
        to_run, report = self._streaming_to_run(plan)
        for group in self._build_fusion_groups(to_run, plan):
            if self._cancel_requested.is_set():
                for bid, _ in group:
                    report.setdefault(bid, "cancelled")
                continue
            if len(group) == 1:
                bid, _ = group[0]
                self.run_block(bid, plan)
                report[bid] = self.status(bid) if bid in self.graph.blocks else "deleted during run"
                continue
            self._run_fused_group(group, plan, report)
        return report

    def refresh(self, block_id: str) -> Status:
        """Re-reads an input block's source and swaps the cached packet. A
        failed refresh flips the block red but keeps the last-known-good
        packet and its 'last successful read' timestamp intact."""
        block = self.graph.blocks[block_id]
        assert block.block_type == "input", "refresh() is only valid for input blocks"
        # Bump before pinning, so the run is planned against the new read.
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
