from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import queue as queue_mod
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

import polars as pl

from .blocks.base import BLOCK_REGISTRY, BlockSpec
from .cache import CacheStore
from .graph import BlockInstance, Graph
from .metadata_transforms import resolve_metadata_transform
from .packet import ColumnMeta, ColumnRole, DataFramePacket, find_duplicate_unique_role, resolve_role_column
from .run_worker import run_fused_group_entry, run_iteration_entry, run_worker_entry
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
        basis: dict[str, Any] = {
            "category": block.category,
            "code_version": block.code_version,
            "params": block.params,
            "column_role_overrides": block.column_role_overrides,
            "code": block.code if block.is_custom else None,
            "upstream": upstream,
            "group_by": block.group_by,
            "max_workers": block.max_workers,
        }
        # A registry block whose fn accepts `block_id` (see run_block's
        # block_id injection) has output that genuinely depends on its own
        # identity -- a stochastic block deriving its RNG from it (see
        # stochastic.seed.spawn_rng), e.g. -- not just on category/params/
        # upstream. Without this, two such blocks with otherwise-identical
        # params (and no upstream) would hash to the same cache key and
        # silently share one cache slot, each overwriting the other's
        # result. Scoped to registry blocks only (a plain dict/attribute
        # lookup, no compilation) so this never has to compile a custom
        # block's code just to answer a routine status poll.
        spec = BLOCK_REGISTRY.get(block.category)
        if spec is not None and accepts_param(spec.fn, "block_id"):
            basis["block_id"] = block_id
        return basis

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


def _is_fusable_block(block: BlockInstance) -> BlockSpec | None:
    """The registry spec for `block` if it's eligible to join a streaming
    run's fusion (see Runner._build_fusion_groups), else None. Fusable
    means: a registry block (never custom/AI-authored code -- the LLM
    contract in llm/prompts.py keeps assuming plain eager pl.DataFrame)
    with a lazy_fn (see BlockSpec.lazy_fn), no group_by (grouping needs
    materialized data to partition_by), and no role/output_dir/block_id
    auto-fill -- fusable blocks (filter/select/groupby_agg/join/read_csv
    today) only ever take plain params, and this also guards against a
    future lazy_fn'd block that actually needs one of these, which
    _run_fused_group doesn't resolve.

    A module-level function (not a Runner method) because it needs no
    instance state -- Runner._fusable_spec just delegates to it, and
    compiler.py's streaming-mode compile (see
    compiler._build_compile_fusion_groups) imports it directly, so the
    live engine and a compiled script agree on exactly what fuses."""
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
class FusionGroup:
    """One streaming run's fusion group (see Runner._build_fusion_groups):
    a connected set of fusable blocks executed as a single polars lazy
    plan in one subprocess. Not required to be a straight-line chain --
    fan-out (one block feeding several fusable consumers) and fan-in (e.g.
    a join whose both sides are fusable) both stay inside one group, as
    long as every member is fusable (see Runner._fusable_spec).

    `members` names every block computed as part of this group's plan, in
    topological order. `exits` names the subset that need an actual
    collected DataFrame and a real cache entry: a member with a
    downstream dataframe consumer outside the group, or none at all.
    Every other member is purely interior -- computed as part of the
    group's one lazy plan, but never independently cached (reported as
    "fused").

    `sinks` maps a terminal-write block id (e.g. write_csv, folded onto
    the group's tail -- see BlockSpec.lazy_sink_fn) to the group member
    whose output it writes straight to disk via the streaming engine,
    without that member ever needing a collected DataFrame. A sink
    produces no DataFrame and is never in `exits`."""

    members: list[str]
    exits: set[str]
    sinks: dict[str, str] = field(default_factory=dict)


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
    # Safety ceiling on a fan-out region's n_iterations (see
    # _run_region_iterations) -- graph fan-out is the "N ~ 10^2-10^4"
    # execution shape (stochastic-engine-proposal.md S3.5/S4); a vectorised
    # simulation block (S5) is the tool for N ~ 10^6+, not this.
    MAX_ITERATIONS = 10_000

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
        # True for the duration of a _sweep() (run_all/force_run_all) -- see
        # _sweep's own comment. False outside of one, including during a
        # single-block run (where is_running()/_active already cover it).
        self.sweep_running = False

    def _st(self, block_id: str) -> RunState:
        return self.state.setdefault(block_id, RunState())

    def is_running(self, block_id: str | None = None) -> bool:
        with self._active_lock:
            return (block_id in self._active) if block_id is not None else bool(self._active)

    def running_elapsed(self, block_id: str) -> float | None:
        """Seconds since this block's current dispatch started, or None if
        it isn't running -- a crude heartbeat for the UI while polars gives
        no finer-grained progress for a `collect(engine="streaming")` or a
        `collect_all(...)` call (see _dispatch/_dispatch_fused_group,
        which both stamp "started_at" here). Several exits of one fused
        group share the same started_at, so they report identically for as
        long as the group's single subprocess is in flight."""
        with self._active_lock:
            handle = self._active.get(block_id)
            if handle is None:
                return None
            return time.monotonic() - handle["started_at"]

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
        streaming run's fusion (see _build_fusion_groups), else None --
        see module-level _is_fusable_block, which this just delegates to
        (it needs no Runner state; compiler.py's streaming-mode compile
        reuses the exact same rule via that free function)."""
        return _is_fusable_block(block)

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
            if block.category == "collect":
                # A 'collect' block's real input isn't its wire's normal
                # cached value (that's just the fan-out region's own
                # single-preview-run output, computed like any other
                # block) -- it's the N-times-iterated, concatenated result
                # of the whole region between its paired 'iterate' block
                # and itself. See _run_region_iterations for the actual
                # fan-out execution (graph fan-out, stochastic-engine-
                # proposal.md S4); everything from here on treats that
                # packet exactly like a normal wired input.
                input_packets = {"value": self._run_region_iterations(block_id, block, plan)}
            else:
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
            # Not restricted to output blocks (unlike output_dir above): a
            # stochastic "standard" block (see blocks/stochastic.py) opts
            # into block_id too, to derive its RNG from (seed param,
            # block_id) -- see stochastic.seed.spawn_rng -- so its stream is
            # reproducible without depending on run order or worker count.
            # No existing block names a param literally "block_id" other
            # than the output ones already relying on this, so widening it
            # here changes nothing for them.
            if accepts_param(fn, "block_id"):
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
            self._active[block_id] = {"cancel_event": cancel_event, "started_at": time.monotonic()}

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
        self, exits: list[str], steps: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, str]], dict[str, pl.DataFrame]]:
        """Runs one streaming run's fused group (see _build_fusion_groups)
        to completion in its own subprocess -- same isolation rationale as
        _dispatch above (a crash or runaway allocation only takes down this
        worker), trimmed to a single task since a fused group is inherently
        one unit of work, not a pool of independent ones, so there's no
        worker-pool bookkeeping to do. `exits` is the group's exit block
        ids -- there can be more than one now that a group is a DAG rather
        than just a chain (see FusionGroup) -- each registered in
        self._active sharing the same cancel_event, so status()/
        is_running() report "running" for every exit while the group is in
        flight, and cancel() called against any one of them stops the
        whole subprocess. Returns (schemas, results) on success; raises
        RunCancelled or RuntimeError exactly like _dispatch does on
        cancellation or a worker crash/error."""
        result_queue = MP_CONTEXT.Queue()
        cancel_event = MP_CONTEXT.Event()
        task_key = "+".join(exits)
        started_at = time.monotonic()

        with self._active_lock:
            for eid in exits:
                self._active[eid] = {"cancel_event": cancel_event, "started_at": started_at}

        p = MP_CONTEXT.Process(target=run_fused_group_entry, args=(steps, result_queue, task_key), daemon=True)
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
                for eid in exits:
                    self._active.pop(eid, None)

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
            elif block.outputs[i].type == "scalar_metric" and all(isinstance(v, dict) for v in values):
                combined.append(pl.DataFrame([{group_col: gval, **v} for gval, v in zip(group_values, values)]))
            else:
                combined.append(dict(zip(group_values, values)))
        return combined[0] if len(combined) == 1 else tuple(combined)

    def _iterate_region(self, ib_id: str, cb_id: str, plan: RunPlan) -> list[str]:
        """Topologically-ordered blocks in the fan-out region bounded by an
        'iterate' block and its paired 'collect' block -- every block on
        some path from `ib_id` to `cb_id` (the intersection of `ib_id`'s
        descendants and `cb_id`'s ancestors), including `ib_id` itself but
        excluding `cb_id`, which runs once, as the reducer, not once per
        iteration."""
        graph = plan.graph
        region = graph.descendants([ib_id]) & graph.ancestors([cb_id])
        region.discard(cb_id)
        order = graph.topo_order()
        return [b for b in order if b in region]

    def _run_region_iterations(self, cb_id: str, cb_block: BlockInstance, plan: RunPlan) -> DataFramePacket:
        """The graph-fan-out engine primitive (stochastic-engine-proposal.md
        S4): runs the fan-out region between `cb_block`'s paired 'iterate'
        block and `cb_block` itself once per iteration, each in its own
        subprocess (see _dispatch_iterations), and concatenates whichever
        value 'collect' is wired from across all of them into one dataframe
        (tagged with an `iteration` column) -- the packet run_block then
        treats as this block's ordinary 'value' input.

        Cache-key note: this needs no changes to RunPlan.basis. `cb_block`'s
        key already recurses into its 'value' wire's source block's own key
        (see RunPlan.basis's `upstream` hashing), which recurses in turn
        through the whole region back to the 'iterate' block -- exactly the
        same mechanism that makes any ordinary multi-block chain's key
        depend on everything upstream of it. n_iterations/iterator/seed live
        in the 'iterate' block's own params, already part of its basis."""
        ib_id = cb_block.params.get("iterate_block")
        if not ib_id or ib_id not in plan.graph.blocks:
            raise ValueError(f"'iterate_block' param must name an existing block id (got {ib_id!r})")
        ib_block = plan.graph.blocks[ib_id]
        if ib_block.category != "iterate":
            raise ValueError(f"'iterate_block' {ib_id!r} is not an 'iterate' block")

        region = self._iterate_region(ib_id, cb_id, plan)
        if ib_id not in region:
            raise ValueError(f"'{ib_id}' does not reach 'collect' block '{cb_id}' -- check the wiring between them")

        for rid in region:
            rblock = plan.graph.blocks[rid]
            if rblock.block_type == "input":
                raise ValueError(f"iterated region contains input block '{rid}' -- input blocks can't be re-run per iteration")
            if rblock.group_by:
                raise ValueError(f"iterated region contains grouped block '{rid}' -- group_by inside a fan-out region isn't supported yet")
            if not self._is_current(rid, plan):
                self.run_block(rid, plan)
                if not self._is_current(rid, plan):
                    raise RuntimeError(f"region block '{rid}' failed to run: {self._st(rid).last_error}")

        value_wire = plan.graph.input_wires(cb_id).get("value")
        if value_wire is None:
            raise ValueError("'collect' block has no wire into its 'value' input")
        if value_wire.from_block not in region:
            raise ValueError("'collect' block's 'value' input must be wired from a block inside the iterated region")
        source_block = plan.graph.blocks[value_wire.from_block]
        source_spec = next((p for p in source_block.outputs if p.name == value_wire.from_port), None)
        if source_spec is None or source_spec.type != "dataframe":
            raise ValueError("'collect' block's 'value' input must be wired from a dataframe-typed output")
        collect_source_port_index = next(i for i, p in enumerate(source_block.outputs) if p.name == value_wire.from_port)

        steps: list[dict[str, Any]] = []
        for rid in region:
            rblock = plan.graph.blocks[rid]
            fn = rblock.resolved_fn()
            kwargs: dict[str, Any] = dict(rblock.params)
            chain_inputs: dict[str, tuple[str, int]] = {}
            input_schema_metas: list[dict] = []
            for port, wire in plan.graph.input_wires(rid).items():
                entry = self.cache.get(plan.key(wire.from_block))
                if entry is None:
                    raise RuntimeError(f"missing cached output for '{wire.from_block}'")
                value = entry.outputs[wire.from_port]
                if isinstance(value, DataFramePacket):
                    input_schema_metas.append(value.schema_meta)
                if wire.from_block in region:
                    src_outputs = plan.graph.blocks[wire.from_block].outputs
                    port_index = next(i for i, p in enumerate(src_outputs) if p.name == wire.from_port)
                    chain_inputs[port] = (wire.from_block, port_index)
                else:
                    # Fixed across every iteration: a normal, already-green
                    # value from outside the region, read once here exactly
                    # like _gather_inputs does for an ordinary block.
                    kwargs[port] = value.data if isinstance(value, DataFramePacket) else value
            for role in ROLE_PARAM_NAMES:
                role_param = find_role_param(fn, role)
                if role_param is not None and role_param not in kwargs:
                    resolved = resolve_role_column(input_schema_metas, role)
                    if resolved is not None:
                        kwargs[role_param] = resolved
            if accepts_param(fn, "block_id"):
                kwargs["block_id"] = rid
            if rblock.block_type == "output" and accepts_param(fn, "output_dir"):
                kwargs["output_dir"] = self.output_dir
            steps.append(
                {
                    "id": rid,
                    "category": rblock.category,
                    "is_custom": rblock.is_custom,
                    "code": rblock.code,
                    "kwargs": kwargs,
                    "chain_inputs": chain_inputs,
                    "n_outputs": len(rblock.outputs),
                    "wants_iteration_index": accepts_param(fn, "_iteration_index"),
                }
            )

        n_iterations = int(ib_block.params.get("n_iterations", 100))
        if n_iterations < 1:
            raise ValueError("'n_iterations' must be >= 1")
        if n_iterations > self.MAX_ITERATIONS:
            raise ValueError(f"n_iterations={n_iterations} is over the {self.MAX_ITERATIONS}-iteration safety limit")
        max_workers = max(1, min(cb_block.max_workers or self.MAX_GROUP_WORKERS, self.MAX_GROUP_WORKERS, n_iterations))

        results = self._dispatch_iterations(cb_id, steps, value_wire.from_block, collect_source_port_index, n_iterations, max_workers)

        frames = []
        for i in range(n_iterations):
            frame = results[i]
            if not isinstance(frame, pl.DataFrame):
                raise RuntimeError(f"iteration {i}'s collected value is not a dataframe (got {type(frame).__name__})")
            frames.append(frame.with_columns(pl.lit(i).alias("iteration")))
        combined = pl.concat(frames, how="diagonal_relaxed")
        schema_meta = {name: ColumnMeta(dtype=str(combined.schema[name])) for name in combined.columns}
        return DataFramePacket(data=combined, schema_meta=schema_meta).with_lineage(cb_id)

    def _dispatch_iterations(
        self,
        block_id: str,
        steps: list[dict[str, Any]],
        collect_source_id: str,
        collect_source_port_index: int,
        n_iterations: int,
        max_workers: int,
    ) -> dict[int, Any]:
        """Runs `steps` (the fan-out region -- see _run_region_iterations)
        once per iteration index, each in its own subprocess via
        run_iteration_entry, up to `max_workers` at a time -- same
        isolation, cancellation, and crash-handling contract as _dispatch,
        just retargeted at a whole region instead of a single block call."""
        result_queue = MP_CONTEXT.Queue()
        cancel_event = MP_CONTEXT.Event()
        pending = list(range(n_iterations))
        running: dict[int, Any] = {}
        results: dict[int, Any] = {}
        errors: dict[int, str] = {}

        with self._active_lock:
            self._active[block_id] = {"cancel_event": cancel_event, "started_at": time.monotonic()}

        def _start_next() -> None:
            while pending and len(running) < max(1, max_workers):
                i = pending.pop(0)
                p = MP_CONTEXT.Process(
                    target=run_iteration_entry,
                    args=(steps, i, collect_source_id, collect_source_port_index, result_queue, i),
                    daemon=True,
                )
                p.start()
                running[i] = p

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
            raise RuntimeError(f"{first_err} ({len(errors)}/{n_iterations} iteration(s) failed)")
        return results

    def _run_fused_group(self, group: FusionGroup, plan: RunPlan, report: dict[str, str]) -> None:
        """Runs one fusion group (see _build_fusion_groups) end to end and
        writes `report` in place -- the streaming-run counterpart of
        run_block for a DAG of 2+ blocks fused into a single polars query.
        Every exit in the group -- there can be more than one now that
        fan-out/fan-in no longer force a checkpoint, see FusionGroup --
        gets a real cache entry/RunState update; every other member is
        marked "fused" and left otherwise untouched. A sink member (e.g.
        write_csv folded onto the group's tail -- see
        BlockSpec.lazy_sink_fn) gets a RunState update too, but no cache
        entry, since it produces no DataFrame -- exactly like write_csv's
        ordinary (non-streaming) run."""
        member_set = set(group.members)
        steps: list[dict[str, Any]] = []
        input_metas_per_step: dict[str, dict[str, dict[str, ColumnMeta]]] = {}
        chain_inputs_per_step: dict[str, dict[str, str]] = {}

        for bid in group.members:
            block = plan.graph.blocks[bid]
            is_sink = bid in group.sinks
            if not is_sink:
                assert self._fusable_spec(block) is not None, f"non-fusable block {bid!r} in a fusion group"
            kwargs: dict[str, Any] = dict(block.params)
            if is_sink:
                # Same output_dir/block_id auto-fill run_block does for an
                # ordinary output block (see accepts_param) -- a sink step
                # never goes through run_block, so it has to happen here.
                sink_spec = BLOCK_REGISTRY[block.category]
                if block.block_type == "output" and accepts_param(sink_spec.lazy_sink_fn, "output_dir"):
                    kwargs["output_dir"] = self.output_dir
                if block.block_type == "output" and accepts_param(sink_spec.lazy_sink_fn, "block_id"):
                    kwargs["block_id"] = bid
            chain_inputs: dict[str, str] = {}
            input_metas: dict[str, dict[str, ColumnMeta]] = {}
            for port, wire in plan.graph.input_wires(bid).items():
                if wire.from_block in member_set:
                    chain_inputs[port] = wire.from_block
                    continue
                df, meta = self._real_df_and_meta(wire, plan)
                kwargs[port] = df
                input_metas[port] = meta
            steps.append(
                {
                    "id": bid,
                    "category": block.category,
                    "kwargs": kwargs,
                    "chain_inputs": chain_inputs,
                    "is_exit": bid in group.exits,
                    "is_sink": is_sink,
                }
            )
            input_metas_per_step[bid] = input_metas
            chain_inputs_per_step[bid] = chain_inputs

        def _fail_group(error: str) -> None:
            now = _now()
            for eid in group.exits:
                st = self._st(eid)
                st.last_attempt_key = plan.key(eid)
                st.last_attempt_at = now
                st.failed = True
                st.last_error = error
                report[eid] = "red"
            for bid in group.members:
                if bid not in group.exits:
                    report[bid] = "fused (group failed)"

        try:
            schemas, results = self._dispatch_fused_group(sorted(group.exits), steps)
        except Exception as e:  # noqa: BLE001 -- captured as block state, same contract as run_block
            _fail_group(f"{type(e).__name__}: {e}")
            return

        # Chain each member's own metadata_transform forward, in the
        # group's topological order, using the schemas the worker
        # returned -- collect_schema() never touched data, so this
        # reconstructs full role/dtype tracking (including
        # column_role_overrides and the duplicate-unique-role check)
        # through the whole fused DAG without having materialized any
        # interior result, not just a bare dtype fallback at the exits. A
        # sink member has no output schema of its own and is skipped here.
        out_meta_by_id: dict[str, dict[str, ColumnMeta]] = {}
        for bid in group.members:
            if bid in group.sinks:
                continue
            block = plan.graph.blocks[bid]
            input_metas = dict(input_metas_per_step[bid])
            for port, src_id in chain_inputs_per_step[bid].items():
                input_metas[port] = out_meta_by_id[src_id]
            spec = BLOCK_REGISTRY[block.category]
            metas = spec.metadata_transform(input_metas, {"out": _SchemaView(schemas[bid])}, block.params)
            out_meta = dict(metas.get("out", {}))
            if block.column_role_overrides:
                for name, role_value in block.column_role_overrides.items():
                    if name in out_meta:
                        out_meta[name] = replace(out_meta[name], role=ColumnRole(role_value))
            dup = find_duplicate_unique_role(out_meta)
            if dup:
                role, cols = dup
                _fail_group(f"role '{role.value}' is set on more than one column: {', '.join(cols)}")
                return
            out_meta_by_id[bid] = out_meta
            if bid not in group.exits:
                report[bid] = "fused"

        now = _now()
        for sink_id in group.sinks:
            key = plan.key(sink_id)
            self.cache.set(key, {})
            st = self._st(sink_id)
            st.last_attempt_key = key
            st.last_attempt_at = now
            st.last_successful_key = key
            st.last_successful_basis = plan.basis(sink_id)
            st.failed = False
            st.last_error = None
            report[sink_id] = self.status(sink_id) if sink_id in self.graph.blocks else "deleted during run"

        for eid in group.exits:
            key = plan.key(eid)
            packet = DataFramePacket(data=results[eid], schema_meta=out_meta_by_id[eid]).with_lineage(eid)
            self.cache.set(key, {"out": packet})
            st = self._st(eid)
            st.last_attempt_key = key
            st.last_attempt_at = now
            st.last_successful_key = key
            st.last_successful_basis = plan.basis(eid)
            st.failed = False
            st.last_error = None
            report[eid] = self.status(eid) if eid in self.graph.blocks else "deleted during run"

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
        # Read by api.py's _graph_out() (as "sweep_running") while a sweep is
        # in flight: individual blocks each go "running" only for their own
        # dispatch (see _active/is_running), so there's a real gap between
        # one block finishing and the next one starting where nothing in the
        # graph looks like it's mid-run. Keeps the frontend polling through
        # those gaps instead of stopping early, so every block's grey ->
        # running -> green/red transition is visible as the sweep goes,
        # rather than the canvas jumping straight from all-grey to done.
        self.sweep_running = True
        try:
            for bid in plan.graph.topo_order():
                if self._cancel_requested.is_set():
                    report[bid] = "cancelled"
                    continue
                block = plan.graph.blocks[bid]
                if block.block_type == "input":
                    # A source that's *never* been read blocks every downstream
                    # block anyway (see _blocked_on_unread_input) -- reporting
                    # the whole rest of the pipeline as permanently "blocked"
                    # instead of just reading it isn't useful to anyone, so the
                    # first read happens here automatically. A source that has
                    # been read before but is merely stale (an edited param,
                    # sample mode toggled, ...) is left alone: re-reading it is
                    # a deliberate, sometimes expensive action (Refresh sources),
                    # not something a plain Run all should decide to do on its
                    # own once data has already been pulled in once.
                    if self._st(bid).last_successful_key is None:
                        self.run_block(bid, plan)
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
        finally:
            self.sweep_running = False

    def run_all(self) -> dict[str, str]:
        """Topological order; skips blocks already current under this run's
        pinned graph. Input blocks are read here too, but only the first
        time -- an input block that's already been read at least once stays
        untouched by Run all, however stale; re-reading it is what Refresh
        sources is for (see refresh()/refresh_all())."""
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

    def _build_fusion_groups(self, to_run: list[str], plan: RunPlan) -> list[FusionGroup]:
        """Partitions `to_run` (topo-ordered) into fusion groups (see
        FusionGroup): each maximal connected component of fusable blocks
        (following dataframe wires in either direction) becomes one
        group, executed as a single polars lazy plan -- fan-out (one
        block feeding several fusable consumers) and fan-in (e.g. a join
        whose both sides are fusable) both stay inside one group, rather
        than forcing a checkpoint the way a chain-only algorithm would. A
        non-fusable block is always its own singleton group and can never
        be absorbed into another one.

        Within a group, a member is an "exit" -- gets a real collected
        DataFrame and its own cache entry -- iff it has a downstream
        dataframe consumer outside the group, or no downstream dataframe
        consumer at all (a leaf the user actually wants materialized).
        Every other member is purely interior, reported as "fused".

        A trailing sink-capable block (write_csv today -- see
        BlockSpec.lazy_sink_fn) is folded onto a group as a terminal
        write, replacing what would otherwise be its predecessor's
        collected exit, whenever that predecessor's *only* dataframe
        consumer anywhere in the graph is this one sink -- see the second
        pass below."""
        to_run_set = set(to_run)
        fusable: dict[str, BlockSpec] = {}
        for bid in to_run:
            spec = self._fusable_spec(plan.graph.blocks[bid])
            if spec is not None:
                fusable[bid] = spec

        parent: dict[str, str] = {bid: bid for bid in fusable}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        port_types_cache: dict[str, dict[str, str]] = {}

        def port_types(bid: str) -> dict[str, str]:
            if bid not in port_types_cache:
                port_types_cache[bid] = {p.name: p.type for p in plan.graph.blocks[bid].inputs}
            return port_types_cache[bid]

        for bid in fusable:
            for port, wire in plan.graph.input_wires(bid).items():
                if port_types(bid).get(port) != "dataframe":
                    continue
                if wire.from_block in fusable:
                    union(bid, wire.from_block)

        groups_by_root: dict[str, list[str]] = {}
        for bid in to_run:  # preserves the caller's topological order
            key = find(bid) if bid in fusable else bid
            groups_by_root.setdefault(key, []).append(bid)

        groups: list[FusionGroup] = []
        group_index: dict[str, int] = {}
        for members in groups_by_root.values():
            member_set = set(members)
            if len(members) == 1 and members[0] not in fusable:
                groups.append(FusionGroup(members=list(members), exits=set(member_set)))
            else:
                exits: set[str] = set()
                for bid in members:
                    out_ports = {p.name for p in plan.graph.blocks[bid].outputs if p.type == "dataframe"}
                    df_consumers = [
                        w
                        for w in plan.graph.wires.values()
                        if w.from_block == bid
                        and w.from_port in out_ports
                        and w.to_block in plan.graph.blocks
                        and port_types(w.to_block).get(w.to_port) == "dataframe"
                    ]
                    if not df_consumers or any(w.to_block not in member_set for w in df_consumers):
                        exits.add(bid)
                groups.append(FusionGroup(members=list(members), exits=exits))
            for bid in members:
                group_index[bid] = len(groups) - 1

        # Sink attachment: fold a to_run sink-capable block into its sole
        # fusable predecessor's group when that predecessor's only
        # dataframe consumer anywhere in the graph is this one sink. A
        # sink block is never itself fusable, so it always starts out as
        # its own singleton group -- that's the only shape this ever folds
        # away.
        for group in groups:
            for eid in list(group.exits):
                block = plan.graph.blocks[eid]
                out_ports = {p.name for p in block.outputs if p.type == "dataframe"}
                all_wires = [w for w in plan.graph.wires.values() if w.from_block == eid and w.from_port in out_ports]
                if len(all_wires) != 1:
                    continue
                sink_id = all_wires[0].to_block
                if sink_id not in to_run_set:
                    continue
                sink_idx = group_index.get(sink_id)
                if sink_idx is None or groups[sink_idx] is group or len(groups[sink_idx].members) != 1:
                    continue
                sink_block = plan.graph.blocks[sink_id]
                if sink_block.is_custom or sink_block.group_by:
                    continue
                sink_spec = BLOCK_REGISTRY.get(sink_block.category)
                if sink_spec is None or sink_spec.lazy_sink_fn is None:
                    continue
                if len(plan.graph.input_wires(sink_id)) != 1:
                    continue
                groups[sink_idx].members = []  # folded into `group`; dropped below
                group.exits.discard(eid)
                group.members.append(sink_id)
                group.sinks[sink_id] = eid

        return [g for g in groups if g.members]

    def run_all_streaming(self) -> dict[str, str]:
        """Like run_all, but fuses whatever connected stretch of the
        pipeline it safely can (see _build_fusion_groups) into a single
        polars query per group, executed with a streaming collect -- the
        one path in this engine where a source larger than memory doesn't
        have to fully materialize at every block boundary. Fan-out (one
        block feeding several fusable consumers) and fan-in (e.g. a join
        whose both sides are fusable) both stay fused rather than forcing
        a checkpoint; a sink-capable terminal block (write_csv) fused onto
        a group's tail skips the collect entirely, streaming straight to
        disk. An explicit, opt-in action: ordinary run_all/run_to_here/
        single-block runs are completely unaffected by this method
        existing.

        A fusion group's interior members (everything but its exits) get
        no independent cache entry or RunState update -- the report marks
        them "fused" rather than green/red/orange, and clicking them
        individually still shows whatever they last were. That's the real
        cost of streaming mode, in exchange for the fused stretch never
        fully materializing at every step.

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
                for bid in group.members:
                    report.setdefault(bid, "cancelled")
                continue
            if len(group.members) == 1:
                bid = group.members[0]
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
