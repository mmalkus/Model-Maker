from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import polars as pl

from .blocks.base import BLOCK_REGISTRY
from .cache import CacheStore
from .graph import Graph
from .metadata_transforms import resolve_metadata_transform
from .packet import DataFramePacket
from .util import accepts_param

Status = Literal["grey", "green", "orange", "red"]


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class Runner:
    def __init__(self, graph: Graph, cache: CacheStore | None = None, output_dir: str = "./output"):
        self.graph = graph
        self.cache = cache or CacheStore()
        self.output_dir = output_dir
        self.state: dict[str, RunState] = {}

    def _st(self, block_id: str) -> RunState:
        return self.state.setdefault(block_id, RunState())

    def compute_key(self, block_id: str, _stack: frozenset[str] = frozenset()) -> str:
        """Lineage-based cache key: hashes block identity/config plus the
        upstream blocks' *keys*, never the underlying DataFrame content, so
        computing it is cheap no matter how large the data is.

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
            basis = {
                "category": block.category,
                "code_version": block.code_version,
                "params": block.params,
                "read_counter": self._st(block_id).read_counter,
            }
            return _hash(basis)
        next_stack = _stack | {block_id}
        upstream = {
            port: f"{self.compute_key(wire.from_block, next_stack)}:{wire.from_port}"
            for port, wire in sorted(self.graph.input_wires(block_id).items())
        }
        basis = {
            "category": block.category,
            "code_version": block.code_version,
            "params": block.params,
            "code": block.code if block.is_custom else None,
            "upstream": upstream,
        }
        return _hash(basis)

    def status(self, block_id: str) -> Status:
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
        key = self.compute_key(block_id)
        st.last_attempt_key = key
        st.last_attempt_at = _now()
        try:
            input_packets = self._gather_inputs(block_id)
            plain_inputs = {
                port: (p.data if isinstance(p, DataFramePacket) else p) for port, p in input_packets.items()
            }
            fn = block.resolved_fn()
            call_kwargs = dict(plain_inputs, **block.params)
            if block.block_type == "output" and accepts_param(fn, "output_dir"):
                call_kwargs["output_dir"] = self.output_dir
            if block.block_type == "output" and accepts_param(fn, "block_id"):
                call_kwargs["block_id"] = block_id
            raw = fn(**call_kwargs)

            out_names = [p.name for p in block.outputs]
            if len(out_names) == 0:
                raw_outputs: dict[str, Any] = {}
            elif len(out_names) == 1:
                raw_outputs = {out_names[0]: raw}
            else:
                raw_outputs = dict(zip(out_names, raw))

            data_outputs = {k: v for k, v in raw_outputs.items() if isinstance(v, pl.DataFrame)}
            transform = block.resolved_metadata_transform()
            metas: dict[str, dict] = {}
            if transform is not None and data_outputs:
                metas = resolve_metadata_transform(transform)(
                    {port: p.schema_meta for port, p in input_packets.items() if isinstance(p, DataFramePacket)},
                    data_outputs,
                    block.params,
                )

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
            st.failed = False
            st.last_error = None
            if block.block_type == "input":
                st.last_successful_read_at = st.last_attempt_at
        except Exception as e:  # noqa: BLE001 -- captured as block state, not propagated
            st.failed = True
            st.last_error = f"{type(e).__name__}: {e}"
        return self.status(block_id)

    def _ordered_ancestors(self, block_id: str) -> list[str]:
        anc = self.graph.ancestors([block_id])
        anc.discard(block_id)
        order = self.graph.topo_order()
        return [b for b in order if b in anc]

    def run_to_here(self, block_id: str) -> Status:
        """Cascades: runs whatever upstream chain isn't green, then this block."""
        for pred in self._ordered_ancestors(block_id):
            if self.graph.blocks[pred].block_type == "input":
                continue
            if self.status(pred) != "green":
                self.run_block(pred)
        return self.run_block(block_id)

    def _blocked_on_ungread_input(self, block_id: str) -> bool:
        preds = self.graph.ancestors([block_id]) - {block_id}
        return any(
            self.graph.blocks[p].block_type == "input" and self.status(p) == "grey" for p in preds
        )

    def run_all(self) -> dict[str, str]:
        """Topological order; skips already-green blocks; input blocks are
        never (re)run here — see refresh()/refresh_all()."""
        report: dict[str, str] = {}
        for bid in self.graph.topo_order():
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
        report: dict[str, str] = {}
        for bid in self.graph.topo_order():
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
