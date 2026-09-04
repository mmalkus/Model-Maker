from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .blocks.base import BLOCK_REGISTRY, BlockType, PortSpec


@dataclass
class Position:
    x: float = 0
    y: float = 0


@dataclass
class Lane:
    name: str
    order: int
    # Pixel height of this lane's band on the canvas -- user-resizable by
    # dragging its top/bottom border (see LaneBand.tsx). Lanes stack
    # end-to-end by `order` with no gap, so lanes can never overlap by
    # construction; resizing one only shifts the ones after it.
    height: float = 260.0


@dataclass
class Wire:
    id: str
    from_block: str
    from_port: str
    to_block: str
    to_port: str


@dataclass
class BlockInstance:
    id: str
    block_type: BlockType
    category: str
    name: str
    lane: str | None
    position: Position
    code_version: int
    params: dict[str, Any]
    inputs: list[PortSpec]
    outputs: list[PortSpec]
    code_ref: str | None = None
    code: str | None = None
    # dict declaration for custom blocks (see metadata_transforms.resolve);
    # None for registry blocks, whose transform lives in the registry.
    metadata_transform: dict[str, Any] | None = None
    # True for an AI-authored block with its own editable code/metadata_transform;
    # False for a block backed by a fixed BLOCK_REGISTRY entry. Independent of
    # block_type -- a custom block can be an input, standard, or output block.
    is_custom: bool = False
    # column name -> ColumnRole value (see packet.py), hand-tagged by the
    # user (see session.set_column_role) and layered on top of whatever role
    # this block's own metadata_transform already computed -- applies to
    # that column on every output port it appears on. Included in
    # Runner.compute_key's hash basis alongside params/code, so retagging a
    # column invalidates this block and everything downstream the same way
    # any other param change does; no separate staleness bookkeeping needed.
    column_role_overrides: dict[str, str] = field(default_factory=dict)
    # output port name -> user-given name for the data on that port (e.g.
    # "raw applications"), settable by clicking that port's data in the UI
    # (see PortInspector.tsx). Purely descriptive except in one place: the
    # compiler uses it, when set, as the compiled variable name that holds
    # this port's data instead of the auto-generated block-name/id scheme
    # (see compiler.compile_graph) -- so naming your data also names it in
    # the generated script.
    port_names: dict[str, str] = field(default_factory=dict)

    def resolved_fn(self):
        if self.is_custom:
            return _compile_code_to_fn(self.code or "")
        return BLOCK_REGISTRY[self.category].fn

    def resolved_metadata_transform(self):
        if self.is_custom:
            return self.metadata_transform
        return BLOCK_REGISTRY[self.category].metadata_transform


def _compile_code_to_fn(code: str):
    # `pl` is available here to match the compiled script's module-level
    # `import polars as pl` (see compiler.py) -- custom block code should be
    # able to rely on it being in scope either way.
    ns: dict[str, Any] = {"pl": pl}
    exec(code, ns)  # trusted, project-local code; matches the plan's no-sandboxing execution model
    fns = [v for v in ns.values() if isinstance(v, types.FunctionType)]
    if len(fns) != 1:
        raise ValueError("custom block code must define exactly one top-level function")
    return fns[0]


@dataclass
class Graph:
    lanes: dict[str, Lane] = field(default_factory=dict)
    blocks: dict[str, BlockInstance] = field(default_factory=dict)
    wires: dict[str, Wire] = field(default_factory=dict)

    def input_wires(self, block_id: str) -> dict[str, Wire]:
        return {w.to_port: w for w in self.wires.values() if w.to_block == block_id}

    def predecessors(self, block_id: str) -> set[str]:
        return {w.from_block for w in self.wires.values() if w.to_block == block_id}

    def successors(self, block_id: str) -> set[str]:
        return {w.to_block for w in self.wires.values() if w.from_block == block_id}

    def ancestors(self, block_ids: list[str]) -> set[str]:
        seen: set[str] = set()
        stack = list(block_ids)
        while stack:
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            stack.extend(self.predecessors(b))
        return seen

    def creates_cycle(self, from_block: str, to_block: str) -> bool:
        """Would a wire from_block -> to_block create a cycle? True if
        to_block is already an ancestor of from_block (including
        from_block == to_block), i.e. a path to_block -> ... -> from_block
        already exists, so adding from_block -> to_block would close a loop.
        Checked before a wire is added -- see session.add_wire -- so the
        graph a user can build is acyclic by construction and topo_order's
        cycle branch is unreachable in normal use."""
        return to_block in self.ancestors([from_block])

    def topo_order(self) -> list[str]:
        """Deterministic topological sort: ties broken by (lane order, canvas
        position, block id) per plan section 7 — lanes play no role in
        execution order itself, only in tie-breaking for stable output.
        Blocks are ordered by y before x since flow runs top-to-bottom, to
        match visual reading order for otherwise-unordered blocks."""
        visited: set[str] = set()
        order: list[str] = []

        def sort_key(bid: str):
            b = self.blocks[bid]
            lane_order = self.lanes[b.lane].order if b.lane in self.lanes else 0
            return (lane_order, b.position.y, b.position.x, bid)

        def visit(bid: str, stack: frozenset[str]):
            if bid in visited:
                return
            if bid in stack:
                raise ValueError(f"cycle detected at block {bid}")
            for pred in sorted(self.predecessors(bid), key=sort_key):
                visit(pred, stack | {bid})
            visited.add(bid)
            order.append(bid)

        for bid in sorted(self.blocks, key=sort_key):
            visit(bid, frozenset())
        return order
