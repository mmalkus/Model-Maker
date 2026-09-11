from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .blocks.base import BLOCK_REGISTRY, PortSpec
from .cache import CacheStore
from .graph import BlockInstance, Graph, Lane, Position, Wire
from .packet import ColumnRole, DataFramePacket, find_duplicate_unique_role
from .project import load_project, save_project
from .runner import Runner


def new_id(prefix: str = "b") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def wire_is_valid(graph: Graph, wire: Wire) -> bool:
    from_block = graph.blocks.get(wire.from_block)
    to_block = graph.blocks.get(wire.to_block)
    if from_block is None or to_block is None:
        return False
    from_spec = next((p for p in from_block.outputs if p.name == wire.from_port), None)
    to_spec = next((p for p in to_block.inputs if p.name == wire.to_port), None)
    if from_spec is None or to_spec is None:
        return False
    # "any" (the generic View value block's ports) matches every other type,
    # in either direction.
    return from_spec.type == to_spec.type or from_spec.type == "any" or to_spec.type == "any"


class ProjectSession:
    """In-memory holder for the currently-open graph, its runner, and where
    it was last loaded from/saved to. One session per running server process
    -- there's no multi-project or multi-user concept yet."""

    def __init__(self) -> None:
        self.graph = Graph()
        self.runner = Runner(self.graph, CacheStore(Path(".modelmaker-cache")))
        self.project_path: Path | None = None
        self.project_name = "untitled"

    def load(self, path: Path) -> None:
        self.graph = load_project(path)
        self.runner = Runner(self.graph, CacheStore(Path(".modelmaker-cache")))
        self.project_path = path
        self.project_name = path.stem

    def save(self, path: Path | None = None) -> Path:
        target = path or self.project_path
        if target is None:
            raise ValueError("no project path set; provide one to save")
        save_project(self.graph, target, project_name=self.project_name)
        self.project_path = target
        return target

    def add_block(
        self,
        category: str,
        block_type: str | None = None,
        name: str | None = None,
        lane: str | None = None,
        x: float = 0,
        y: float = 0,
        params: dict[str, Any] | None = None,
        code: str | None = None,
        inputs: list[dict] | None = None,
        outputs: list[dict] | None = None,
        metadata_transform: dict[str, Any] | None = None,
    ) -> BlockInstance:
        spec = BLOCK_REGISTRY.get(category)
        is_custom = spec is None
        if spec is not None:
            resolved_type = block_type or spec.block_type
            resolved_inputs = [PortSpec(**p) for p in inputs] if inputs else list(spec.inputs)
            resolved_outputs = [PortSpec(**p) for p in outputs] if outputs else list(spec.outputs)
        else:
            if block_type is None:
                raise ValueError(f"unknown category '{category}'; block_type is required for custom blocks")
            resolved_type = block_type
            resolved_inputs = [PortSpec(**p) for p in (inputs or [])]
            resolved_outputs = [PortSpec(**p) for p in (outputs or [])]

        bid = new_id()
        block = BlockInstance(
            id=bid,
            block_type=resolved_type,
            category=category,
            name=name or category,
            lane=lane,
            position=Position(x=x, y=y),
            code_version=1,
            params=params or {},
            inputs=resolved_inputs,
            outputs=resolved_outputs,
            code=code,
            metadata_transform=metadata_transform,
            is_custom=is_custom,
        )
        self.graph.blocks[bid] = block
        return block

    def update_block(self, block_id: str, **fields: Any) -> BlockInstance:
        block = self.graph.blocks[block_id]
        if "name" in fields and fields["name"] is not None:
            block.name = fields["name"]
        if "lane" in fields:
            block.lane = fields["lane"]
        if "position" in fields and fields["position"] is not None:
            block.position = Position(**fields["position"])
        if "params" in fields and fields["params"] is not None:
            block.params = fields["params"]
        if "group_by" in fields:
            block.group_by = fields["group_by"]
        if "max_workers" in fields:
            block.max_workers = fields["max_workers"]
        if "code" in fields and fields["code"] is not None and fields["code"] != block.code:
            block.code = fields["code"]
            block.code_version += 1
        if "metadata_transform" in fields and fields["metadata_transform"] is not None:
            block.metadata_transform = fields["metadata_transform"]
        return block

    def set_column_role(self, block_id: str, column: str, role: str) -> BlockInstance:
        """Hand-tag one column's role on this block (see
        BlockInstance.column_role_overrides). role="unassigned" clears a
        previous tag. Rejects a role already sitting on a different column
        of this block's own last-known output where that's checkable (the
        block has been run at least once) -- the immediate half of the
        uniqueness rule; the other half (two upstream branches colliding
        once merged, e.g. by a join) can only be caught once that merge
        actually runs, in Runner.run_block."""
        block = self.graph.blocks[block_id]
        try:
            role_enum = ColumnRole(role)
        except ValueError as e:
            raise ValueError(f"unknown role: {role!r}") from e
        if role_enum == ColumnRole.PREDICTED:
            raise ValueError("'predicted' is assigned automatically by modelling blocks, not settable directly")

        overrides = dict(block.column_role_overrides)
        if role_enum == ColumnRole.UNASSIGNED:
            overrides.pop(column, None)
        else:
            overrides[column] = role_enum.value

        st = self.runner.state.get(block_id)
        if st and st.last_successful_key:
            entry = self.runner.cache.get(st.last_successful_key)
            if entry:
                for value in entry.outputs.values():
                    if not isinstance(value, DataFramePacket):
                        continue
                    prospective = dict(value.schema_meta)
                    for name, override_role in overrides.items():
                        if name in prospective:
                            prospective[name] = replace(prospective[name], role=ColumnRole(override_role))
                    dup = find_duplicate_unique_role(prospective)
                    if dup:
                        dup_role, cols = dup
                        others = [c for c in cols if c != column] or cols
                        raise ValueError(
                            f"role '{dup_role.value}' can only be on one column here -- already set on {others[0]!r}"
                        )

        block.column_role_overrides = overrides
        return block

    def delete_block(self, block_id: str) -> None:
        self.graph.blocks.pop(block_id, None)
        dead_wires = [wid for wid, w in self.graph.wires.items() if w.from_block == block_id or w.to_block == block_id]
        for wid in dead_wires:
            del self.graph.wires[wid]
        self.runner.state.pop(block_id, None)

    def add_wire(self, from_block: str, from_port: str, to_block: str, to_port: str) -> Wire:
        if self.graph.creates_cycle(from_block, to_block):
            raise ValueError("connecting these blocks would create a cycle")
        # a target port accepts at most one incoming wire
        for wid, w in list(self.graph.wires.items()):
            if w.to_block == to_block and w.to_port == to_port:
                del self.graph.wires[wid]
        wid = new_id("w")
        wire = Wire(id=wid, from_block=from_block, from_port=from_port, to_block=to_block, to_port=to_port)
        self.graph.wires[wid] = wire
        return wire

    def delete_wire(self, wire_id: str) -> None:
        self.graph.wires.pop(wire_id, None)

    def rename_port(self, block_id: str, port: str, name: str | None) -> BlockInstance:
        block = self.graph.blocks[block_id]
        if not any(p.name == port for p in block.outputs):
            raise ValueError(f"no such output port: {port}")
        port_names = dict(block.port_names)
        if name:
            port_names[port] = name
        else:
            port_names.pop(port, None)
        block.port_names = port_names
        return block

    def set_lane(self, lane_id: str, name: str, order: int, height: float | None = None) -> None:
        existing = self.graph.lanes.get(lane_id)
        resolved_height = height if height is not None else (existing.height if existing else 260.0)
        self.graph.lanes[lane_id] = Lane(name=name, order=order, height=resolved_height)

    def delete_lane(self, lane_id: str) -> None:
        self.graph.lanes.pop(lane_id, None)
        for b in self.graph.blocks.values():
            if b.lane == lane_id:
                b.lane = None
