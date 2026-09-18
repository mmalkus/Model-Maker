from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .blocks.base import BLOCK_REGISTRY, PortSpec
from .cache import CacheStore
from .graph import BlockInstance, Graph, Lane, Position, Wire
from .packet import ColumnRole, DataFramePacket, find_duplicate_unique_role
from .project import graph_from_dict, graph_to_dict, load_project, save_project
from .runner import RunState, Runner

CACHE_DIR = Path(".modelmaker-cache")
# Where the crash-recovery snapshot lives. Inside the (gitignored) cache
# directory deliberately: it is regenerable working state, not part of the
# versioned project.
RECOVERY_PATH = CACHE_DIR / "recovery.json"
# How many edits back Undo reaches.
UNDO_LIMIT = 50


def new_id(prefix: str = "b") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Snapshot:
    """One undo point: the graph as data, plus the run state that went with
    it -- so undoing a delete brings the block back still green, rather than
    resurrecting it as never-run."""

    graph: dict[str, Any]
    state: dict[str, RunState] = field(default_factory=dict)


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

    def __init__(self, recovery_path: Path | None = RECOVERY_PATH) -> None:
        self.graph = Graph()
        self.runner = Runner(self.graph, CacheStore(CACHE_DIR))
        self.project_path: Path | None = None
        self.project_name = "untitled"
        self.recovery_path = recovery_path
        # Bumped by every successful edit; `dirty` is simply "has anything
        # changed since the revision we last wrote to the project file".
        self.revision = 0
        self.saved_revision = 0
        self._undo: list[Snapshot] = []
        self._redo: list[Snapshot] = []

    # ---- edit bookkeeping ---------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.revision != self.saved_revision

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _snapshot(self) -> Snapshot:
        return Snapshot(graph=graph_to_dict(self.graph, self.project_name), state=copy.deepcopy(self.runner.state))

    def _restore(self, snap: Snapshot) -> None:
        # Swap the graph *into* the existing runner rather than building a
        # new one: the runner owns the cache, and every cached output is
        # still valid (keys are derived from block config, not from runner
        # identity), so an undo should never cost a re-run.
        self.graph = graph_from_dict(snap.graph)
        self.runner.graph = self.graph
        # Run state is merged, never rolled back wholesale. Undo is about the
        # graph; what has been *run* is not part of the edit, and restoring
        # old statuses would throw away results produced since -- including
        # results from a run still in flight when the edit was made. Putting
        # the config back is enough on its own, because a status is derived
        # from the cache key: revert the params and the old key (and its
        # green status) comes back by itself. The one thing that can't
        # re-derive is state for a block the undo brings back from the dead,
        # since deleting it dropped its entry -- so fill only those in.
        for block_id, run_state in snap.state.items():
            if block_id in self.graph.blocks and block_id not in self.runner.state:
                self.runner.state[block_id] = copy.deepcopy(run_state)

    @contextmanager
    def edit(self) -> Iterator[None]:
        """Wraps one graph mutation: records an undo point and bumps the
        revision, but only if the mutation actually succeeds -- a rejected
        edit (a wire that would create a cycle, a duplicate role) must not
        leave a no-op entry on the undo stack for the user to step through."""
        before = self._snapshot()
        yield
        self._undo.append(before)
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self.revision += 1
        self._write_recovery()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self.revision += 1
        self._write_recovery()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self.revision += 1
        self._write_recovery()
        return True

    # ---- persistence ---------------------------------------------------

    def _write_recovery(self) -> None:
        """Snapshot the graph to the recovery file after every edit.

        Deliberately never writes the user's own project file: an editor
        that silently rewrites the thing under version control turns every
        idle session into a git diff. Save stays explicit; this exists only
        so a crash or a closed browser can't lose work."""
        if self.recovery_path is None:
            return
        try:
            self.recovery_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "project_name": self.project_name,
                "project_path": str(self.project_path) if self.project_path else None,
                "graph": graph_to_dict(self.graph, self.project_name),
            }
            self.recovery_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            # Recovery is a safety net, not a feature the user asked for --
            # an unwritable cache directory must not fail their edit.
            pass

    def recovery_info(self) -> dict[str, Any] | None:
        """Metadata about an available recovery snapshot, or None. Used to
        offer recovery on a fresh start; never auto-applied."""
        if self.recovery_path is None or not self.recovery_path.exists():
            return None
        try:
            payload = json.loads(self.recovery_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        blocks = payload.get("graph", {}).get("blocks", {})
        if not blocks:
            return None
        return {
            "saved_at": payload.get("saved_at"),
            "project_name": payload.get("project_name"),
            "project_path": payload.get("project_path"),
            "block_count": len(blocks),
        }

    def clear_recovery(self) -> None:
        """Discard the crash-recovery snapshot. Called after a save (the
        work it protected is now safe in the user's own project file) and
        when the user explicitly declines the restore prompt at startup --
        either way, re-offering the same stale snapshot on every later
        startup would just be noise."""
        if self.recovery_path is None:
            return
        try:
            self.recovery_path.unlink(missing_ok=True)
        except OSError:
            pass

    def recover(self) -> None:
        if self.recovery_path is None or not self.recovery_path.exists():
            raise ValueError("no recovery snapshot available")
        payload = json.loads(self.recovery_path.read_text(encoding="utf-8"))
        with self.edit():
            self.graph = graph_from_dict(payload["graph"])
            self.runner.graph = self.graph
            self.runner.state = {}
            self.project_name = payload.get("project_name") or "untitled"
            saved_path = payload.get("project_path")
            self.project_path = Path(saved_path) if saved_path else None

    def new(self) -> None:
        """Start a brand-new, empty project -- the in-memory equivalent of a
        fresh server start, callable mid-session: an empty graph/runner, no
        project path, and a clean edit history. Never touches the
        filesystem; nothing exists on disk until the next Save."""
        self.graph = Graph()
        self.runner = Runner(self.graph, CacheStore(CACHE_DIR), sample_rows=self.runner.sample_rows)
        self.project_path = None
        self.project_name = "untitled"
        self._undo.clear()
        self._redo.clear()
        self.revision = 0
        self.saved_revision = 0
        self.clear_recovery()

    def load(self, path: Path) -> None:
        self.graph = load_project(path)
        self.runner = Runner(self.graph, CacheStore(CACHE_DIR), sample_rows=self.runner.sample_rows)
        self.project_path = path
        self.project_name = path.stem
        # A different project is a different edit history.
        self._undo.clear()
        self._redo.clear()
        self.revision = 0
        self.saved_revision = 0

    def save(self, path: Path | None = None) -> Path:
        target = path or self.project_path
        if target is None:
            raise ValueError("no project path set; provide one to save")
        save_project(self.graph, target, project_name=self.project_name)
        self.project_path = target
        self.saved_revision = self.revision
        # The work this snapshot exists to protect is now safely on disk in
        # the user's own project file -- leaving it behind would just nag
        # with the same stale "restore?" prompt on every future startup,
        # forever, even though there's nothing left to recover.
        self.clear_recovery()
        return target

    def set_sample_rows(self, rows: int | None) -> None:
        """Sample mode is a view of the data, not part of the project, so it
        is neither saved nor undoable -- but it does change every block's
        cache key (see Runner._basis), so statuses shift as soon as it's set."""
        if rows is not None and rows < 1:
            raise ValueError("sample rows must be at least 1")
        self.runner.sample_rows = rows

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
