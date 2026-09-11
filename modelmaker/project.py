from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .blocks.base import PortSpec
from .graph import BlockInstance, Graph, Lane, Position, Wire


def graph_from_dict(data: dict[str, Any], project_dir: Path | None = None) -> Graph:
    """Rebuild a Graph from its serialized form. A custom block's code comes
    from an inline "code" key when present (how in-memory snapshots carry it
    -- see graph_to_dict), else from the sidecar file named by "code_ref",
    resolved against project_dir (how a saved project carries it)."""
    lanes = {
        lid: Lane(name=lane["name"], order=lane["order"], height=lane.get("height", 260.0))
        for lid, lane in data.get("lanes", {}).items()
    }

    blocks: dict[str, BlockInstance] = {}
    for bid, b in data.get("blocks", {}).items():
        code = b.get("code")
        if code is None and b.get("code_ref") and project_dir is not None:
            code = (project_dir / b["code_ref"]).read_text(encoding="utf-8")
        pos = b.get("position", {"x": 0, "y": 0})
        blocks[bid] = BlockInstance(
            id=bid,
            block_type=b["block_type"],
            category=b["category"],
            name=b["name"],
            lane=b.get("lane"),
            position=Position(x=pos["x"], y=pos["y"]),
            code_version=b.get("code_version", 1),
            params=b.get("params", {}),
            inputs=[PortSpec(**p) for p in b["ports"]["inputs"]],
            outputs=[PortSpec(**p) for p in b["ports"]["outputs"]],
            code_ref=b.get("code_ref"),
            code=code,
            metadata_transform=b.get("metadata_transform"),
            is_custom=b.get("is_custom", False),
            column_role_overrides=b.get("column_role_overrides", {}),
            port_names=b.get("port_names", {}),
            group_by=b.get("group_by"),
            max_workers=b.get("max_workers"),
        )

    wires = {
        wid: Wire(
            id=wid,
            from_block=w["from"]["block"],
            from_port=w["from"]["port"],
            to_block=w["to"]["block"],
            to_port=w["to"]["port"],
        )
        for wid, w in data.get("wires", {}).items()
    }

    return Graph(lanes=lanes, blocks=blocks, wires=wires)


def graph_to_dict(graph: Graph, project_name: str = "project", project_dir: Path | None = None) -> dict[str, Any]:
    """Serialize a Graph. With project_dir given, a custom block's code is
    written to a sidecar .py file and referenced by "code_ref" (keeps git
    diffs line-level -- see the plan's section 8.1); without one, the code
    rides inline under "code" instead, for snapshots that never touch disk."""
    blocks_out: dict[str, Any] = {}
    for bid, b in sorted(graph.blocks.items()):
        code_ref = b.code_ref
        inline_code: str | None = None
        if b.is_custom and b.code is not None:
            if project_dir is not None:
                code_ref = code_ref or f"blocks/{bid}.py"
                sidecar = project_dir / code_ref
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(b.code, encoding="utf-8")
            else:
                inline_code = b.code
        entry = {
            "block_type": b.block_type,
            "is_custom": b.is_custom,
            "category": b.category,
            "name": b.name,
            "lane": b.lane,
            "position": {"x": b.position.x, "y": b.position.y},
            "code_version": b.code_version,
            "params": b.params,
            "code_ref": code_ref,
            "metadata_transform": b.metadata_transform if b.is_custom else None,
            "column_role_overrides": b.column_role_overrides,
            "port_names": b.port_names,
            "group_by": b.group_by,
            "max_workers": b.max_workers,
            "ports": {
                "inputs": [asdict(p) for p in b.inputs],
                "outputs": [asdict(p) for p in b.outputs],
            },
        }
        if inline_code is not None:
            entry["code"] = inline_code
        blocks_out[bid] = entry

    return {
        "modelmaker_version": 1,
        "project_name": project_name,
        "lanes": {
            lid: {"name": lane.name, "order": lane.order, "height": lane.height}
            for lid, lane in sorted(graph.lanes.items(), key=lambda kv: kv[1].order)
        },
        "blocks": blocks_out,
        "wires": {
            wid: {
                "from": {"block": w.from_block, "port": w.from_port},
                "to": {"block": w.to_block, "port": w.to_port},
            }
            for wid, w in sorted(graph.wires.items())
        },
    }


def load_project(path: Path) -> Graph:
    return graph_from_dict(json.loads(path.read_text(encoding="utf-8")), project_dir=path.parent)


def save_project(graph: Graph, path: Path, project_name: str = "project") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = graph_to_dict(graph, project_name=project_name, project_dir=path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
