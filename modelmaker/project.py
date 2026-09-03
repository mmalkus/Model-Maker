from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .blocks.base import PortSpec
from .graph import BlockInstance, Graph, Lane, Position, Wire


def load_project(path: Path) -> Graph:
    data = json.loads(path.read_text(encoding="utf-8"))
    project_dir = path.parent

    lanes = {lid: Lane(name=l["name"], order=l["order"]) for lid, l in data.get("lanes", {}).items()}

    blocks: dict[str, BlockInstance] = {}
    for bid, b in data.get("blocks", {}).items():
        code = None
        if b.get("code_ref"):
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


def save_project(graph: Graph, path: Path, project_name: str = "project") -> None:
    project_dir = path.parent

    blocks_out = {}
    for bid, b in sorted(graph.blocks.items()):
        code_ref = b.code_ref
        if b.block_type == "llm_authored" and b.code is not None:
            code_ref = code_ref or f"blocks/{bid}.py"
            sidecar = project_dir / code_ref
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(b.code, encoding="utf-8")
        blocks_out[bid] = {
            "block_type": b.block_type,
            "category": b.category,
            "name": b.name,
            "lane": b.lane,
            "position": {"x": b.position.x, "y": b.position.y},
            "code_version": b.code_version,
            "params": b.params,
            "code_ref": code_ref,
            "metadata_transform": b.metadata_transform if b.block_type == "llm_authored" else None,
            "ports": {
                "inputs": [asdict(p) for p in b.inputs],
                "outputs": [asdict(p) for p in b.outputs],
            },
        }

    wires_out = {
        wid: {
            "from": {"block": w.from_block, "port": w.from_port},
            "to": {"block": w.to_block, "port": w.to_port},
        }
        for wid, w in sorted(graph.wires.items())
    }

    lanes_out = {
        lid: {"name": l.name, "order": l.order}
        for lid, l in sorted(graph.lanes.items(), key=lambda kv: kv[1].order)
    }

    data = {
        "modelmaker_version": 1,
        "project_name": project_name,
        "lanes": lanes_out,
        "blocks": blocks_out,
        "wires": wires_out,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
