from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from . import gitops
from .blocks.base import PortSpec
from .graph import Artifact, BlockInstance, Graph, Lane, Position, Wire

# The fixed filename Save/Load look for inside a project folder -- the
# folder itself is "the project" (it's what gets versioned, git-inited, and
# pushed); this file is just where its graph lives within it.
PROJECT_FILENAME = "model.json"

# Written into a project folder's .gitignore the first time it's given one
# (see ensure_project_scaffold) -- covers the regenerable local cache and
# common Python/editor cruft; the project JSON, its block code files, and
# whatever the user drops in files/ are meant to be versioned.
DEFAULT_GITIGNORE = """\
.modelmaker-cache/
__pycache__/
*.pyc
.env
.DS_Store
"""


def ensure_project_scaffold(project_dir: Path) -> None:
    """Give a project folder the standard layout the first time it's saved
    to: a `files/` subdirectory for data/artifacts the project owns, a
    .gitignore, and a git repo so the folder is ready to version and push
    right away. Idempotent -- never overwrites a .gitignore the user has
    since edited, never re-inits an existing repo, safe to call on every
    save."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "files").mkdir(exist_ok=True)
    gitignore = project_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(DEFAULT_GITIGNORE, encoding="utf-8")
    try:
        gitops.init(project_dir)
    except gitops.GitError:
        # No git on PATH, or some other local git failure -- the project
        # folder itself is still perfectly usable without version control.
        pass


def is_project_dir(path: Path) -> bool:
    """Whether `path` is a folder holding a saved project -- i.e. it has a
    PROJECT_FILENAME in it. Used to validate a folder picked for Load and to
    let the browse endpoint flag which subfolders are project folders."""
    return (path / PROJECT_FILENAME).is_file()


# Version 2: blocks are built-ins or embedded library definitions; the
# canvas-only custom blocks of version 1 no longer exist.
PROJECT_FORMAT = 2


class ProjectFormatError(ValueError):
    pass


def graph_from_dict(data: dict[str, Any], project_dir: Path | None = None) -> Graph:
    """Rebuild a Graph from its serialized form. A library block's embedded
    definition carries its code inline under "code" (in-memory snapshots --
    see graph_to_dict) or in the file named by "code_ref", resolved against
    project_dir (a saved project). Ports are refreshed from each block's
    definition, which is what actually runs."""
    lanes = {
        lid: Lane(name=lane["name"], order=lane["order"], height=lane.get("height", 260.0))
        for lid, lane in data.get("lanes", {}).items()
    }

    blocks: dict[str, BlockInstance] = {}
    for bid, b in data.get("blocks", {}).items():
        if b.get("is_custom"):
            raise ProjectFormatError(
                f"block {b.get('name', bid)!r} is a custom block from an older Model-Maker -- custom blocks now "
                "live in the block library; recreate it there and place it again"
            )
        definition = b.get("definition")
        if definition is not None:
            definition = dict(definition)
            ref = definition.pop("code_ref", None)
            if "code" not in definition:
                if ref is None or project_dir is None:
                    raise ProjectFormatError(f"block {bid!r}: embedded definition has no code")
                definition["code"] = (project_dir / ref).read_text(encoding="utf-8")
        pos = b.get("position", {"x": 0, "y": 0})
        block = BlockInstance(
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
            definition=definition,
            library_ref=b.get("library_ref"),
            builtin_sha=b.get("builtin_sha"),
            column_role_overrides=b.get("column_role_overrides", {}),
            port_names=b.get("port_names", {}),
            group_by=b.get("group_by"),
            max_workers=b.get("max_workers"),
            column_tags=b.get("column_tags", {}),
            provenance=b.get("provenance"),
        )
        try:
            resolved = block.resolve()
        except KeyError:
            pass  # unknown built-in (e.g. removed in this version): kept as saved, goes red when run
        else:
            block.inputs = [replace(p) for p in resolved.inputs]
            block.outputs = [replace(p) for p in resolved.outputs]
            block.block_type = resolved.block_type
            if block.definition is None and block.builtin_sha is None:
                # Saved before built-ins were pinned: pin to the installed one.
                block.builtin_sha = resolved.sha
        blocks[bid] = block

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

    artifacts = {
        aid: Artifact(
            id=aid,
            kind=a["kind"],
            title=a["title"],
            block_id=a["block_id"],
            port=a["port"],
            document=a["document"],
            source_key=a.get("source_key"),
            created_at=a["created_at"],
            updated_at=a["updated_at"],
        )
        for aid, a in data.get("artifacts", {}).items()
    }

    return Graph(lanes=lanes, blocks=blocks, wires=wires, artifacts=artifacts)


def graph_to_dict(graph: Graph, project_name: str = "project", project_dir: Path | None = None) -> dict[str, Any]:
    """Serialize a Graph. With project_dir given, a library block's embedded
    code is written to blocks/<block id>.py and referenced by "code_ref"
    (keeps git diffs line-level); without one, the code rides inline under
    "code", for snapshots that never touch disk."""
    blocks_out: dict[str, Any] = {}
    for bid, b in sorted(graph.blocks.items()):
        entry: dict[str, Any] = {
            "block_type": b.block_type,
            "category": b.category,
            "name": b.name,
            "lane": b.lane,
            "position": {"x": b.position.x, "y": b.position.y},
            "code_version": b.code_version,
            "params": b.params,
            "column_role_overrides": b.column_role_overrides,
            "port_names": b.port_names,
            "group_by": b.group_by,
            "max_workers": b.max_workers,
            "column_tags": b.column_tags,
            "ports": {
                "inputs": [asdict(p) for p in b.inputs],
                "outputs": [asdict(p) for p in b.outputs],
            },
        }
        # Only written when set, so projects with no AI-built blocks save
        # byte-identically to before this field existed.
        if b.provenance is not None:
            entry["provenance"] = b.provenance
        if b.builtin_sha is not None:
            entry["builtin_sha"] = b.builtin_sha
        if b.definition is not None:
            definition = dict(b.definition)
            if project_dir is not None:
                code_ref = f"blocks/{bid}.py"
                sidecar = project_dir / code_ref
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(definition.pop("code"), encoding="utf-8")
                definition["code_ref"] = code_ref
            entry["definition"] = definition
            entry["library_ref"] = b.library_ref
        blocks_out[bid] = entry

    return {
        "modelmaker_version": PROJECT_FORMAT,
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
        "artifacts": {aid: asdict(a) for aid, a in sorted(graph.artifacts.items())},
    }


def load_project(path: Path) -> Graph:
    return graph_from_dict(json.loads(path.read_text(encoding="utf-8")), project_dir=path.parent)


def save_project(graph: Graph, path: Path, project_name: str = "project") -> None:
    ensure_project_scaffold(path.parent)
    data = graph_to_dict(graph, project_name=project_name, project_dir=path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
