"""Moving a placed block to another version of its definition -- a library
block's "Update to vN", a built-in's "Accept" after an upgrade changed it --
and the status badges that offer it. Nothing here changes a model on its
own: the UI shows `preview` (code diff plus structural impact) and the user
confirms `apply`.
"""

from __future__ import annotations

import difflib
import inspect
from typing import Any

from . import block_library as bl
from . import library_store
from .blocks.base import BLOCK_REGISTRY, BlockDefinition
from .graph import BlockInstance


def status_for(block: BlockInstance) -> dict[str, Any] | None:
    """Where a placed block's definition came from and whether a newer one
    is available -- see api._block_out's `definition_info`."""
    if block.definition is None:
        current = BLOCK_REGISTRY.get(block.category)
        if current is None:
            return {"source": "builtin", "missing": True}
        changed = block.builtin_sha is not None and block.builtin_sha != current.sha
        return {"source": "builtin", "sha": current.sha, "changed_since_saved": changed}
    ref = block.library_ref or {}
    embedded = bl.definition_for(block)
    info: dict[str, Any] = {
        "source": "library",
        "file": ref.get("file"),
        "version": embedded.version,
        "sha": embedded.sha,
        "in_library": False,
        "update_available": False,
    }
    try:
        found = library_store.find(ref["file"], ref.get("id", block.category)) if ref.get("file") else None
    except library_store.LibraryError:
        found = None
    if found is None:
        hit = library_store.find_by_id(block.category)
        found = hit[1] if hit else None
        if hit:
            info["file"] = hit[0].name
    if found is not None:
        info["in_library"] = True
        info["latest_version"] = found.version
        info["update_available"] = found.sha != embedded.sha
    return info


def target_for(block: BlockInstance) -> tuple[BlockDefinition, dict[str, Any] | None]:
    """The definition `apply` would move this block to: the installed
    built-in, or the library's current version of its block."""
    if block.definition is None:
        return BLOCK_REGISTRY[block.category], None
    info = status_for(block) or {}
    if not info.get("in_library"):
        raise library_store.LibraryError("this block's definition isn't in the library any more")
    f = library_store.get_file(info["file"])
    d = f.definition(block.category)
    return d, library_store.library_ref(f, d)


def _required_params(d: BlockDefinition) -> set[str]:
    ports = {p.name for p in d.inputs}
    return {
        n
        for n, p in inspect.signature(d.fn).parameters.items()
        if p.default is inspect.Parameter.empty
        and n not in ports
        and n not in bl.INJECTED_PARAMS
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _param_names(d: BlockDefinition) -> set[str]:
    return set(inspect.signature(d.fn).parameters) - {p.name for p in d.inputs}


def migrate_params(params: dict[str, Any], new: BlockDefinition) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """(carried params, {old: new} renames applied, dropped param names)."""
    names = _param_names(new)
    out: dict[str, Any] = {}
    renamed: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in params.items():
        target = new.renamed.get(key, key)
        if target in names or key == "probe_query":
            out[target] = value
            if target != key:
                renamed[key] = target
        else:
            dropped.append(key)
    return out, renamed, dropped


def preview(graph, block_id: str, target: BlockDefinition | None = None) -> dict[str, Any]:
    """What moving `block_id` to `target` (default: target_for) would do:
    the code diff, params carried/renamed/dropped/newly required, and the
    wires that stay, get renamed or would be disconnected."""
    block = graph.blocks[block_id]
    current = block.resolve()
    if target is None:
        target, _ = target_for(block)
    params, renamed, dropped = migrate_params(block.params, target)
    new_required = sorted(_required_params(target) - set(params))
    wires = _wire_plan(graph, block_id, target)
    diff = "".join(
        difflib.unified_diff(
            current.code.splitlines(keepends=True),
            target.code.splitlines(keepends=True),
            fromfile=f"{current.id} v{current.version}",
            tofile=f"{target.id} v{target.version}",
        )
    )
    meta_changed = current.meta != target.meta
    return {
        "block": block_id,
        "from_version": current.version,
        "to_version": target.version,
        "code_diff": diff,
        "meta_changed": meta_changed,
        "params_carried": sorted(params),
        "params_renamed": renamed,
        "params_dropped": dropped,
        "params_newly_required": new_required,
        "wires_kept": [w["wire"] for w in wires if w["action"] == "keep"],
        "wires_renamed": {w["wire"]: w["to_port"] for w in wires if w["action"] == "rename"},
        "wires_disconnected": [w for w in wires if w["action"] == "disconnect"],
    }


def _wire_plan(graph, block_id: str, target: BlockDefinition) -> list[dict[str, Any]]:
    ins = {p.name: p.type for p in target.inputs}
    outs = {p.name: p.type for p in target.outputs}
    plan = []
    for w in graph.wires.values():
        if w.to_block == block_id:
            side, port, ports = "input", w.to_port, ins
            other_type = next((p.type for p in graph.blocks[w.from_block].outputs if p.name == w.from_port), "any")
        elif w.from_block == block_id:
            side, port, ports = "output", w.from_port, outs
            other_type = next((p.type for p in graph.blocks[w.to_block].inputs if p.name == w.to_port), "any")
        else:
            continue
        new_port = target.renamed_ports.get(port, port)
        entry = {"wire": w.id, "side": side, "port": port, "to_port": new_port}
        if new_port not in ports:
            entry.update(action="disconnect", reason=f"{side} port {port!r} no longer exists")
        elif ports[new_port] != other_type and "any" not in (ports[new_port], other_type):
            entry.update(action="disconnect", reason=f"{side} port {new_port!r} is now {ports[new_port]}")
        else:
            entry["action"] = "rename" if new_port != port else "keep"
        plan.append(entry)
    return plan


def apply(session, block_id: str) -> dict[str, Any]:
    """Move the block to its target definition (see preview), carrying
    params and wires over and removing wires that can't stay. Call inside
    session.edit() so it's one undo step."""
    graph = session.graph
    block = graph.blocks[block_id]
    target, ref = target_for(block)
    summary = preview(graph, block_id, target)
    params, _, _ = migrate_params(block.params, target)
    for w in _wire_plan(graph, block_id, target):
        wire = graph.wires[w["wire"]]
        if w["action"] == "disconnect":
            session.delete_wire(w["wire"])
        elif w["action"] == "rename":
            if w["side"] == "input":
                wire.to_port = w["to_port"]
            else:
                wire.from_port = w["to_port"]
    session.set_definition(block_id, None if block.definition is None else target, library_ref=ref, params=params)
    return summary
