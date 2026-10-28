"""Read-only views of the open project for an outside assistant -- the VS
Code extension's Copilot Chat tools (vscode/src/tools.ts). Edits go through
the canvas's own endpoints, so they land in the undo history like any
other. What's shown follows the AI builder's rule (agent/tools.py): the
graph, block documentation, and output schemas with redacted summary
statistics -- never a dataframe's rows."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .agent import catalogue
from .agent.tools import is_statistics_output, summarize_value, table_rows
from .packet import DataFramePacket
from .session import wire_is_valid

router = APIRouter(prefix="/api/assistant")

# A red block's error, trimmed to what helps fix it.
ERROR_CHARS = 1200


def _session():
    from . import api

    return api.SESSION


def _block_brief(bid: str) -> dict[str, Any]:
    session = _session()
    block = session.graph.blocks[bid]
    runner = session.runner
    status = runner.status(bid)
    st = runner.state.get(bid)
    out: dict[str, Any] = {
        "id": bid,
        "name": block.name,
        "category": block.category,
        "lane": block.lane,
        "status": status,
        "params": block.params,
        "inputs": [f"{p.name}:{p.type}" for p in block.inputs],
        "outputs": [f"{p.name}:{p.type}" for p in block.outputs],
        "position": {"x": block.position.x, "y": block.position.y},
    }
    if block.definition is not None:
        out["library_block"] = True
    if block.column_role_overrides:
        out["column_roles"] = block.column_role_overrides
    if status == "red" and st and st.last_error:
        out["error"] = st.last_error[-ERROR_CHARS:]
    if status == "orange":
        out["stale_reason"] = runner.stale_reason(bid)
    return out


@router.get("/graph")
def graph() -> dict[str, Any]:
    """Lanes, every block (status, params, ports, errors) in run order, and
    every wire."""
    session = _session()
    g = session.graph
    return {
        "project": session.project_name,
        "lanes": [{"id": lid, "name": lane.name} for lid, lane in sorted(g.lanes.items(), key=lambda kv: kv[1].order)],
        "blocks": [_block_brief(bid) for bid in g.topo_order()],
        "wires": [
            {"id": wid, "from": f"{w.from_block}.{w.from_port}", "to": f"{w.to_block}.{w.to_port}", "valid": wire_is_valid(g, w)}
            for wid, w in g.wires.items()
        ],
        "running": session.runner.sweep_running or any(session.runner.status(b) == "running" for b in g.blocks),
    }


@router.get("/blocks/{block_id}")
def block(block_id: str) -> dict[str, Any]:
    if block_id not in _session().graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")
    return _block_brief(block_id)


@router.get("/block_types")
def block_types(tag: str | None = None) -> dict[str, Any]:
    """The tags, or the blocks with one tag (built-ins and the library)."""
    if not tag:
        return {"tags": catalogue.list_tags(True)}
    try:
        return {"tag": tag, "blocks": catalogue.list_blocks_for_tag(tag, True)}
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


@router.get("/block_types/{category}")
def block_type(category: str) -> dict[str, Any]:
    try:
        return catalogue.describe_block_type(category, True)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


@router.get("/output")
def output(block: str, port: str | None = None) -> dict[str, Any]:
    """A block's last output on one port: a dataframe's schema, row count and
    redacted statistics; a statistics table's rows; a metric's value."""
    from .api import _get_cached_output

    session = _session()
    value = _get_cached_output(block, port)
    port = port or session.graph.blocks[block].outputs[0].name
    if isinstance(value, DataFramePacket) and is_statistics_output(session.graph, block, port):
        summary = table_rows(value)
    else:
        summary = summarize_value(value)
    return {"block": block, "port": port, "status": session.runner.status(block), **summary}
