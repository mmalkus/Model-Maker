"""Read-only views of the open project for an outside assistant -- the VS
Code extension's Copilot Chat tools (vscode/src/tools.ts). Edits go through
the canvas's own endpoints, so they land in the undo history like any
other. What's shown follows the AI builder's rule (agent/tools.py): the
graph, block documentation, and output schemas with redacted summary
statistics -- never a dataframe's rows."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import capabilities, features

from .agent import catalogue, leakage
from .agent.hints import decision_hints
from .agent.layout import DEFAULT_LANE_HEIGHT, Placer, next_lane_order
from .llm.prompts import LARGE_DATA_RULES
from .agent.tools import is_statistics_output, leak_graph, summarize_value, suspicious_in_graph, table_rows
from .packet import DataFramePacket
from .blocks.base import BLOCK_REGISTRY
from .session import new_id, wire_is_valid

router = APIRouter(prefix="/api/assistant")

# check's per-block hints, for at most this many blocks (the most recent in
# run order), so a big graph doesn't flood the chat.
MAX_HINTED_BLOCKS = 12

# A red block's error, trimmed to what helps fix it.
ERROR_CHARS = 1200


def _workspace():
    from . import api

    return api.current_workspace()


def _session():
    return _workspace().session


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
    if block.block_type == "input":
        out["data_source"] = True
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


# Blocks the AI builder leaves to the user (catalogue.AGENT_DISALLOWED) but a
# chat assistant may place, since the user confirms each edit: reading the
# data in, and writing results out. Listed under tags of their own.
IO_TAGS = {
    "data_sources": ("Read the data in: files, the workspace's datasets and sources, the demo data. Start here.", "input"),
    "exports": ("Write results out to a file or the workspace's exports.", "output"),
}
IO_BLOCKS = {"input": ("read_csv", "read_parquet", "read_json", "read_excel", "read_sql", "read_dataset", "read_source", "demo_credit_data"),
             "output": ("write_csv", "write_parquet", "export")}


def _io_blocks(kind: str) -> list[dict[str, Any]]:
    catalogue.ensure_blocks_registered()
    out = []
    for category in IO_BLOCKS[kind]:
        spec = BLOCK_REGISTRY.get(category)
        if spec is not None and capabilities.allowed(spec):
            out.append({"category": category, "summary": catalogue._first_sentence(spec.summary or spec.display_name)})
    if kind == "input":
        # The user's own data-source blocks (the AI builder's catalogue
        # leaves library input blocks out; here the user confirms each edit).
        for file, d in _library_inputs():
            out.append({"category": d.id, "summary": catalogue._first_sentence(d.summary or d.display_name), "library": file})
    return out


def _library_inputs():
    """(file, definition) of each valid library input block this workspace allows."""
    from . import library_store

    if features.CUSTOM_BLOCKS not in _workspace().features:
        return []
    found = []
    try:
        files = library_store.list_files()
    except library_store.LibraryError:
        return []
    for f in files:
        if f.parsed.file_errors:
            continue
        for d in f.parsed.definitions:
            if d.block_type == "input" and not f.parsed.errors.get(d.id) and d.id not in BLOCK_REGISTRY and capabilities.allowed(d):
                found.append((f.name, d))
    return found


@router.get("/block_types")
def block_types(tag: str | None = None) -> dict[str, Any]:
    """The tags, or the blocks with one tag (built-ins and the library)."""
    if not tag:
        io = [{"tag": t, "about": about} for t, (about, kind) in IO_TAGS.items() if _io_blocks(kind)]
        return {"tags": io + catalogue.list_tags(True)}
    if tag in IO_TAGS:
        return {"tag": tag, "blocks": _io_blocks(IO_TAGS[tag][1])}
    try:
        return {"tag": tag, "blocks": catalogue.list_blocks_for_tag(tag, True)}
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


class NewBlock(BaseModel):
    category: str
    name: str | None = None
    lane: str | None = None  # a lane id or name
    params: dict[str, Any] = {}


class NewLane(BaseModel):
    name: str


def _lane_id(lane: str | None) -> str | None:
    """A lane given by id or (case-insensitive) name, as models tend to use
    the name they were shown."""
    if not lane:
        return None
    lanes = _session().graph.lanes
    if lane in lanes:
        return lane
    for lid, ln in lanes.items():
        if ln.name.strip().lower() == lane.strip().lower():
            return lid
    known = ", ".join(f"{ln.name} ({lid})" for lid, ln in lanes.items()) or "none -- add one with add_lane"
    raise HTTPException(400, f'no lane "{lane}"; the lanes are: {known}')


@router.post("/blocks")
def add_block(req: NewBlock) -> dict[str, Any]:
    """Place a block the way the AI builder does: a built-in or a library
    block by its category, its params checked against the block's own, laid
    out in its lane after what's already there."""
    from . import library_store

    spec = catalogue.lookup(req.category, True)
    library_input = None
    if spec is None:
        library_input = next(((f, d) for f, d in _library_inputs() if d.id == req.category), None)
        spec = library_input[1] if library_input else None
    if spec is None:
        raise HTTPException(400, f"unknown block category {req.category!r}{catalogue.unknown_category_hint(req.category, True)}")
    known = {p["name"] for p in catalogue.block_params(spec)}
    unknown = sorted(set(req.params) - known)
    if unknown:
        raise HTTPException(400, f"{req.category} has no params {unknown}; its params are {sorted(known)}")
    definition = ref = None
    if spec.source == "library":
        if features.CUSTOM_BLOCKS not in _workspace().features:
            raise HTTPException(403, features.refusal(features.CUSTOM_BLOCKS))
        try:
            definition, ref = library_store.placeable(library_input[0] if library_input else spec.origin, req.category)
        except library_store.LibraryError as e:
            raise HTTPException(400, str(e))
    session = _session()
    lane = _lane_id(req.lane)
    x, y = Placer(session.graph).place(lane)
    try:
        with session.edit():
            block = session.add_block(
                req.category, name=req.name or (spec.display_name if definition else None), lane=lane, x=x, y=y,
                params=req.params, definition=definition, library_ref=ref,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _block_brief(block.id)


@router.post("/lanes")
def add_lane(req: NewLane) -> dict[str, Any]:
    """A new lane (a phase band, e.g. "Data", "Estimation"), below the others."""
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "a lane needs a name")
    session = _session()
    for lid, ln in session.graph.lanes.items():
        if ln.name.strip().lower() == name.lower():
            return {"id": lid, "name": ln.name, "existed": True}
    lane_id = new_id("lane")
    with session.edit():
        session.set_lane(lane_id, name, next_lane_order(session.graph), height=DEFAULT_LANE_HEIGHT)
    return {"id": lane_id, "name": name}


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


def _last_outputs(block_id: str) -> dict[str, Any] | None:
    """{port: value} of a block's last successful run, or None."""
    runner = _session().runner
    st = runner.state.get(block_id)
    entry = runner.cache.get(st.last_successful_key) if st and st.last_successful_key else None
    return dict(entry.outputs) if entry is not None else None


@router.get("/check")
def check(block: str | None = None) -> dict[str, Any]:
    """The AI builder's judgement, for a pipeline someone else is building:
    holdout leakage in the wiring, features fit_binning banded 'suspicious'
    (IV >= 0.5, usually leakage), and the decision hints of blocks that ran
    green -- every such block, or just `block`. Advice, not a guard: the
    chat tools report it and the user decides."""
    session = _session()
    g = session.graph
    if block is not None and block not in g.blocks:
        raise HTTPException(404, f"no such block: {block}")
    leaks = [{"block": lk.block, "fitter": lk.fitter, "message": lk.message} for lk in leakage.find_leaks(*leak_graph(g))]

    def summary(bid: str) -> Any:
        return (_last_outputs(bid) or {}).get("summary")

    suspicious = suspicious_in_graph(g, summary)
    hints: dict[str, list[str]] = {}
    candidates = [block] if block else [b for b in g.topo_order() if session.runner.status(b) == "green"][-MAX_HINTED_BLOCKS:]
    for bid in candidates:
        outputs = _last_outputs(bid)
        if outputs is None:
            continue
        try:
            found = decision_hints(g.blocks[bid].resolve(), outputs)
        except Exception as e:  # noqa: BLE001 -- a rule tripping over odd output shouldn't sink the whole check
            found = [f"(couldn't evaluate this block's hints: {e})"]
        if found:
            hints[bid] = found
    return {
        "leaks": leaks,
        "suspicious_features": [{"feature": f, "iv": iv} for f, iv in sorted(suspicious.items())],
        "hints": hints,
        "ok": not leaks and not suspicious,
    }


# ---- writing library blocks: an assistant's guide and check ------------------

BLOCK_FILE_RULES = [
    "A block file is one .py file in the library folder holding one or more blocks: a module docstring "
    "(the file's heading in the library), `import polars as pl`, a `BLOCKS = {...}` dict literal keyed by block "
    "id (read without running the file, so literals only), and for each id a top-level function of that name. "
    "The template below shows every key, commented; start from it.",
    "A function's first parameters are its input ports (named as in \"inputs\"), then its params, each with "
    "a default. It returns one value per output port (a tuple for several).",
    "Never hardcode an existing input column's name in the body: take it as a `<descriptive>_col` str param "
    "defaulted to the real column name (`<descriptive>_cols` for a list). New columns get literal names via "
    ".alias(...).",
    "Only polars at module level; import numpy, scipy or sklearn inside the function that uses them. No "
    "pandas, no network, no global state, and no file I/O except an input block reading its source.",
    "Give each block an `_example_<id>()` self-check: build a tiny pl.DataFrame by hand, call the block, "
    "assert something meaningful. check_block_file runs it.",
    "Block ids are unique across the library and the built-ins. When you change a block, bump its "
    "\"version\": copies already placed on a canvas then offer Update.",
    "Large data: " + LARGE_DATA_RULES,
    "When the file is written, call check_block_file with its name and fix what it reports until ok; then "
    "place a block with add_block, its id as the category.",
]


def _library_or_403():
    from . import library_store

    if features.CUSTOM_BLOCKS not in _workspace().features:
        raise HTTPException(403, features.refusal(features.CUSTOM_BLOCKS))
    return library_store


@router.get("/block_guide")
def block_guide() -> dict[str, Any]:
    """How to write a block file for the user's library, for an assistant
    that writes files itself (Claude Code): where the library is, what's in
    it, the rules, and the commented template."""
    from . import block_library as bl

    store = _library_or_403()
    files = []
    for f in store.list_files(force=True):
        files.append({"name": f.name, "blocks": [d.id for d in f.parsed.definitions], **({"errors": True} if not f.parsed.ok else {})})
    return {
        "library_dir": _workspace().shown(store.library_dir()),
        "files": files,
        "rules": BLOCK_FILE_RULES,
        "template": bl.TEMPLATE_PATH.read_text(encoding="utf-8"),
        "full_guide": "https://github.com/mmalkus/Model-Maker/blob/main/docs/block-authoring.md",
    }


class CheckFile(BaseModel):
    name: str  # the file's name in the library folder, e.g. "fannie_mae.py"


@router.post("/check_block_file")
def check_block_file(req: CheckFile) -> dict[str, Any]:
    """Re-reads the library folder, then checks one file there: parse and
    validation errors per block, and its _example_ self-checks run (in a
    worker process, never this one)."""
    from . import block_library as bl

    store = _library_or_403()
    store.reload()
    try:
        f = store.get_file(req.name)
    except store.LibraryError as e:
        names = ", ".join(x.name for x in store.list_files()) or "none yet"
        raise HTTPException(404, f"{e} (the library folder holds: {names})")
    blocks = [{"id": d.id, "block_type": d.block_type, **({"errors": f.parsed.errors[d.id]} if f.parsed.errors.get(d.id) else {})} for d in f.parsed.definitions]
    examples: dict[str, Any] = {}
    if not f.parsed.file_errors and f.parsed.examples:
        session = _workspace().session
        examples = bl.run_examples(f.source, f.name, None, executor=session.runner.executor, env=_workspace().env_vars.snapshot())
    ok = f.parsed.ok and all(r.get("ok") is not False for r in examples.values()) and bool(f.parsed.definitions)
    out: dict[str, Any] = {"file": f.name, "ok": ok, "file_errors": f.parsed.file_errors, "blocks": blocks, "examples": examples}
    missing = [d.id for d in f.parsed.definitions if d.id not in f.parsed.examples]
    if missing:
        out["missing_examples"] = missing
    if ok:
        out["next"] = "place a block with add_block, its id as the category"
    return out
