"""HTTP routes for the block library (see library_store.py) and for moving
placed blocks between definition versions (see library_updates.py).
Included by api.py; reads the live session from there."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import block_library as bl
from . import library_store as store
from . import library_updates
from .blocks.base import BLOCK_REGISTRY, BlockDefinition
from .forms import block_form
from .packet import DataFramePacket

router = APIRouter()


def _session():
    from . import api

    return api.SESSION


def _fail(e: Exception) -> HTTPException:
    if isinstance(e, store.ConflictError):
        return HTTPException(409, str(e))
    return HTTPException(400, str(e))


def architecture(d: BlockDefinition) -> dict[str, Any]:
    """What the architecture will do for this block, derived from what it
    declares -- shown in the library so nothing is hidden: change
    detection, streaming, hints, statistics tables."""
    kinds = {f.kind for f in d.form}
    changes = []
    if "path" in kinds:
        changes.append("file changes (modified time and size)")
    if "sql" in kinds and "env" in kinds:
        changes.append("its probe query, when set")
    df_inputs = [p for p in d.inputs if p.type == "dataframe"]
    single_df_out = len(d.outputs) == 1 and d.outputs[0].type == "dataframe"
    if d.block_type == "input" and single_df_out and d.outputs[0].lazy:
        streaming = "fuses: its lazy scan streams into the blocks after it"
    elif single_df_out and df_inputs and len(df_inputs) == len(d.inputs) and all(p.lazy for p in df_inputs):
        streaming = "fuses with neighbouring lazy blocks in a streaming run"
    elif d.block_type == "output" and not d.outputs and len(d.inputs) == 1 and d.inputs[0].lazy:
        streaming = "streams its input straight to its destination (a sink)"
    else:
        streaming = None
    return {
        "change_detection": changes if d.block_type == "input" else [],
        "streaming": streaming,
        "hints": d.hints,
        "statistics_tables": list(d.aggregate_outputs),
        "sampling": d.block_type == "input",
    }


def block_summary(d: BlockDefinition, errors: list[str] | None = None) -> dict[str, Any]:
    out = {
        "id": d.id,
        "display_name": d.display_name,
        "summary": d.summary,
        "description": d.description,
        "version": d.version,
        "sha": d.sha,
        "block_type": d.block_type,
        "group": d.group or d.block_type,
        "tags": list(d.tags),
        "requires": list(d.requires),
        "inputs": [asdict(p) for p in d.inputs],
        "outputs": [asdict(p) for p in d.outputs],
        "errors": errors or [],
        "unmet_requirements": bl.check_requirements(d),
        "architecture": architecture(d),
        "source": d.source,
        "origin": d.origin,
    }
    if not errors:
        try:
            out["form"] = block_form(d)
        except Exception:  # noqa: BLE001 -- shown as having no form
            out["form"] = None
    return out


def file_summary(f: store.LibraryFile) -> dict[str, Any]:
    return {
        "name": f.name,
        "doc": f.parsed.docstring,
        "sha": f.sha,
        "is_draft": f.is_draft,
        "file_errors": f.parsed.file_errors,
        "examples": sorted(f.parsed.examples),
        "blocks": [block_summary(d, f.parsed.errors.get(d.id)) for d in f.parsed.definitions],
    }


def _usage() -> dict[str, list[str]]:
    """Block id -> ids of the placed blocks in the open project using it."""
    out: dict[str, list[str]] = {}
    for bid, b in _session().graph.blocks.items():
        out.setdefault(b.category, []).append(bid)
    return out


@router.get("/api/library")
def list_library(refresh: bool = False) -> dict[str, Any]:
    """Every library file and its blocks, plus the built-ins grouped by file
    (read-only), and which blocks the open project uses."""
    files = store.list_files(force=refresh)
    builtin_files: dict[str, list[dict[str, Any]]] = {}
    for d in BLOCK_REGISTRY.values():
        builtin_files.setdefault(d.origin or "built-in", []).append(block_summary(d))
    return {
        "dir": str(store.library_dir()),
        "generation": store.generation,
        "files": [file_summary(f) for f in files],
        "builtin_files": [{"name": name, "blocks": blocks} for name, blocks in builtin_files.items()],
        "usage": _usage(),
    }


@router.get("/api/library/template")
def library_template() -> dict[str, str]:
    """The commented starting point for a new block file (see blocks/_template.py)."""
    return {"source": bl.TEMPLATE_PATH.read_text(encoding="utf-8")}


@router.post("/api/library/reload")
def reload_library() -> dict[str, Any]:
    return {"generation": store.reload()}


@router.get("/api/library/files/{name:path}")
def get_library_file(name: str) -> dict[str, Any]:
    try:
        f = store.get_file(name)
    except store.LibraryError as e:
        raise HTTPException(404, str(e))
    return {**file_summary(f), "source": f.source}


class CheckRequest(BaseModel):
    source: str
    name: str = "untitled.py"


@router.post("/api/library/check")
def check_source(req: CheckRequest) -> dict[str, Any]:
    """Parse and validate unsaved source -- the editor's live checklist.
    Defines the functions (to check forms against signatures) but never
    calls them."""
    taken = set(BLOCK_REGISTRY) | {d.id for f in store.list_files() if f.name != req.name for d in f.parsed.definitions}
    parsed = bl.validate(bl.parse_source(req.source, req.name), taken)
    return {
        "file_errors": parsed.file_errors,
        "blocks": [block_summary(d, parsed.errors.get(d.id)) for d in parsed.definitions],
        "examples": sorted(parsed.examples),
        "ok": parsed.ok,
    }


class SaveRequest(BaseModel):
    source: str
    # The sha the editor opened the file with; required to overwrite.
    expected_sha: str | None = None
    create: bool = False


@router.put("/api/library/files/{name:path}")
def save_library_file(name: str, req: SaveRequest) -> dict[str, Any]:
    try:
        f = store.save_file(name, req.source, expected_sha=req.expected_sha, create=req.create)
    except store.LibraryError as e:
        raise _fail(e)
    return {**file_summary(f), "source": f.source}


@router.delete("/api/library/files/{name:path}/blocks/{block_id}")
def delete_library_block(name: str, block_id: str) -> dict[str, Any]:
    try:
        f = store.delete_block(name, block_id)
    except store.LibraryError as e:
        raise _fail(e)
    return file_summary(f)


@router.delete("/api/library/files/{name:path}")
def delete_library_file(name: str, expected_sha: str | None = None) -> dict[str, str]:
    try:
        store.delete_file(name, expected_sha=expected_sha)
    except store.LibraryError as e:
        raise _fail(e)
    return {"deleted": name}


class TryRequest(BaseModel):
    # Run these blocks' _example_ self-checks (all, when omitted)...
    blocks: list[str] | None = None
    # ...or call one block on a canvas block's current (sampled) output.
    block: str | None = None
    input_block: str | None = None
    input_port: str | None = None
    params: dict[str, Any] = {}
    # Unsaved editor text to try instead of the file on disk.
    source: str | None = None


def _value_preview(value: Any) -> Any:
    import polars as pl

    if isinstance(value, pl.DataFrame):
        return {
            "type": "dataframe",
            "rows": value.height,
            "columns": [{"name": c, "dtype": str(t)} for c, t in value.schema.items()],
            "head": json.loads(value.head(20).write_json()),
        }
    if isinstance(value, (bytes, bytearray)):
        return {"type": "image", "bytes": len(value)}
    try:
        return {"type": "value", "value": json.loads(json.dumps(value, default=str))}
    except (TypeError, ValueError):
        return {"type": "value", "value": repr(value)[:2000]}


@router.post("/api/library/files/{name:path}/try")
def try_library_file(name: str, req: TryRequest) -> dict[str, Any]:
    """Try it: run the file's examples, or call one block on a canvas
    block's output. Runs in a separate process with a timeout; the result
    is shown, never sent anywhere else."""
    if req.source is not None:
        source = req.source
    else:
        try:
            source = store.get_file(name).source
        except store.LibraryError as e:
            raise HTTPException(404, str(e))
    if req.block is None:
        return {"examples": bl.run_examples(source, name, req.blocks)}
    parsed = bl.parse_source(source, name)
    d = next((x for x in parsed.definitions if x.id == req.block), None)
    if d is None:
        raise HTTPException(400, f"{name} has no block {req.block!r}")
    kwargs: dict[str, Any] = dict(req.params)
    if req.input_block is not None:
        session = _session()
        st = session.runner.state.get(req.input_block)
        entry = session.runner.cache.get(st.last_successful_key) if st and st.last_successful_key else None
        if entry is None:
            raise HTTPException(400, "run the input block first -- it has no output yet")
        value = entry.outputs.get(req.input_port or next(iter(entry.outputs), ""))
        port = next((p.name for p in d.inputs), None)
        if port is None:
            raise HTTPException(400, f"{d.id} has no input port to feed")
        kwargs[port] = value.data if isinstance(value, DataFramePacket) else value
    ok, result = bl.try_call(d, kwargs)
    if not ok:
        return {"ok": False, "error": result}
    outputs = result if isinstance(result, tuple) else (result,)
    names = [p.name for p in d.outputs] or ["result"]
    return {"ok": True, "outputs": {n: _value_preview(v) for n, v in zip(names, outputs)}}


@router.get("/api/library/export")
def export_library(files: str | None = None, file: str | None = None, blocks: str | None = None) -> Response:
    """`?file=a.py&blocks=x,y`: just those blocks (and their helpers) as
    one .py; `?file=a.py`: that file; `?files=a.py,b.py` (or nothing: the
    whole library): a .zip."""
    try:
        if file is not None:
            ids = blocks.split(",") if blocks else None
            text = store.extract(file, ids) if ids else store.get_file(file).source
            filename = f"{ids[0]}.py" if ids and len(ids) == 1 else file.split("/")[-1]
            return Response(text, media_type="text/x-python", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
        names = files.split(",") if files else [f.name for f in store.list_files() if not f.is_draft]
        data = store.export_zip(names)
    except store.LibraryError as e:
        raise _fail(e)
    return Response(data, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="blocks.zip"'})


class Upload(BaseModel):
    filename: str
    content_base64: str


def _read_upload(up: Upload) -> list[tuple[str, str]]:
    try:
        data = base64.b64decode(up.content_base64, validate=True)
    except ValueError:
        raise HTTPException(400, "content_base64 isn't valid base64")
    try:
        return store.read_upload(up.filename, data)
    except store.LibraryError as e:
        raise _fail(e)


@router.post("/api/library/import/preview")
def import_preview(up: Upload) -> dict[str, Any]:
    """What an upload holds, per file and block -- validated from the
    literal alone; nothing is defined or run."""
    try:
        return {"files": store.import_preview(_read_upload(up))}
    except store.LibraryError as e:
        raise _fail(e)


class ImportRequest(Upload):
    # "<file>:<block id>" -> "add" | "overwrite" | "rename" | "skip"
    actions: dict[str, str] = {}


@router.post("/api/library/import")
def import_commit(req: ImportRequest) -> dict[str, Any]:
    try:
        written = store.import_commit(_read_upload(req), req.actions)
    except store.LibraryError as e:
        raise _fail(e)
    return {"imported": written}


class KeepRequest(BaseModel):
    file: str
    block: str
    to: str


@router.post("/api/library/drafts/keep")
def keep_draft(req: KeepRequest) -> dict[str, Any]:
    """Move an AI-drafted block into a regular library file. Placed blocks
    keep running their embedded copy and now point at the kept file."""
    try:
        f = store.move_block(req.file, req.block, req.to)
    except store.LibraryError as e:
        raise _fail(e)
    session = _session()
    d = f.definition(req.block)
    with session.edit():
        for b in session.graph.blocks.values():
            if (b.library_ref or {}).get("file") == req.file and b.category == req.block:
                b.library_ref = store.library_ref(f, d) if d is not None and d.sha == b.resolve().sha else {**b.library_ref, "file": f.name}
    return file_summary(f)


class DiscardRequest(BaseModel):
    file: str


@router.post("/api/library/drafts/discard")
def discard_draft(req: DiscardRequest) -> dict[str, str]:
    """Delete an AI build's drafts file. Placed blocks embed their
    definition, so the model keeps working."""
    try:
        f = store.get_file(req.file)
        if not f.is_draft:
            raise store.LibraryError(f"{req.file} isn't an AI drafts file")
        store.delete_file(req.file)
    except store.LibraryError as e:
        raise _fail(e)
    return {"deleted": req.file}


# ---- placed blocks: update / accept / save to library -------------------


@router.get("/api/blocks/{block_id}/update_preview")
def update_preview(block_id: str) -> dict[str, Any]:
    session = _session()
    if block_id not in session.graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")
    try:
        return library_updates.preview(session.graph, block_id)
    except (store.LibraryError, KeyError) as e:
        raise HTTPException(400, str(e))


@router.post("/api/blocks/{block_id}/update")
def update_block_definition(block_id: str) -> dict[str, Any]:
    """Move the block to its library's current version (or, for a
    built-in, accept the installed one), as previewed."""
    from . import api

    session = _session()
    if block_id not in session.graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")
    try:
        with session.edit():
            summary = library_updates.apply(session, block_id)
    except (store.LibraryError, KeyError) as e:
        raise HTTPException(400, str(e))
    return {"applied": summary, "block": api._block_out(block_id)}
