from __future__ import annotations

import functools
import inspect
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import blocks as _blocks_pkg  # noqa: F401 -- populates BLOCK_REGISTRY
from .blocks import library as _library  # noqa: F401
from .blocks import modelling as _modelling  # noqa: F401
from .blocks import stat_tests as _stat_tests  # noqa: F401
from .blocks.base import BLOCK_REGISTRY
from .compiler import CompileError, compile_graph
from .llm import ColumnInfo, DraftContext, LLM_PROVIDER_REGISTRY, LLMProvider, get_provider
from .llm import claude_cli_provider as _claude_cli_provider
from .llm import gemini_provider as _gemini_provider
from .llm import lmstudio_provider as _lmstudio_provider
from .llm import openai_provider as _openai_provider
from .llm.settings import LLMSettingsStore
from .packet import DataFramePacket
from .session import ProjectSession, wire_is_valid

try:
    from .llm import anthropic_provider as _anthropic_provider
except ImportError:
    _anthropic_provider = None

# Load a local .env file (if present) into the process environment before
# anything reads provider API keys/config from os.environ -- lets
# ANTHROPIC_API_KEY etc. live in a gitignored .env instead of being
# exported by hand or typed into Settings every run. A missing .env is not
# an error; existing environment variables always take precedence.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

app = FastAPI(title="Model-Maker API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION = ProjectSession()
LLM_SETTINGS = LLMSettingsStore()

# Static curated list -- the `claude` CLI has no scriptable "list models"
# command, so this is what the model dropdown offers for the claude_cli
# provider instead of a live query.
CLAUDE_CLI_KNOWN_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5-1",
    "claude-haiku-4-5-20251001",
]

# Where Save/Load default to when no project path is already known (see
# Toolbar.tsx) -- a subdirectory next to wherever the server runs. Not
# created at import time (nothing should touch the filesystem just from
# importing this module, e.g. under test) -- see project_default_dir below,
# which creates it on first request instead.
DEFAULT_PROJECTS_DIR = Path(os.environ.get("MODELMAKER_PROJECTS_DIR", "./projects")).expanduser().resolve()


# ---- schemas -----------------------------------------------------------


class PortSpecOut(BaseModel):
    name: str
    type: str
    required: bool = True


class BlockCreate(BaseModel):
    category: str
    block_type: str | None = None
    name: str | None = None
    lane: str | None = None
    x: float = 0
    y: float = 0
    params: dict[str, Any] = {}
    code: str | None = None
    inputs: list[dict] | None = None
    outputs: list[dict] | None = None
    metadata_transform: dict[str, Any] | None = None


class BlockUpdate(BaseModel):
    name: str | None = None
    lane: str | None = None
    position: dict[str, float] | None = None
    params: dict[str, Any] | None = None
    code: str | None = None
    metadata_transform: dict[str, Any] | None = None


class ColumnRoleUpdate(BaseModel):
    column: str
    role: str


class WireCreate(BaseModel):
    from_block: str
    from_port: str
    to_block: str
    to_port: str


class PortNameUpdate(BaseModel):
    port: str
    name: str | None = None


class LaneUpsert(BaseModel):
    id: str
    name: str
    order: int
    height: float | None = None


class LoadRequest(BaseModel):
    path: str


class SaveRequest(BaseModel):
    path: str | None = None


class CompileRequest(BaseModel):
    output_blocks: list[str] | None = None
    strict: bool = True


class DraftRequest(BaseModel):
    instruction: str = ""
    provider: str | None = None


class LLMSettingsUpdate(BaseModel):
    active_provider: str | None = None
    # Global toggle for including the Polars API reference in the system
    # prompt -- applies across every provider (see DraftContext.include_reference).
    include_reference: bool | None = None
    # Per-provider fields, e.g. {"lmstudio": {"model": "...", "base_url": "..."}}
    # or {"openai": {"model": "...", "base_url": "...", "api_key": "sk-..."}}.
    # Unrecognized keys for a given provider are ignored by get_provider. An
    # explicit null for a field (e.g. {"openai": {"api_key": null}}) clears
    # a previously-set override -- see LLMSettingsStore.update. API keys are
    # write-only: GET/PUT never echo a raw key back, only whether one is set
    # (see _redacted_provider_settings).
    settings: dict[str, dict[str, Any]] | None = None


# ---- helpers -------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _safe_getsource(fn) -> str | None:
    """inspect.getsource(), but None instead of raising for a function whose
    source isn't available (e.g. defined in a REPL or compiled extension).
    Cached since registry block functions are fixed at import time, and this
    is called on every /api/graph and /api/blocks/{id} poll."""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return None


def _source_for_block(block) -> str | None:
    """Read-only source for the block's underlying Python function. For
    custom (is_custom) blocks the editable source is already exposed via
    `code`; for registry blocks it lives in BLOCK_REGISTRY as a plain
    function, so pull it with inspect for display purposes only."""
    if block.is_custom:
        return None
    spec = BLOCK_REGISTRY.get(block.category)
    if spec is None:
        return None
    return _safe_getsource(spec.fn)


def _block_out(block_id: str) -> dict[str, Any]:
    block = SESSION.graph.blocks[block_id]
    st = SESSION.runner.state.get(block_id)
    return {
        "id": block.id,
        "block_type": block.block_type,
        "is_custom": block.is_custom,
        "category": block.category,
        "name": block.name,
        "lane": block.lane,
        "position": {"x": block.position.x, "y": block.position.y},
        "code_version": block.code_version,
        "params": block.params,
        "code": block.code,
        "source": _source_for_block(block),
        "metadata_transform": block.metadata_transform,
        "inputs": [asdict(p) for p in block.inputs],
        "outputs": [asdict(p) for p in block.outputs],
        "port_names": block.port_names,
        "status": SESSION.runner.status(block_id),
        "last_error": st.last_error if st else None,
        "last_successful_read_at": st.last_successful_read_at if st else None,
        "last_attempt_at": st.last_attempt_at if st else None,
    }


def _graph_out() -> dict[str, Any]:
    return {
        "project_name": SESSION.project_name,
        "project_path": str(SESSION.project_path) if SESSION.project_path else None,
        "lanes": {lid: {"name": l.name, "order": l.order, "height": l.height} for lid, l in SESSION.graph.lanes.items()},
        "blocks": {bid: _block_out(bid) for bid in SESSION.graph.blocks},
        "wires": {
            wid: {
                "from_block": w.from_block,
                "from_port": w.from_port,
                "to_block": w.to_block,
                "to_port": w.to_port,
                "valid": wire_is_valid(SESSION.graph, w),
            }
            for wid, w in SESSION.graph.wires.items()
        },
    }


def _packet_preview(packet: DataFramePacket, rows: int, with_summary: bool) -> dict[str, Any]:
    if with_summary:
        packet = packet.compute_summary()
    columns = [
        {
            "name": name,
            "dtype": meta.dtype,
            "role": meta.role.value if hasattr(meta.role, "value") else meta.role,
            "description": meta.description,
        }
        for name, meta in packet.schema_meta.items()
    ]
    summary = None
    if with_summary and packet.summary:
        summary = {
            name: {
                "count": s.count,
                "null_count": s.null_count,
                "n_unique": s.n_unique,
                "mean": s.mean,
                "std": s.std,
                "min": s.min,
                "max": s.max,
            }
            for name, s in packet.summary.items()
        }
    return {
        "columns": columns,
        "rows": packet.data.head(rows).to_dicts(),
        "row_count": packet.data.height,
        "lineage": packet.lineage,
        "summary": summary,
    }


def _input_schema_for_block(block_id: str) -> dict[str, list[ColumnInfo]]:
    result: dict[str, list[ColumnInfo]] = {p.name: [] for p in SESSION.graph.blocks[block_id].inputs}
    for port, wire in SESSION.graph.input_wires(block_id).items():
        result.setdefault(port, [])
        if SESSION.runner.status(wire.from_block) not in ("green", "orange"):
            continue
        st = SESSION.runner.state.get(wire.from_block)
        if st is None or st.last_successful_key is None:
            continue
        entry = SESSION.runner.cache.get(st.last_successful_key)
        if entry is None:
            continue
        packet = entry.outputs.get(wire.from_port)
        if not isinstance(packet, DataFramePacket):
            continue
        result[port] = [
            ColumnInfo(name=name, dtype=meta.dtype, role=meta.role.value if hasattr(meta.role, "value") else meta.role)
            for name, meta in packet.schema_meta.items()
        ]
    return result


# ---- routes ----------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/registry")
def registry() -> list[dict[str, Any]]:
    return [
        {
            "category": spec.category,
            "block_type": spec.block_type,
            "group": spec.group or spec.block_type,
            "display_name": spec.display_name,
            "inputs": [asdict(p) for p in spec.inputs],
            "outputs": [asdict(p) for p in spec.outputs],
        }
        for spec in BLOCK_REGISTRY.values()
    ]


@app.get("/api/graph")
def get_graph() -> dict[str, Any]:
    return _graph_out()


@app.get("/api/browse")
def browse(path: str | None = None, ext: str | None = None) -> dict[str, Any]:
    """List a server-side directory so the UI can offer a file picker for
    params like read_csv's `path` -- the frontend runs in a regular browser,
    which can't hand back an absolute filesystem path from <input type=file>,
    so this stands in for a native file dialog. Local dev tool, run by the
    user against their own machine, so no extra path sandboxing beyond
    resolving `..`/symlinks."""
    base = Path(path).expanduser().resolve() if path else Path.cwd()
    if not base.exists():
        base = Path.cwd()
    if base.is_file():
        base = base.parent

    entries = []
    try:
        for e in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if e.name.startswith("."):
                continue
            if not e.is_dir() and ext and not e.name.lower().endswith(ext.lower()):
                continue
            entries.append({"name": e.name, "path": str(e), "is_dir": e.is_dir()})
    except PermissionError:
        pass

    parent = base.parent
    return {
        "path": str(base),
        "parent": str(parent) if parent != base else None,
        "entries": entries,
    }


@app.get("/api/project/default_dir")
def project_default_dir() -> dict[str, str]:
    DEFAULT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return {"path": str(DEFAULT_PROJECTS_DIR)}


@app.post("/api/project/load")
def load_project_ep(req: LoadRequest) -> dict[str, Any]:
    try:
        SESSION.load(Path(req.path))
    except FileNotFoundError:
        raise HTTPException(404, f"project file not found: {req.path}")
    return _graph_out()


@app.post("/api/project/save")
def save_project_ep(req: SaveRequest) -> dict[str, Any]:
    try:
        path = SESSION.save(Path(req.path) if req.path else None)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"path": str(path)}


@app.post("/api/blocks")
def create_block(req: BlockCreate) -> dict[str, Any]:
    try:
        block = SESSION.add_block(
            category=req.category,
            block_type=req.block_type,
            name=req.name,
            lane=req.lane,
            x=req.x,
            y=req.y,
            params=req.params,
            code=req.code,
            inputs=req.inputs,
            outputs=req.outputs,
            metadata_transform=req.metadata_transform,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _block_out(block.id)


@app.patch("/api/blocks/{block_id}")
def update_block(block_id: str, req: BlockUpdate) -> dict[str, Any]:
    if block_id not in SESSION.graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")
    SESSION.update_block(block_id, **req.model_dump(exclude_unset=True))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/column_role")
def set_column_role(block_id: str, req: ColumnRoleUpdate) -> dict[str, Any]:
    """Hand-tag a column's role (id/target/weight/feature/date/segment/
    excluded, or "unassigned" to clear it) on this block -- see
    ProjectSession.set_column_role. Changing it is a normal block edit: it
    changes the block's cache key, so this block and everything downstream
    go stale like any other param change, rather than needing separate
    invalidation."""
    _require_block(block_id)
    try:
        SESSION.set_column_role(block_id, req.column, req.role)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _block_out(block_id)


@app.get("/api/blocks/{block_id}/input_schema")
def input_schema_ep(block_id: str) -> dict[str, list[dict[str, str]]]:
    """Column names/dtypes/roles available on each of a block's input ports
    (empty for a port whose upstream hasn't produced output yet) -- lets the
    UI offer column dropdowns for params instead of free-text entry."""
    _require_block(block_id)
    return {port: [asdict(c) for c in cols] for port, cols in _input_schema_for_block(block_id).items()}


@app.delete("/api/blocks/{block_id}")
def delete_block(block_id: str) -> dict[str, str]:
    SESSION.delete_block(block_id)
    return {"deleted": block_id}


def _get_cached_output(block_id: str, port: str | None, want_type: str | None = None) -> Any:
    """Look up a block's cached output value on the given port (defaulting
    to its declared port of `want_type`, or its first output port). Raises
    the appropriate HTTPException if the block isn't runnable, has no
    matching port, or nothing is cached yet."""
    if block_id not in SESSION.graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")
    status = SESSION.runner.status(block_id)
    if status not in ("green", "orange"):
        raise HTTPException(409, f"block '{block_id}' has no output to view (status={status})")
    st = SESSION.runner.state[block_id]
    entry = SESSION.runner.cache.get(st.last_successful_key)  # type: ignore[arg-type]
    if entry is None:
        raise HTTPException(409, "cached output missing")
    block = SESSION.graph.blocks[block_id]
    if port is None and want_type is not None:
        port = next((p.name for p in block.outputs if p.type == want_type), None)
    port = port or (block.outputs[0].name if block.outputs else None)
    if port is None or port not in entry.outputs:
        raise HTTPException(404, f"no such output port: {port}")
    return entry.outputs[port]


@app.get("/api/blocks/{block_id}/preview")
def preview_block(block_id: str, port: str | None = None, rows: int = 20, summary: bool = False) -> dict[str, Any]:
    value = _get_cached_output(block_id, port)
    if not isinstance(value, DataFramePacket):
        raise HTTPException(400, "output port is not a dataframe")
    return _packet_preview(value, rows=rows, with_summary=summary)


@app.get("/api/blocks/{block_id}/image")
def block_image(block_id: str, port: str | None = None) -> Response:
    value = _get_cached_output(block_id, port, want_type="image")
    if not isinstance(value, (bytes, bytearray)):
        raise HTTPException(400, "output port is not an image")
    return Response(content=bytes(value), media_type="image/png")


@app.get("/api/blocks/{block_id}/value")
def block_value(block_id: str, port: str | None = None) -> Any:
    """Raw JSON-shaped output (a scalar_metric or model port's dict, or any
    other plain value) for ports that aren't a dataframe or an image --
    those have their own endpoints above."""
    value = _get_cached_output(block_id, port)
    if isinstance(value, DataFramePacket):
        raise HTTPException(400, "output port is a dataframe -- use /preview")
    if isinstance(value, (bytes, bytearray)):
        raise HTTPException(400, "output port is binary (e.g. an image) -- use /image")
    return value


@app.post("/api/wires")
def create_wire(req: WireCreate) -> dict[str, Any]:
    for bid in (req.from_block, req.to_block):
        if bid not in SESSION.graph.blocks:
            raise HTTPException(404, f"no such block: {bid}")
    try:
        wire = SESSION.add_wire(req.from_block, req.from_port, req.to_block, req.to_port)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "id": wire.id,
        "from_block": wire.from_block,
        "from_port": wire.from_port,
        "to_block": wire.to_block,
        "to_port": wire.to_port,
        "valid": wire_is_valid(SESSION.graph, wire),
    }


@app.delete("/api/wires/{wire_id}")
def delete_wire(wire_id: str) -> dict[str, str]:
    SESSION.delete_wire(wire_id)
    return {"deleted": wire_id}


@app.patch("/api/blocks/{block_id}/port_name")
def rename_port(block_id: str, req: PortNameUpdate) -> dict[str, Any]:
    """Name (or clear the name of) the data on one of this block's output
    ports -- see BlockInstance.port_names. Set by clicking that port's data
    in the UI (PortInspector); used, when set, as the compiled script's
    variable name for it (see compiler.compile_graph)."""
    _require_block(block_id)
    try:
        SESSION.rename_port(block_id, req.port, req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _block_out(block_id)


@app.put("/api/lanes")
def upsert_lane(req: LaneUpsert) -> dict[str, Any]:
    SESSION.set_lane(req.id, req.name, req.order, height=req.height)
    return _graph_out()["lanes"]


@app.delete("/api/lanes/{lane_id}")
def delete_lane(lane_id: str) -> dict[str, Any]:
    SESSION.delete_lane(lane_id)
    return _graph_out()["lanes"]


def _require_block(block_id: str) -> None:
    if block_id not in SESSION.graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")


@app.post("/api/blocks/{block_id}/run")
def run_block_ep(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    try:
        SESSION.runner.run_block(block_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/run_to_here")
def run_to_here_ep(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    try:
        SESSION.runner.run_to_here(block_id)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/refresh")
def refresh_ep(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    if SESSION.graph.blocks[block_id].block_type != "input":
        raise HTTPException(400, "refresh is only valid for input blocks")
    try:
        SESSION.runner.refresh(block_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/check_changes")
def check_changes(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    return {"block_id": block_id, "changed": SESSION.runner.check_for_changes(block_id)}


@app.post("/api/run_all")
def run_all() -> dict[str, str]:
    try:
        return SESSION.runner.run_all()
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/force_run_all")
def force_run_all() -> dict[str, str]:
    try:
        return SESSION.runner.force_run_all()
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/refresh_all")
def refresh_all() -> dict[str, str]:
    return SESSION.runner.refresh_all()


@app.post("/api/check_all_sources")
def check_all_sources() -> dict[str, bool]:
    return SESSION.runner.check_all_sources()


@app.post("/api/compile")
def compile_ep(req: CompileRequest) -> dict[str, str]:
    try:
        source = compile_graph(
            SESSION.graph, runner=SESSION.runner, output_blocks=req.output_blocks, strict=req.strict
        )
    except (CompileError, ValueError) as e:
        raise HTTPException(409, str(e))
    return {"source": source}


# Env vars (beyond the per-provider override stored in LLM_SETTINGS) each
# key-requiring provider's API key can fall back to, in priority order --
# mirrors what each provider's own __init__ checks, used only to report
# *whether* a key is available, never its value.
_API_KEY_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["MODELMAKER_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "gemini": ["MODELMAKER_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


def _resolve_api_key(provider_name: str) -> str | None:
    """The API key that would actually be used for this provider right now
    (a Settings override, else the first set env var) -- never exposed to
    the client, only used server-side to call the provider or check
    liveness (e.g. for /api/llm/models)."""
    override = LLM_SETTINGS.for_provider(provider_name).get("api_key")
    if override:
        return override
    for var in _API_KEY_ENV_VARS.get(provider_name, []):
        value = os.environ.get(var)
        if value:
            return value
    return None


def _redacted_provider_settings(provider_name: str, **fields: Any) -> dict[str, Any]:
    """Build one provider's entry for GET/PUT /api/llm/settings: the given
    resolved (non-secret) fields, plus an api_key_set/api_key_source pair
    that reports *whether* a key is configured and where it came from --
    never the key itself. This is the only place a stored key's presence is
    surfaced to the client."""
    result: dict[str, Any] = dict(fields)
    if provider_name in _API_KEY_ENV_VARS:
        override = LLM_SETTINGS.for_provider(provider_name).get("api_key")
        if override:
            result["api_key_set"] = True
            result["api_key_source"] = "override"
        else:
            env_value = next(
                (os.environ.get(var) for var in _API_KEY_ENV_VARS[provider_name] if os.environ.get(var)), None
            )
            result["api_key_set"] = bool(env_value)
            result["api_key_source"] = "env" if env_value else None
    return result


def _effective_llm_settings() -> dict[str, Any]:
    """The Settings panel's view of LLM config: every field resolved to what
    would actually be used right now (a UI override, else the env var each
    provider itself falls back to, else its hardcoded default) -- so text
    fields and checkboxes always show a real value instead of blank. API
    keys are the exception: only their presence/source is reported, never
    the value (see _redacted_provider_settings)."""
    lmstudio_override = LLM_SETTINGS.for_provider("lmstudio")
    claude_cli_override = LLM_SETTINGS.for_provider("claude_cli")
    openai_override = LLM_SETTINGS.for_provider("openai")
    gemini_override = LLM_SETTINGS.for_provider("gemini")
    settings: dict[str, Any] = {
        "lmstudio": _redacted_provider_settings(
            "lmstudio",
            base_url=lmstudio_override.get("base_url")
            or os.environ.get("MODELMAKER_LLM_BASE_URL", _lmstudio_provider.DEFAULT_BASE_URL),
            model=lmstudio_override.get("model") or os.environ.get("MODELMAKER_LLM_MODEL"),
        ),
        "claude_cli": _redacted_provider_settings(
            "claude_cli",
            model=claude_cli_override.get("model")
            or os.environ.get("MODELMAKER_LLM_MODEL")
            or _claude_cli_provider.DEFAULT_MODEL,
        ),
        "openai": _redacted_provider_settings(
            "openai",
            base_url=openai_override.get("base_url")
            or os.environ.get("MODELMAKER_OPENAI_BASE_URL", _openai_provider.DEFAULT_BASE_URL),
            model=openai_override.get("model") or os.environ.get("MODELMAKER_LLM_MODEL"),
        ),
        "gemini": _redacted_provider_settings(
            "gemini",
            model=gemini_override.get("model") or os.environ.get("MODELMAKER_LLM_MODEL"),
        ),
    }
    if _anthropic_provider is not None:
        anthropic_override = LLM_SETTINGS.for_provider("anthropic")
        settings["anthropic"] = _redacted_provider_settings(
            "anthropic",
            model=anthropic_override.get("model")
            or os.environ.get("MODELMAKER_LLM_MODEL")
            or _anthropic_provider.DEFAULT_MODEL,
        )
    return {
        "providers": sorted(LLM_PROVIDER_REGISTRY),
        "active_provider": LLM_SETTINGS.active_provider or os.environ.get("MODELMAKER_LLM_PROVIDER", "claude_cli"),
        # Tri-state: null means "each provider's own default" (on for LM
        # Studio, off elsewhere) rather than an explicit choice.
        "include_reference": LLM_SETTINGS.include_reference,
        "settings": settings,
    }


@app.get("/api/llm/settings")
def get_llm_settings() -> dict[str, Any]:
    return _effective_llm_settings()


@app.put("/api/llm/settings")
def update_llm_settings(req: LLMSettingsUpdate) -> dict[str, Any]:
    if req.active_provider is not None and req.active_provider not in LLM_PROVIDER_REGISTRY:
        raise HTTPException(400, f"unknown LLM provider: {req.active_provider}")
    LLM_SETTINGS.update(req.active_provider, req.include_reference, req.settings)
    return _effective_llm_settings()


@app.get("/api/llm/models")
def list_llm_models(provider: str, base_url: str | None = None) -> dict[str, list[str]]:
    """Query a provider for the models it currently has available -- used by
    the Settings panel's "Fetch models" so the user can pick one instead of
    typing an id from memory. `base_url` (LM Studio and openai only) lets
    the panel query a not-yet-saved address before committing it via
    PUT /llm/settings."""
    if provider == "lmstudio":
        effective_base_url = base_url or _effective_llm_settings()["settings"]["lmstudio"]["base_url"]
        try:
            return {"models": _lmstudio_provider.list_models(effective_base_url)}
        except RuntimeError as e:
            raise HTTPException(502, str(e))
    if provider == "claude_cli":
        return {"models": CLAUDE_CLI_KNOWN_MODELS}
    if provider == "anthropic":
        if _anthropic_provider is None:
            raise HTTPException(400, "the anthropic provider is unavailable -- install the `anthropic` package")
        api_key = _resolve_api_key("anthropic")
        try:
            client = _anthropic_provider.anthropic.Anthropic(api_key=api_key) if api_key else _anthropic_provider.anthropic.Anthropic()
            return {"models": [m.id for m in client.models.list()]}
        except Exception as e:
            raise HTTPException(502, f"could not list Anthropic models: {e}")
    if provider == "openai":
        api_key = _resolve_api_key("openai")
        if not api_key:
            raise HTTPException(400, "no OpenAI API key configured -- set one in Settings first")
        effective_base_url = base_url or _effective_llm_settings()["settings"]["openai"]["base_url"]
        try:
            return {"models": _openai_provider.list_models(effective_base_url, api_key)}
        except RuntimeError as e:
            raise HTTPException(502, str(e))
    if provider == "gemini":
        api_key = _resolve_api_key("gemini")
        if not api_key:
            raise HTTPException(400, "no Gemini API key configured -- set one in Settings first")
        try:
            return {"models": _gemini_provider.list_models(_gemini_provider.DEFAULT_BASE_URL, api_key)}
        except RuntimeError as e:
            raise HTTPException(502, str(e))
    if provider == "stub":
        return {"models": []}
    raise HTTPException(400, f"unknown LLM provider: {provider}")


def _provider_for(name: str | None) -> LLMProvider:
    """Resolve a provider for a draft/suggest_fix call: an explicit request
    override if given, else the Settings panel's active provider, else the
    env var default -- configured with whatever overrides (model, base_url)
    are on file for it."""
    provider_name = name or LLM_SETTINGS.active_provider or os.environ.get("MODELMAKER_LLM_PROVIDER", "claude_cli")
    overrides = dict(LLM_SETTINGS.for_provider(provider_name))
    if LLM_SETTINGS.include_reference is not None:
        overrides.setdefault("include_reference", LLM_SETTINGS.include_reference)
    return get_provider(provider_name, **overrides)


def _draft_context_for_block(block, block_id: str, instruction: str, error: str | None = None) -> DraftContext:
    """Build the DraftContext for either flavor of block: custom (is_custom)
    blocks get the original "author full code" mode; registry blocks have
    fixed code, so the model can only choose values for their existing
    parameters ("params_only" mode) -- this is what makes "Draft with AI"
    available on every block, not just custom ones, without letting the
    model touch fixed code."""
    input_ports = _input_schema_for_block(block_id)
    param_names = list(block.params)
    include_reference = LLM_SETTINGS.include_reference if LLM_SETTINGS.include_reference is not None else False

    if block.is_custom:
        return DraftContext(
            instruction=instruction,
            function_name=block.category,
            input_ports=input_ports,
            param_names=param_names,
            existing_code=block.code,
            error=error,
            include_reference=include_reference,
        )

    spec = BLOCK_REGISTRY.get(block.category)
    if spec is None:
        raise HTTPException(400, f"unknown block category: {block.category}")
    fixed_source = _safe_getsource(spec.fn)
    return DraftContext(
        instruction=instruction,
        function_name=block.category,
        input_ports=input_ports,
        param_names=param_names,
        fixed_source=fixed_source,
        error=error,
        mode="params_only",
        include_reference=include_reference,
    )


@app.post("/api/blocks/{block_id}/draft")
def draft_block(block_id: str, req: DraftRequest) -> dict[str, Any]:
    """Draft (or redraft) a block from a natural-language instruction.
    llm_authored blocks get a full function body proposal; every other
    block type gets suggested values for its existing parameters only,
    since their code is fixed. Never applied automatically -- returns the
    proposal for the caller to review and save via PATCH /api/blocks/{id}."""
    _require_block(block_id)
    block = SESSION.graph.blocks[block_id]
    if not req.instruction.strip():
        raise HTTPException(400, "instruction is required")

    ctx = _draft_context_for_block(block, block_id, req.instruction)
    try:
        provider = _provider_for(req.provider)
        result = provider.draft(ctx)
    except Exception as e:
        raise HTTPException(502, f"LLM draft failed: {e}")
    return {
        "code": result.code,
        "metadata_transform": result.metadata_transform,
        "params": result.params,
        "explanation": result.explanation,
    }


@app.post("/api/blocks/{block_id}/suggest_fix")
def suggest_fix(block_id: str, req: DraftRequest = DraftRequest()) -> dict[str, Any]:
    """AI-assisted fix for a red block: sends the code (or fixed source),
    params, actual error, and input schema; returns a proposed diff, never
    auto-applied."""
    _require_block(block_id)
    block = SESSION.graph.blocks[block_id]
    st = SESSION.runner.state.get(block_id)
    if st is None or not st.last_error:
        raise HTTPException(400, "block has no recorded error to fix")

    ctx = _draft_context_for_block(block, block_id, req.instruction.strip() or "Fix the error.", error=st.last_error)
    try:
        provider = _provider_for(req.provider)
        result = provider.draft(ctx)
    except Exception as e:
        raise HTTPException(502, f"LLM suggest_fix failed: {e}")
    return {
        "code": result.code,
        "metadata_transform": result.metadata_transform,
        "params": result.params,
        "explanation": result.explanation,
    }


# Serve the built frontend (from `npm run build` in frontend/, which emits
# into this directory) if present, so `modelmaker-api` alone can serve the
# whole app on one port. Mounted last so it never shadows the /api routes
# above. Absent in a dev checkout that hasn't built the frontend -- run the
# Vite dev server separately in that case.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="frontend")


def main() -> None:
    """Entry point for the `modelmaker-api` console script."""
    import uvicorn

    uvicorn.run(
        "modelmaker.api:app",
        host=os.environ.get("MODELMAKER_HOST", "127.0.0.1"),
        port=int(os.environ.get("MODELMAKER_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
