from __future__ import annotations

import functools
import inspect
import os
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import blocks as _blocks_pkg  # noqa: F401 -- populates BLOCK_REGISTRY
from . import gitops, project
from .blocks import data_quality as _data_quality  # noqa: F401
from .blocks import feature_analysis as _feature_analysis  # noqa: F401
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

# Names set via PUT /api/env_vars (see below) -- lets the Settings-style
# "is this set, and from where" distinction (env vs. override) extend to
# arbitrary env vars too, e.g. a Read SQL block's connection_env. The value
# itself lives only in os.environ, exactly where a block reading it (live
# or compiled) already looks, and is never stored or echoed anywhere else.
# Maps name -> whatever os.environ held right before the first override (or
# None if it was unset), so DELETE can restore that instead of just wiping
# a name that happened to already be set from the shell/.env.
_ENV_VAR_OVERRIDES: dict[str, str | None] = {}
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Run orchestration: at most one run (a single block, a cascade, or a full
# sweep) is active at a time, executed on a background thread so a run that
# genuinely takes a while -- above all a heavily-grouped block, the case
# this whole mechanism exists to keep safe -- doesn't block the HTTP
# request indefinitely: the frontend polls GET /api/graph (whose per-block
# `status` already reflects "running", see Runner.status) to watch it
# progress, and POST /api/run/cancel to stop it. _start_background_run
# joins the thread for up to `wait_seconds` before returning, though, so
# the common case (a normal-sized run finishes in well under that) still
# gets a response that reflects the final state, exactly as if the call
# had been synchronous -- only a run that's still going after the wait
# comes back as "running" for the client to poll. The isolation that
# actually matters -- one subprocess per block call, so a crash or a
# runaway group_by can't take the server down -- is Runner's job (see
# runner.py) and applies either way.
_RUN_LOCK = threading.Lock()
_RUN_THREAD: threading.Thread | None = None
# A precondition failure (e.g. "Run" clicked on a block whose upstream
# isn't green yet) raises before any block state changes, so there's
# nothing in the graph's own status to show it happened -- stashed here,
# raised directly if the run finishes within the wait window (matching the
# old synchronous endpoints' behavior), and otherwise surfaced once via
# _graph_out(), which clears it on read.
_LAST_RUN_ERROR: str | None = None


def _start_background_run(fn: Callable[[], Any], wait_seconds: float = 20.0) -> Any:
    """Returns fn()'s return value if it finishes within wait_seconds (the
    common case -- callers that don't care, like the single-block run
    endpoints, can just ignore it and re-read block state), else None to
    mean "still running" (poll GET /api/graph)."""
    global _RUN_THREAD, _LAST_RUN_ERROR
    with _RUN_LOCK:
        if _RUN_THREAD is not None and _RUN_THREAD.is_alive():
            raise HTTPException(409, "a run is already in progress")
        _LAST_RUN_ERROR = None
        holder: dict[str, Any] = {}

        def _target() -> None:
            global _LAST_RUN_ERROR
            try:
                holder["result"] = fn()
            except Exception as e:  # noqa: BLE001 -- has no HTTP response to attach to once the wait below gives up
                _LAST_RUN_ERROR = f"{type(e).__name__}: {e}"

        thread = threading.Thread(target=_target, daemon=True)
        _RUN_THREAD = thread
        thread.start()
    thread.join(timeout=wait_seconds)
    if thread.is_alive():
        return None
    if _LAST_RUN_ERROR:
        err, _LAST_RUN_ERROR = _LAST_RUN_ERROR, None
        raise HTTPException(409, err)
    return holder.get("result")

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
    # Column to run this block once per distinct value of, instead of once
    # overall (see BlockInstance.group_by) -- like `lane`, an explicit null
    # in the request body clears it (session.update_block applies it
    # whenever the key is present at all, not only when non-None).
    group_by: str | None = None
    max_workers: int | None = None


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


class SampleModeUpdate(BaseModel):
    # None turns sample mode off; an int is the row cap applied to every
    # input block's output (see Runner.sample_rows).
    rows: int | None = None


class LoadRequest(BaseModel):
    # A project *folder* -- valid only if it holds project.PROJECT_FILENAME.
    path: str


class SaveRequest(BaseModel):
    # A project *folder* to save into (created if it doesn't exist yet);
    # None reuses whatever folder the session is already saved to.
    path: str | None = None


class GitRemoteRequest(BaseModel):
    url: str


class GitCommitRequest(BaseModel):
    message: str


class CompileRequest(BaseModel):
    output_blocks: list[str] | None = None
    strict: bool = True
    # Mirrors Runner.run_all_streaming for the compiled script itself (see
    # compile_graph's `stream` param): fuses whatever connected stretch of
    # compatible blocks it safely can into one polars lazy plan per group.
    # Default False keeps every existing "Compile" click's output unchanged.
    stream: bool = False


class DraftRequest(BaseModel):
    instruction: str = ""
    provider: str | None = None


class AnalyzeDataRequest(BaseModel):
    port: str | None = None
    provider: str | None = None


class SuggestNamesRequest(BaseModel):
    provider: str | None = None


class ArtifactRename(BaseModel):
    title: str


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
    status = SESSION.runner.status(block_id)
    return {
        # Why a stale block is stale (see Runner.stale_reason) -- only ever
        # asked for an orange block, since that's the state whose cause isn't
        # self-evident: grey has never run and red carries its error.
        "stale_reason": SESSION.runner.stale_reason(block_id) if status == "orange" else None,
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
        "group_by": block.group_by,
        "max_workers": block.max_workers,
        "status": status,
        "last_error": st.last_error if st else None,
        "last_successful_read_at": st.last_successful_read_at if st else None,
        "last_attempt_at": st.last_attempt_at if st else None,
        # A crude heartbeat while a block (or a whole fused streaming
        # group, for its every exit -- see Runner.running_elapsed) is
        # mid-dispatch -- polars gives no finer-grained progress for a
        # single collect() call, so this is "how long has this been
        # running", not "how far along is it".
        "running_seconds": SESSION.runner.running_elapsed(block_id) if status == "running" else None,
    }


def _graph_out() -> dict[str, Any]:
    global _LAST_RUN_ERROR
    run_error, _LAST_RUN_ERROR = _LAST_RUN_ERROR, None
    return {
        "project_name": SESSION.project_name,
        # The project *folder*, not the PROJECT_FILENAME inside it -- that's
        # the unit Save/Load/the Git panel all operate on; SESSION.project_path
        # itself stays the full file path internally (see project.py).
        "project_path": str(SESSION.project_path.parent) if SESSION.project_path else None,
        "dirty": SESSION.dirty,
        "can_undo": SESSION.can_undo,
        "can_redo": SESSION.can_redo,
        "sample_rows": SESSION.runner.sample_rows,
        # True only while a Run all/Force run all sweep is actually in
        # flight (see Runner._sweep) -- keeps the frontend polling through
        # the gaps between one block's dispatch ending and the next one
        # starting, so per-block status colors visibly progress through the
        # sweep instead of jumping straight from all-grey to done.
        "sweep_running": SESSION.runner.sweep_running,
        "lanes": {lid: {"name": l.name, "order": l.order, "height": l.height} for lid, l in SESSION.graph.lanes.items()},
        "blocks": {bid: _block_out(bid) for bid in SESSION.graph.blocks},
        "run_error": run_error,
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


def _packet_preview(
    packet: DataFramePacket, rows: int, with_summary: bool, tags: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    if with_summary:
        packet = packet.compute_summary()
    columns = [
        {
            "name": name,
            "dtype": meta.dtype,
            "role": meta.role.value if hasattr(meta.role, "value") else meta.role,
            "description": meta.description,
            "tags": (tags or {}).get(name, []),
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
            entry: dict[str, Any] = {"name": e.name, "path": str(e), "is_dir": e.is_dir()}
            # Flagged so a project-folder picker (Save/Load) can tell a
            # folder that already holds a project from one that's just a
            # plain directory to browse into -- harmless, and ignored, for
            # every other use of this endpoint (e.g. read_csv's path picker).
            if e.is_dir():
                entry["is_project"] = project.is_project_dir(e)
            entries.append(entry)
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


@app.post("/api/project/new")
def new_project_ep() -> dict[str, Any]:
    """The 'New' toolbar button: discards the in-memory graph in favor of a
    blank one. Nothing is written to disk -- an existing project file on
    disk is untouched, and the crash-recovery snapshot for the discarded
    graph is cleared so it isn't offered back on the next reload."""
    SESSION.new()
    return _graph_out()


@app.post("/api/project/load")
def load_project_ep(req: LoadRequest) -> dict[str, Any]:
    folder = Path(req.path)
    if not project.is_project_dir(folder):
        raise HTTPException(404, f"not a project folder (no {project.PROJECT_FILENAME} in it): {req.path}")
    try:
        SESSION.load(folder / project.PROJECT_FILENAME)
    except FileNotFoundError:
        raise HTTPException(404, f"project file not found: {req.path}")
    return _graph_out()


@app.post("/api/project/save")
def save_project_ep(req: SaveRequest) -> dict[str, Any]:
    target = Path(req.path) / project.PROJECT_FILENAME if req.path else None
    try:
        path = SESSION.save(target)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"path": str(path.parent)}


def _project_dir() -> Path:
    if SESSION.project_path is None:
        raise HTTPException(400, "save the project first -- git needs a project folder to work in")
    return SESSION.project_path.parent


@app.get("/api/project/git/status")
def git_status_ep() -> dict[str, Any]:
    """Whether the currently-open project's folder is a git repo, its
    branch/remote, uncommitted changes, and how far ahead/behind its
    upstream it is -- backs the Toolbar's Git panel. Never raises for "no
    project open yet" (unlike the write endpoints below): the panel wants to
    show "not a repo" rather than an error in that case."""
    if SESSION.project_path is None:
        return {"is_repo": False, "branch": None, "remote": None, "changes": [], "ahead": 0, "behind": 0}
    return gitops.status(_project_dir())


@app.post("/api/project/git/init")
def git_init_ep() -> dict[str, Any]:
    try:
        gitops.init(_project_dir())
    except gitops.GitError as e:
        raise HTTPException(400, str(e))
    return gitops.status(_project_dir())


@app.post("/api/project/git/remote")
def git_remote_ep(req: GitRemoteRequest) -> dict[str, Any]:
    if not req.url.strip():
        raise HTTPException(400, "a remote URL is required")
    try:
        gitops.set_remote(_project_dir(), req.url.strip())
    except gitops.GitError as e:
        raise HTTPException(400, str(e))
    return gitops.status(_project_dir())


@app.post("/api/project/git/commit")
def git_commit_ep(req: GitCommitRequest) -> dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(400, "a commit message is required")
    try:
        gitops.commit(_project_dir(), req.message.strip())
    except gitops.GitError as e:
        raise HTTPException(400, str(e))
    return gitops.status(_project_dir())


@app.post("/api/project/git/push")
def git_push_ep() -> dict[str, Any]:
    try:
        gitops.push(_project_dir())
    except gitops.GitError as e:
        raise HTTPException(502, str(e))
    return gitops.status(_project_dir())


@app.post("/api/project/git/pull")
def git_pull_ep() -> dict[str, Any]:
    try:
        gitops.pull(_project_dir())
    except gitops.GitError as e:
        raise HTTPException(502, str(e))
    return gitops.status(_project_dir())


@app.get("/api/project/recovery")
def recovery_info_ep() -> dict[str, Any]:
    """Metadata about the crash-recovery snapshot, if there is one. Offered
    to the user on a fresh start; never applied on its own."""
    return {"recovery": SESSION.recovery_info()}


@app.post("/api/project/recover")
def recover_ep() -> dict[str, Any]:
    try:
        SESSION.recover()
    except ValueError as e:
        raise HTTPException(404, str(e))
    return _graph_out()


@app.post("/api/project/recovery/dismiss")
def dismiss_recovery_ep() -> dict[str, str]:
    """Discard the crash-recovery snapshot without applying it -- called
    when the user declines the restore prompt, so the same stale snapshot
    doesn't keep coming back on every future startup."""
    SESSION.clear_recovery()
    return {"status": "ok"}


@app.post("/api/undo")
def undo_ep() -> dict[str, Any]:
    if not SESSION.undo():
        raise HTTPException(409, "nothing to undo")
    return _graph_out()


@app.post("/api/redo")
def redo_ep() -> dict[str, Any]:
    if not SESSION.redo():
        raise HTTPException(409, "nothing to redo")
    return _graph_out()


@app.put("/api/sample_mode")
def set_sample_mode(req: SampleModeUpdate) -> dict[str, Any]:
    """Turn sample mode on (rows=N) or off (rows=null). Changes every
    block's cache key, so statuses shift immediately -- nothing re-runs
    until asked. Safe mid-run: a run in flight is pinned to the sample
    setting it started with (see Runner.pin)."""
    try:
        SESSION.set_sample_rows(req.rows)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _graph_out()


@app.post("/api/blocks")
def create_block(req: BlockCreate) -> dict[str, Any]:
    try:
        with SESSION.edit():
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
    with SESSION.edit():
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
        with SESSION.edit():
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
    with SESSION.edit():
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
    return _packet_preview(value, rows=rows, with_summary=summary, tags=SESSION.graph.blocks[block_id].column_tags)


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
        with SESSION.edit():
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
    with SESSION.edit():
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
        with SESSION.edit():
            SESSION.rename_port(block_id, req.port, req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _block_out(block_id)


@app.put("/api/lanes")
def upsert_lane(req: LaneUpsert) -> dict[str, Any]:
    with SESSION.edit():
        SESSION.set_lane(req.id, req.name, req.order, height=req.height)
    return _graph_out()["lanes"]


@app.delete("/api/lanes/{lane_id}")
def delete_lane(lane_id: str) -> dict[str, Any]:
    with SESSION.edit():
        SESSION.delete_lane(lane_id)
    return _graph_out()["lanes"]


def _require_block(block_id: str) -> None:
    if block_id not in SESSION.graph.blocks:
        raise HTTPException(404, f"no such block: {block_id}")


@app.post("/api/blocks/{block_id}/run")
def run_block_ep(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    _start_background_run(lambda: SESSION.runner.run_block(block_id))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/run_to_here")
def run_to_here_ep(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    _start_background_run(lambda: SESSION.runner.run_to_here(block_id))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/refresh")
def refresh_ep(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    if SESSION.graph.blocks[block_id].block_type != "input":
        raise HTTPException(400, "refresh is only valid for input blocks")
    _start_background_run(lambda: SESSION.runner.refresh(block_id))
    return _block_out(block_id)


@app.post("/api/blocks/{block_id}/check_changes")
def check_changes(block_id: str) -> dict[str, Any]:
    _require_block(block_id)
    return {"block_id": block_id, "changed": SESSION.runner.check_for_changes(block_id)}


@app.post("/api/run_all")
def run_all() -> dict[str, Any]:
    report = _start_background_run(lambda: SESSION.runner.run_all())
    return report if report is not None else {"started": True}


@app.post("/api/force_run_all")
def force_run_all() -> dict[str, Any]:
    report = _start_background_run(lambda: SESSION.runner.force_run_all())
    return report if report is not None else {"started": True}


@app.post("/api/run_all_streaming")
def run_all_streaming() -> dict[str, Any]:
    """Opt-in streaming run (see Runner.run_all_streaming): fuses whatever
    contiguous stretch of compatible blocks it safely can into a single
    polars query per group, so a source larger than memory doesn't fully
    materialize at every block boundary. A precondition failure (sample
    mode is on) surfaces the same way run_to_here's does -- see
    _start_background_run."""
    report = _start_background_run(lambda: SESSION.runner.run_all_streaming())
    return report if report is not None else {"started": True}


@app.post("/api/refresh_all")
def refresh_all() -> dict[str, Any]:
    report = _start_background_run(lambda: SESSION.runner.refresh_all())
    return report if report is not None else {"started": True}


@app.post("/api/run/cancel")
def cancel_run() -> dict[str, bool]:
    """Stop whatever's currently running -- a single block, a run_to_here
    cascade, a run_all/force_run_all/refresh_all/run_all_streaming sweep,
    or one streaming run's fused group. The in-flight block's own
    subprocess(es) are terminated immediately (see Runner._dispatch/
    _dispatch_fused_group); a cascade also stops issuing further blocks rather
    than continuing on to the next one."""
    return {"cancelled": SESSION.runner.cancel()}


@app.post("/api/check_all_sources")
def check_all_sources() -> dict[str, bool]:
    return SESSION.runner.check_all_sources()


@app.post("/api/compile")
def compile_ep(req: CompileRequest) -> dict[str, str]:
    try:
        source = compile_graph(
            SESSION.graph,
            runner=SESSION.runner,
            output_blocks=req.output_blocks,
            strict=req.strict,
            stream=req.stream,
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


class EnvVarUpdate(BaseModel):
    value: str


def _env_var_status(name: str) -> dict[str, Any]:
    """Whether-and-where a given env var is set -- never the value itself.
    Same write-only shape as an LLM provider's api_key_set/api_key_source
    (see _redacted_provider_settings): 'override' means this endpoint set
    it for the running server process; 'env' means it was already present
    (shell env or .env) before that."""
    if name not in os.environ:
        return {"name": name, "is_set": False, "source": None}
    return {"name": name, "is_set": True, "source": "override" if name in _ENV_VAR_OVERRIDES else "env"}


@app.get("/api/env_vars/{name}")
def get_env_var(name: str) -> dict[str, Any]:
    return _env_var_status(name)


@app.put("/api/env_vars/{name}")
def set_env_var(name: str, req: EnvVarUpdate) -> dict[str, Any]:
    """Sets an env var for the rest of this server process's lifetime --
    e.g. a Read SQL block's connection string. Deliberately writes straight
    into os.environ (unlike LLM_SETTINGS' own separate override dict):
    block functions run in a spawned subprocess (see runner.MP_CONTEXT),
    which inherits the parent's *current* os.environ at spawn time, so this
    is what actually makes the value visible there. Never written to disk,
    never persisted in a project file, and never echoed back by any
    endpoint -- gone on restart, exactly like an LLM API key override."""
    if not _ENV_VAR_NAME_RE.match(name):
        raise HTTPException(400, f"not a valid environment variable name: {name!r}")
    if name not in _ENV_VAR_OVERRIDES:
        _ENV_VAR_OVERRIDES[name] = os.environ.get(name)
    os.environ[name] = req.value
    return _env_var_status(name)


@app.delete("/api/env_vars/{name}")
def clear_env_var(name: str) -> dict[str, Any]:
    # Restores whatever this name held before it was first overridden
    # (possibly nothing) -- a name that was already present in the
    # shell/.env before any PUT here is left alone entirely.
    if name in _ENV_VAR_OVERRIDES:
        prior = _ENV_VAR_OVERRIDES.pop(name)
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior
    return _env_var_status(name)


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


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "block"


@app.post("/api/blocks/{block_id}/analyze_data")
def analyze_data(block_id: str, req: AnalyzeDataRequest = AnalyzeDataRequest()) -> dict[str, Any]:
    """AI-assisted data profiling: looks at this block's own output columns
    (names, dtypes, current roles, summary stats -- never row data, see
    DraftContext.mode="analyze_data") and proposes per-column tags plus a
    written description of the dataset.

    Tags are applied immediately rather than returned for review-then-save
    like a code draft: they're purely descriptive (see
    BlockInstance.column_tags), so there's nothing to validate or risk
    breaking a run over. The document itself is always returned, and also
    written into the project's files/ folder when one has been saved, so
    it's there to reopen later rather than only living in this one response."""
    _require_block(block_id)
    block = SESSION.graph.blocks[block_id]
    packet = _get_cached_output(block_id, req.port)
    if not isinstance(packet, DataFramePacket):
        raise HTTPException(400, "output port is not a dataframe")
    packet = packet.compute_summary()
    summary = packet.summary or {}

    columns = [
        ColumnInfo(
            name=name,
            dtype=meta.dtype,
            role=meta.role.value if hasattr(meta.role, "value") else meta.role,
            count=summary[name].count if name in summary else None,
            null_count=summary[name].null_count if name in summary else None,
            n_unique=summary[name].n_unique if name in summary else None,
            mean=summary[name].mean if name in summary else None,
            std=summary[name].std if name in summary else None,
            min=summary[name].min if name in summary else None,
            max=summary[name].max if name in summary else None,
        )
        for name, meta in packet.schema_meta.items()
    ]
    ctx = DraftContext(
        instruction="Analyze these columns as described in the system prompt: write the document and propose tags.",
        function_name="(not applicable to this action -- see the system prompt)",
        input_ports={"columns": columns},
        mode="analyze_data",
    )
    try:
        provider = _provider_for(req.provider)
        result = provider.draft(ctx)
    except Exception as e:
        raise HTTPException(502, f"LLM analyze_data failed: {e}")

    tags = {
        column: [t.strip() for t in value.split(",") if t.strip()]
        for column, value in (result.params or {}).items()
        if isinstance(value, str) and column in packet.schema_meta
    }
    port = req.port or block.outputs[0].name
    artifact_title = f"{block.name} :: {port} -- data analysis"
    with SESSION.edit():
        SESSION.set_column_tags(block_id, tags)
        artifact = SESSION.upsert_data_analysis_artifact(block_id, port, artifact_title, result.explanation)

    document_path = None
    if SESSION.project_path is not None and result.explanation.strip():
        files_dir = SESSION.project_path.parent / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        doc_file = files_dir / f"{_slug(block.name)}_data_analysis.md"
        doc_file.write_text(result.explanation, encoding="utf-8")
        document_path = str(doc_file)

    return {
        "document": result.explanation,
        "tags": tags,
        "document_path": document_path,
        "artifact_id": artifact.id,
    }


@app.get("/api/artifacts")
def list_artifacts() -> list[dict[str, Any]]:
    """Every generated document attached to a block's output (today: AI
    data-analysis write-ups -- see analyze_data above), newest-updated
    first, each flagged `stale` when its source block has since changed."""
    return SESSION.list_artifacts()


def _require_artifact(artifact_id: str) -> None:
    if artifact_id not in SESSION.graph.artifacts:
        raise HTTPException(404, f"no such artifact: {artifact_id}")


@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str) -> dict[str, Any]:
    _require_artifact(artifact_id)
    artifact = SESSION.graph.artifacts[artifact_id]
    block = SESSION.graph.blocks.get(artifact.block_id)
    return {
        "id": artifact.id,
        "kind": artifact.kind,
        "title": artifact.title,
        "block_id": artifact.block_id,
        "block_name": block.name if block else None,
        "port": artifact.port,
        "document": artifact.document,
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
        "stale": SESSION.artifact_is_stale(artifact),
    }


@app.patch("/api/artifacts/{artifact_id}")
def rename_artifact(artifact_id: str, req: ArtifactRename) -> dict[str, Any]:
    _require_artifact(artifact_id)
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "title must not be empty")
    with SESSION.edit():
        SESSION.rename_artifact(artifact_id, title)
    return get_artifact(artifact_id)


@app.delete("/api/artifacts/{artifact_id}")
def delete_artifact(artifact_id: str) -> None:
    _require_artifact(artifact_id)
    with SESSION.edit():
        SESSION.delete_artifact(artifact_id)


@app.post("/api/blocks/{block_id}/suggest_names")
def suggest_names(block_id: str, req: SuggestNamesRequest = SuggestNamesRequest()) -> dict[str, Any]:
    """AI-assisted naming: proposes a better display name for this block
    and a name for each of its output ports (see DraftContext.mode="rename"),
    from its category and -- once it's been run -- its actual output
    schema(s). Applied immediately, same as analyze_data: names are purely
    descriptive, so there's nothing to review before accepting the way a
    code change needs. Every proposed port name still goes through
    ProjectSession.rename_port's collision check; a colliding proposal gets
    a numeric suffix appended until it's unique rather than being dropped,
    so this can never silently fail to apply or clash with a name already
    used elsewhere in the graph."""
    _require_block(block_id)
    block = SESSION.graph.blocks[block_id]

    entry = None
    st = SESSION.runner.state.get(block_id)
    if st and st.last_successful_key:
        entry = SESSION.runner.cache.get(st.last_successful_key)

    input_ports: dict[str, list[ColumnInfo]] = {}
    for port_spec in block.outputs:
        packet = entry.outputs.get(port_spec.name) if entry else None
        input_ports[port_spec.name] = (
            [
                ColumnInfo(name=name, dtype=meta.dtype, role=meta.role.value if hasattr(meta.role, "value") else meta.role)
                for name, meta in packet.schema_meta.items()
            ]
            if isinstance(packet, DataFramePacket)
            else []
        )

    used_elsewhere = sorted(
        {name for bid, other in SESSION.graph.blocks.items() if bid != block_id for name in other.port_names.values()}
    )
    ctx = DraftContext(
        instruction=(
            f"Current block name: {block.name!r} (category: {block.category!r}). "
            f"Current output port names: {block.port_names or '(none set)'}. "
            f"Names already in use elsewhere in this graph -- never propose any of these: "
            f"{used_elsewhere or '(none)'}."
        ),
        function_name=block.category,
        input_ports=input_ports,
        mode="rename",
    )
    try:
        provider = _provider_for(req.provider)
        result = provider.draft(ctx)
    except Exception as e:
        raise HTTPException(502, f"LLM suggest_names failed: {e}")

    proposed_name = result.params.get("name")
    with SESSION.edit():
        if isinstance(proposed_name, str) and proposed_name.strip():
            SESSION.update_block(block_id, name=proposed_name.strip())
        for port_spec in block.outputs:
            proposal = result.params.get(port_spec.name)
            if not isinstance(proposal, str) or not proposal.strip():
                continue
            base = proposal.strip()
            candidate = base
            suffix = 2
            while suffix <= 50:
                try:
                    SESSION.rename_port(block_id, port_spec.name, candidate)
                    break
                except ValueError:
                    candidate = f"{base}_{suffix}"
                    suffix += 1

    return {**_block_out(block_id), "explanation": result.explanation}


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
