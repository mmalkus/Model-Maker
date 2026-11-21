"""`modelmaker-mcp`: Model-Maker's chat tools for Claude Code (or any MCP
client), over stdio -- the same tools the VS Code extension gives Copilot
Chat (vscode/src/tools.ts), so an assistant can build a model on the canvas
you're looking at: open or create a project, find the data, add lanes and
blocks, wire, run, check for leakage, compile.

It holds no project state and runs no block code itself. Every tool is a
call to a running Model-Maker backend's HTTP API, the same endpoints the
canvas uses, so edits land in its undo history and the open canvas shows
them within a couple of seconds. Which backend:

- **VS Code** (the default when one is running): the extension writes a
  discovery file per open canvas (~/.modelmaker/vscode-backends/, see
  vscode/src/discovery.ts) with its URL and token. The tools act on the
  canvas you used last, like Copilot's.
- **The browser app**: `--url http://127.0.0.1:8001` (and `--token`, if
  the backend was started with MODELMAKER_AUTH_TOKEN), or with no VS Code
  canvas open, a `modelmaker-api` on the default port.

What an assistant sees follows the AI builder's rule: the graph, block
documentation, output schemas and summary statistics -- never a
dataframe's rows.

    claude mcp add modelmaker -- /path/to/python -m modelmaker.assistant_mcp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

DEFAULT_URL = "http://127.0.0.1:8001"
DISCOVERY_DIR = Path.home() / ".modelmaker" / "vscode-backends"
EXTENSION_ID = "quantology.modelmaker"
RUN_TIMEOUT_SECONDS = 10 * 60
OPEN_TIMEOUT_SECONDS = 120
SCRIPT_CHARS = 30_000
PROJECT_FILE = "model.json"

# Sent to the client at initialization (MCP server instructions). The
# backend's HTTP API is these tools' private transport: an assistant that
# calls it itself, with the token from a discovery file, skips the hints and
# the check step and breaks whenever the API changes (issue #92).
INSTRUCTIONS = (
    "Model-Maker's canvas is driven only through these tools. The canvas backend's HTTP API -- and the URL and "
    "token in ~/.modelmaker/vscode-backends/ -- is private to them: never read that token, call the backend's "
    "/api endpoints, or script against it (curl, Python, fetch). It skips the hints and the check step these tools "
    "add, and it changes without notice. If something you need has no tool, or a tool keeps failing, stop and tell "
    "the user what you needed instead of working around it."
)


class ToolInputError(Exception):
    """A mistake the model can fix: goes back to it as the tool's error."""


class BackendError(Exception):
    """The backend refused a request (its `detail`), or can't be reached."""


# ---- reaching a backend -------------------------------------------------------


@dataclass
class Target:
    url: str
    token: str | None
    folder: str | None = None  # the project folder, for a VS Code canvas
    vscode: bool = False

    def request(self, method: str, path: str, body: Any = None, timeout: float = 60) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.url}/api{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(text).get("detail", text)
            except (json.JSONDecodeError, AttributeError):
                detail = text
            raise BackendError(detail if isinstance(detail, str) else json.dumps(detail)) from None
        except (urllib.error.URLError, OSError) as e:
            raise BackendError(f"can't reach Model-Maker at {self.url}: {getattr(e, 'reason', e)}") from None
        return json.loads(raw) if raw else {}

    def get(self, endpoint: str, /, **params: Any) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        return self.request("GET", f"{endpoint}?{query}" if query else endpoint)

    def post(self, path: str, body: Any = None, timeout: float = 60) -> Any:
        return self.request("POST", path, body if body is not None else {}, timeout)


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def vscode_canvases(directory: Path = DISCOVERY_DIR) -> list[dict[str, Any]]:
    """The VS Code canvases whose backend answers, most recently used first."""
    found = []
    for f in directory.glob("*.json") if directory.is_dir() else []:
        try:
            entry = json.loads(f.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(entry, dict) and entry.get("url") and entry.get("folder") and _healthy(entry["url"]):
            found.append(entry)
    return sorted(found, key=lambda e: e.get("active_at", 0), reverse=True)


def _same_folder(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


class Resolver:
    """Picks the backend each tool call goes to (see the module doc)."""

    def __init__(self, url: str | None, token: str | None, discovery: Path = DISCOVERY_DIR) -> None:
        self.url = url.rstrip("/") if url else None
        self.token = token
        self.discovery = discovery
        # The project open_project chose: stays the target while its canvas is open.
        self.pinned: str | None = None

    def target(self) -> Target:
        if self.url:
            return Target(self.url, self.token)
        canvases = vscode_canvases(self.discovery)
        if canvases:
            chosen = canvases[0]
            if self.pinned:
                chosen = next((c for c in canvases if _same_folder(c["folder"], self.pinned)), chosen)
            return Target(chosen["url"], chosen.get("token"), chosen["folder"], vscode=True)
        if _healthy(DEFAULT_URL):
            return Target(DEFAULT_URL, self.token or os.environ.get("MODELMAKER_AUTH_TOKEN"))
        raise BackendError(
            "no Model-Maker canvas is open. In VS Code, open a project's model.json with the Model-Maker canvas "
            "(or call open_project); for the browser app, start modelmaker-api."
        )


# ---- helpers shared by the tools (mirroring vscode/src/assistant.ts) -----------


def find_block(graph: dict[str, Any], block: str) -> dict[str, Any]:
    blocks = graph["blocks"]
    match = next((b for b in blocks if b["id"] == block), None) or next(
        (b for b in blocks if b["name"].lower() == block.strip().lower()), None
    )
    if match is None:
        raise ToolInputError(f'no block "{block}" -- get the graph for block ids')
    return match


def resolve_lane(graph: dict[str, Any], lane: str | None) -> str | None:
    if not lane:
        return None
    for ln in graph["lanes"]:
        if ln["id"] == lane:
            return ln["id"]
    for ln in graph["lanes"]:
        if ln["name"].lower() == lane.strip().lower():
            return ln["id"]
    known = ", ".join(f"{ln['name']} ({ln['id']})" for ln in graph["lanes"]) or "none"
    raise ToolInputError(f'no lane "{lane}"; the lanes are: {known}')


def find_wire(graph: dict[str, Any], wire: str | None, to_block: str | None, to_port: str | None) -> dict[str, Any]:
    if wire:
        match = next((w for w in graph["wires"] if w["id"] == wire), None)
        if match is None:
            raise ToolInputError(f'no wire "{wire}" -- get the graph for wire ids')
        return match
    if not to_block or not to_port:
        raise ToolInputError("give the wire id, or to_block and to_port")
    to = f"{find_block(graph, to_block)['id']}.{to_port}"
    match = next((w for w in graph["wires"] if w["to"] == to), None)
    if match is None:
        raise ToolInputError(f"nothing is wired into {to}")
    return match


def run_outcome(graph: dict[str, Any], target: str | None) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for b in graph["blocks"]:
        counts[b["status"]] = counts.get(b["status"], 0) + 1
    blocks = [b for b in graph["blocks"] if b["id"] == target] if target else [b for b in graph["blocks"] if b["status"] != "green"]
    out: dict[str, Any] = {
        "statuses": counts,
        "blocks": [{"id": b["id"], "name": b["name"], "status": b["status"], **({"error": b["error"]} if b.get("error") else {})} for b in blocks],
    }
    # A cascade doesn't (re)read data sources -- like the AI builder, that's
    # left to a full run.
    if any(_out_of_memory(b.get("error") or "") for b in blocks):
        out["next"] = OUT_OF_MEMORY_NEXT
    elif any("is not green" in (b.get("error") or "") and "input block" in (b.get("error") or "") for b in blocks):
        out["next"] = "an upstream data source has not been read yet: run the whole pipeline (run without a block), which reads it"
    return out


# A block whose output didn't fit in memory -- stopped by the collect guard
# (run_worker._collect_lazy) or a worker that died. Running it again fails
# the same way; the source has to hand on less.
OUT_OF_MEMORY_NEXT = (
    "a block's output is too big for this machine's memory -- running it again the same way fails the same way. "
    "First, mark the dataframe input of the block it feeds \"lazy\": true (for a library block, in its file, then "
    "check_block_file) and keep that block polars expressions without a collect, then run the whole pipeline with "
    "streaming: true, so the two run as one streamed query that never holds the full table. Or reduce it inside the "
    "block that produces it: select only the needed columns, filter to the population, sample whole ids or "
    "aggregate to the model's grain before returning (for big zipped or plain CSVs, read them through Stage CSV as "
    "Parquet rather than a staging script of your own). Turn on sample mode while building (a streaming run needs "
    "it off)."
)


def _out_of_memory(error: str) -> bool:
    return "too big to hold in memory" in error or "likely out of memory" in error


# ---- the tools -------------------------------------------------------------------

BLOCK = {"type": "string", "description": "A block id (or exact name) from get_graph."}


@dataclass
class Tool:
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    run: Callable[[Resolver, dict[str, Any]], Any]

    def definition(self) -> types.Tool:
        schema: dict[str, Any] = {"type": "object", "properties": self.properties}
        if self.required:
            schema["required"] = self.required
        return types.Tool(name=self.name, description=self.description, inputSchema=schema)


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None):
    def reg(fn):
        TOOLS[name] = Tool(name, description, properties or {}, required or [], fn)
        return fn

    return reg


def _graph(t: Target) -> dict[str, Any]:
    g = t.get("/assistant/graph")
    if t.folder:
        g["folder"] = t.folder
    return g


@tool(
    "open_project",
    "Opens a Model-Maker project's canvas, so the other tools work on it; with create: true, first makes a new, "
    "empty project in that folder (model.json, a files/ folder and a git repository). Returns the project's graph. "
    "Use this when no canvas is open, to switch projects, or to start a new model from scratch.",
    {
        "folder": {"type": "string", "description": "The project folder: relative to the current directory, or absolute."},
        "create": {"type": "boolean", "description": "Create the project if the folder has none."},
    },
    ["folder"],
)
def open_project(r: Resolver, a: dict[str, Any]) -> Any:
    folder = os.path.abspath(a["folder"])
    create = bool(a.get("create"))
    exists = (Path(folder) / PROJECT_FILE).is_file()
    if not exists and not create:
        raise ToolInputError(f"{a['folder']} isn't a Model-Maker project (no {PROJECT_FILE}); pass create: true to make one there")
    if r.url is None and (vscode_canvases(r.discovery) or not _healthy(DEFAULT_URL)):
        return _open_in_vscode(r, folder, create)
    t = r.target()
    current = t.get("/graph")
    if current.get("dirty"):
        if current.get("project_path"):
            t.post("/project/save")
        elif current.get("blocks"):
            raise ToolInputError("the open canvas has unsaved work and no project folder; ask the user to save it first")
    if exists:
        t.post("/project/load", {"path": folder})
    else:
        Path(folder).mkdir(parents=True, exist_ok=True)
        t.post("/project/new")
        t.post("/project/save", {"path": folder})
    return {"folder": folder, **_graph(t)}


def _open_in_vscode(r: Resolver, folder: str, create: bool) -> Any:
    """Asks VS Code (through the extension's URI handler) to open, or create
    and open, the project's canvas, then waits for its backend."""
    for c in vscode_canvases(r.discovery):
        if _same_folder(c["folder"], folder):
            r.pinned = c["folder"]
            return {"folder": c["folder"], **_graph(r.target())}
    scheme = editor_uri_scheme(r.discovery)
    editor = EDITOR_NAMES.get(scheme, "VS Code")
    query = urllib.parse.urlencode({"folder": folder, **({"create": "1"} if create else {})})
    uri = f"{scheme}://{EXTENSION_ID}/open?{query}"
    if not _launch_uri(uri, scheme):
        raise ToolInputError(
            f"couldn't ask {editor} to open {folder}. Ask the user to run 'Model-Maker: "
            f"{'New Project' if create else 'Open Project'}' for it, then call open_project again."
        )
    deadline = time.monotonic() + OPEN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        for c in vscode_canvases(r.discovery):
            if _same_folder(c["folder"], folder):
                r.pinned = c["folder"]
                return {"folder": c["folder"], **_graph(r.target())}
        time.sleep(1)
    raise ToolInputError(
        f"{editor} didn't open the canvas for {folder} within {OPEN_TIMEOUT_SECONDS}s -- the user may need to confirm "
        f"it in {editor}, or open it with 'Model-Maker: Open Project'; then call open_project again."
    )


# URI scheme -> the editor's name and its command-line launcher. Any other
# VS Code-compatible editor's scheme is also its CLI's name (cursor, windsurf).
EDITOR_NAMES = {"vscode": "VS Code", "vscode-insiders": "VS Code Insiders", "cursor": "Cursor", "windsurf": "Windsurf"}
EDITOR_CLIS = {"vscode": ("code",), "vscode-insiders": ("code-insiders",)}
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]{0,31}$")


def editor_uri_scheme(directory: Path = DISCOVERY_DIR) -> str:
    """The URI scheme of the editor to open canvases in: MODELMAKER_EDITOR_URI_SCHEME
    (set by the extension's MCP setup in Cursor or Windsurf), else the editor of
    the canvas used last, else VS Code's."""
    candidates = [os.environ.get("MODELMAKER_EDITOR_URI_SCHEME", "")]
    candidates += [str(c.get("uri_scheme", "")) for c in vscode_canvases(directory)[:1]]
    return next((c for c in (c.strip().lower() for c in candidates) if _SCHEME.match(c)), "vscode")


def _launch_uri(uri: str, scheme: str = "vscode") -> bool:
    """Hands the URI to its editor: the editor's CLI when on PATH, else the OS."""
    cli = next(filter(None, map(shutil.which, EDITOR_CLIS.get(scheme, (scheme,)))), None)
    try:
        if cli and not sys.platform.startswith("win"):
            return subprocess.run([cli, "--open-url", uri], timeout=30, capture_output=True).returncode == 0
        if sys.platform.startswith("win"):
            os.startfile(uri)  # type: ignore[attr-defined]  # noqa: S606 -- the OS routes the scheme to its editor
            return True
        opener = "open" if sys.platform == "darwin" else shutil.which("xdg-open")
        return bool(opener) and subprocess.run([opener, uri], timeout=30, capture_output=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@tool(
    "get_graph",
    "The open Model-Maker project's pipeline: lanes, every block in run order (id, name, category, lane, status "
    "grey/green/orange/red/running, params, input and output ports as name:type, the error of a red block, why an "
    "orange block is stale) and every wire. Call this first, and again after edits. Building a model from scratch: "
    "add lanes for its phases, a data source block (tag data_sources), tag the target column, then add, wire and run "
    "blocks one phase at a time, checking with check as you go. Once the model is built, offer to write a model "
    "document (Markdown in the project folder): purpose and data, then one section per lane in build order "
    "explaining each step and the choices made, with the numbers that drove them, then results, limitations and "
    "next checks.",
)
def get_graph(r: Resolver, a: dict[str, Any]) -> Any:
    return _graph(r.target())


@tool(
    "list_files",
    "Lists a folder on the machine the backend runs on, to find data files for a read_csv/stage_csv_parquet/read_parquet/read_excel/"
    "read_json block's path param (use the path exactly as returned). Without a path: the project's folder. Also "
    "lists the workspace's datasets and sources when it offers them (for read_dataset/read_source blocks).",
    {
        "path": {"type": "string", "description": "A folder path from an earlier listing."},
        "ext": {"type": "string", "description": "Only files with this extension, e.g. .csv."},
    },
)
def list_files(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    path = a.get("path")
    if path is None and t.folder:
        path = t.folder
    files = t.get("/browse", path=path, ext=a.get("ext"))
    out = {**files, "entries": [{"name": e["name"], "path": e["path"], **({"dir": True} if e["is_dir"] else {})} for e in files["entries"]]}
    for key in ("datasets", "sources"):
        try:
            found = t.get(f"/{key}")
        except BackendError:
            found = []
        if found:
            out[key] = found
    return out


@tool(
    "list_block_types",
    "Without a tag: the block tags (data_sources and exports first, then e.g. data_prep, sampling, binning_woe, "
    "regression, performance, stability) with a description of each. With a tag: that tag's blocks (category id and "
    "one-sentence summary), built-ins and the user's library. Use describe_block_type before adding one. If no "
    "block does what's needed, write one for the library: get_block_guide.",
    {"tag": {"type": "string", "description": "A tag from the tag list."}},
)
def list_block_types(r: Resolver, a: dict[str, Any]) -> Any:
    return r.target().get("/assistant/block_types", tag=a.get("tag") or None)


@tool(
    "describe_block_type",
    "Full documentation for one block category: what it does, its parameters (types, defaults, which ones auto-fill "
    "from a column role) and its input/output ports.",
    {"category": {"type": "string", "description": "A block category, e.g. train_test_split."}},
    ["category"],
)
def describe_block_type(r: Resolver, a: dict[str, Any]) -> Any:
    return r.target().get(f"/assistant/block_types/{urllib.parse.quote(a['category'], safe='')}")


@tool(
    "get_output_summary",
    "A block's last output on one port (default: its first). For a dataframe: row count and each column's dtype, "
    "role, nulls, distinct count and mean/std/min/max -- never the rows. For a statistics table (binning summary, "
    "metric tables): its rows. For a metric or model: the value. The block must have run (green or orange).",
    {"block": BLOCK, "port": {"type": "string", "description": "An output port name; default the first."}},
    ["block"],
)
def get_output_summary(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    block = find_block(_graph(t), a["block"])
    return t.get("/assistant/output", block=block["id"], port=a.get("port"))


@tool(
    "get_input_columns",
    "The columns (name, dtype, role) a block receives on each input port, from what its upstream blocks last "
    "produced -- use it to pick column-valued params such as target, features or by. Empty until the upstream "
    "blocks have run.",
    {"block": BLOCK},
    ["block"],
)
def get_input_columns(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    block = find_block(_graph(t), a["block"])
    ports = t.get(f"/blocks/{block['id']}/input_schema")
    out: dict[str, Any] = {"block": block["id"], "ports": ports}
    if all(not cols for cols in ports.values()):
        out["note"] = "no columns yet: the blocks upstream have to run first (run)"
    return out


@tool(
    "check",
    "Reviews the pipeline as the AI builder would: holdout leakage in the wiring (a binning, WoE or model fitted on "
    "rows that include the test/out-of-time sample it is applied to), features fit_binning banded 'suspicious' "
    "(IV >= 0.5: very strong -- leakage if measured after the snapshot date or derived from the outcome), and "
    "decision hints -- what each green block's result means and what to do next (event counts, IV screens, Gini, PSI, calibration...). With a block: just that block's hints. Report problems "
    "to the user rather than silently working around them.",
    {"block": BLOCK},
)
def check(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    target = find_block(_graph(t), a["block"])["id"] if a.get("block") else None
    return t.get("/assistant/check", block=target)


@tool(
    "get_block_guide",
    "How to write a new block (or change one) for the user's block library, when no built-in block does what's "
    "needed: the library folder to write the .py file into, the files already there, the rules (including how "
    "to handle data too big for memory: lazy scans, reduce early, streaming collect) and the commented "
    "template. Call this before writing or editing a block file, then check_block_file.",
)
def get_block_guide(r: Resolver, a: dict[str, Any]) -> Any:
    return r.target().get("/assistant/block_guide")


@tool(
    "check_block_file",
    "Checks a block file you wrote in the library folder: re-reads the folder, validates the file (the BLOCKS "
    "entry, the functions, the params) and runs its _example_ self-checks. Fix what it reports until ok, then "
    "place the block with add_block (its id as the category).",
    {"name": {"type": "string", "description": "The file's name in the library folder, e.g. fannie_mae.py."}},
    ["name"],
)
def check_block_file(r: Resolver, a: dict[str, Any]) -> Any:
    return r.target().post("/assistant/check_block_file", {"name": a["name"]}, timeout=300)


@tool(
    "add_lane",
    "Adds a lane: a labelled horizontal band that groups one phase of the model (e.g. Data, Sampling, Binning, "
    "Estimation, Validation), below the existing ones. Returns its id. Adding a lane whose name exists returns that lane.",
    {"name": {"type": "string", "description": "The lane's label."}},
    ["name"],
)
def add_lane(r: Resolver, a: dict[str, Any]) -> Any:
    return r.target().post("/assistant/lanes", {"name": a["name"]})


@tool(
    "add_block",
    "Adds a block to the canvas, placed in its lane after what's there (it starts grey; wire its inputs with "
    "connect, then run it). Params are checked against the block's own. Returns the new block with its id and ports.",
    {
        "category": {"type": "string", "description": "The block category (from list/describe block types)."},
        "name": {"type": "string", "description": "A short display name."},
        "lane": {"type": "string", "description": "Lane id or name to put it in."},
        "params": {"type": "object", "description": "Parameter values; omitted ones take their defaults."},
    },
    ["category"],
)
def add_block(r: Resolver, a: dict[str, Any]) -> Any:
    body = {"category": a["category"], "name": a.get("name"), "lane": a.get("lane"), "params": a.get("params") or {}}
    return r.target().post("/assistant/blocks", body)


@tool(
    "set_params",
    "Changes some of a block's parameters (merged into its current ones), and/or renames it or moves it to another "
    "lane. Downstream blocks go stale until re-run.",
    {
        "block": BLOCK,
        "params": {"type": "object", "description": "The parameters to change, with their new values."},
        "name": {"type": "string", "description": "A new display name."},
        "lane": {"type": "string", "description": "Lane id or name to move it to."},
    },
    ["block"],
)
def set_params(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    g = _graph(t)
    block = find_block(g, a["block"])
    body: dict[str, Any] = {}
    if a.get("params"):
        body["params"] = {**block["params"], **a["params"]}
    if a.get("name"):
        body["name"] = a["name"]
    if a.get("lane"):
        body["lane"] = resolve_lane(g, a["lane"])
    if not body:
        raise ToolInputError("nothing to change: give params, name or lane")
    t.request("PATCH", f"/blocks/{block['id']}", body)
    return t.get(f"/assistant/blocks/{block['id']}")


@tool(
    "set_column_role",
    "Tags a column's role on a block's output, from there downstream: target, id, weight, feature, date, segment, "
    "excluded (never to be used as a feature), or unassigned to clear it. Params named target/target_col etc. "
    "auto-fill from the roles, so tag the target on the data source block early.",
    {
        "block": BLOCK,
        "column": {"type": "string"},
        "role": {"type": "string", "enum": ["target", "id", "weight", "feature", "date", "segment", "excluded", "unassigned"]},
    },
    ["block", "column", "role"],
)
def set_column_role(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    block = find_block(_graph(t), a["block"])
    t.post(f"/blocks/{block['id']}/column_role", {"column": a["column"], "role": a["role"]})
    return t.get(f"/assistant/blocks/{block['id']}")


@tool(
    "connect",
    "Wires one block's output port to another block's input port (ports as listed by get_graph, without the :type suffix).",
    {"from_block": BLOCK, "from_port": {"type": "string"}, "to_block": BLOCK, "to_port": {"type": "string"}},
    ["from_block", "from_port", "to_block", "to_port"],
)
def connect(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    g = _graph(t)
    body = {
        "from_block": find_block(g, a["from_block"])["id"],
        "from_port": a["from_port"],
        "to_block": find_block(g, a["to_block"])["id"],
        "to_port": a["to_port"],
    }
    return t.post("/wires", body)


@tool(
    "disconnect",
    "Removes a wire: give its id (from get_graph), or the input it feeds (to_block and to_port).",
    {"wire": {"type": "string", "description": "A wire id from get_graph."}, "to_block": BLOCK, "to_port": {"type": "string"}},
)
def disconnect(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    wire = find_wire(_graph(t), a.get("wire"), a.get("to_block"), a.get("to_port"))
    t.request("DELETE", f"/wires/{wire['id']}")
    return {"removed": wire}


@tool("delete_block", "Deletes a block and the wires to and from it.", {"block": BLOCK}, ["block"])
def delete_block(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    block = find_block(_graph(t), a["block"])
    t.request("DELETE", f"/blocks/{block['id']}")
    return {"deleted": block["id"], "name": block["name"]}


@tool(
    "run",
    "Runs a block and the blocks upstream of it, or with no block the whole pipeline (which also reads the data "
    "sources -- a single block's run doesn't), and waits for it. Returns status counts and each affected block's "
    "status, with the error of any red block. Also reports any holdout leakage or suspicious features in the "
    "pipeline, and the target block's decision hints. streaming: true runs the whole pipeline as a streaming run "
    "(no block; sample mode off): stretches of lazy blocks fuse into one streamed query, so a source larger than "
    "memory never fully materializes between them -- the way forward when a block's output is too big.",
    {
        "block": BLOCK,
        "streaming": {
            "type": "boolean",
            "description": "Run the whole pipeline streaming, fusing lazy blocks into streamed queries. Leave out block.",
        },
    },
)
def run(r: Resolver, a: dict[str, Any]) -> Any:
    t = r.target()
    streaming = bool(a.get("streaming"))
    if streaming and a.get("block"):
        raise ToolInputError("a streaming run runs the whole pipeline: leave out block")
    target = find_block(_graph(t), a["block"])["id"] if a.get("block") else None
    deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
    t.post("/run_all_streaming" if streaming else f"/blocks/{target}/run_to_here" if target else "/run_all", timeout=120)
    g = _wait_idle(t, deadline)
    if g["running"]:
        return {"still_running": True, "note": "the run is still going; check the graph later", **run_outcome(g, target)}
    out = run_outcome(g, target)
    try:
        found = t.get("/assistant/check", block=target)
    except BackendError:
        return out
    if found["leaks"]:
        out["leaks"] = found["leaks"]
    if found["suspicious_features"]:
        out["suspicious_features"] = found["suspicious_features"]
    if target and found["hints"].get(target):
        out["hints"] = found["hints"][target]
    elif not target and found["hints"]:
        out["next_check"] = "check gives each block's decision hints"
    return out


def _wait_idle(t: Target, deadline: float) -> dict[str, Any]:
    """The graph once nothing runs, or as it stands at the deadline."""
    while True:
        g = _graph(t)
        if not g["running"] or time.monotonic() > deadline:
            return g
        time.sleep(0.5)


@tool(
    "compile",
    "Compiles the pipeline into one self-contained, readable Python script and returns its source. Every block must be green.",
    {"streaming": {"type": "boolean", "description": "Fuse compatible stretches into streaming polars queries."}},
)
def compile_(r: Resolver, a: dict[str, Any]) -> Any:
    source = r.target().post("/compile", {"stream": bool(a.get("streaming"))}, timeout=120)["source"]
    return {"source": source[:SCRIPT_CHARS], "truncated": True} if len(source) > SCRIPT_CHARS else {"source": source}


# ---- the server -------------------------------------------------------------------


def call(resolver: Resolver, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    """(result, is_error) of one tool call."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"no tool {name!r}"}, True
    try:
        return spec.run(resolver, args or {}), False
    except (ToolInputError, BackendError) as e:
        return {"error": str(e)}, True


def build_server(resolver: Resolver) -> Server:
    server = Server("modelmaker", instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [t.definition() for t in TOOLS.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        result, is_error = await asyncio.to_thread(call, resolver, name, arguments)
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result, default=str))], isError=is_error)

    return server


async def _main(resolver: Resolver) -> None:
    server = build_server(resolver)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get("MODELMAKER_URL"), help="a modelmaker-api to use (default: the VS Code canvas you used last)")
    parser.add_argument("--token", default=os.environ.get("MODELMAKER_AUTH_TOKEN"), help="its MODELMAKER_AUTH_TOKEN, if it has one")
    args = parser.parse_args()
    asyncio.run(_main(Resolver(args.url, args.token)))


if __name__ == "__main__":
    main()
