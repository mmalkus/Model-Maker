"""The tools an AI build calls, as plain functions of (build, **args) with a
hand-written JSON schema each. The same table backs every loop -- the
in-process API loops and the MCP server the claude CLI talks to -- so the
guards here (ownership, excluded columns, no rows) hold whichever model
or provider is driving. See /agent-builder-proposal.md §5-§6."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..blocks.base import BLOCK_REGISTRY
from ..graph import _compile_code_to_fn
from ..llm.redact import column_info_for_llm
from ..metadata_transforms import resolve_metadata_transform
from ..packet import ColumnRole, DataFramePacket
from ..runslot import RunBusy, RunFailed
from ..session import new_id, wire_is_valid
from . import catalogue
from .build import AWAITING_APPROVAL, AWAITING_INPUT, BUILDING, PLANNING, AgentBuild, ToolError
from .layout import Placer, next_lane_order

READ = frozenset({PLANNING, BUILDING})
WRITE = frozenset({BUILDING})


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., dict[str, Any]]
    phases: frozenset[str]

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict[str, Any], required: list[str], phases: frozenset[str]):
    def reg(fn):
        TOOLS[name] = Tool(
            name=name,
            description=description,
            schema={"type": "object", "properties": properties, "required": required, "additionalProperties": False},
            fn=fn,
            phases=phases,
        )
        return fn

    return reg


def tools_for_phase(phase: str) -> list[Tool]:
    return [t for t in TOOLS.values() if phase in t.phases]


# ---- shared helpers ----------------------------------------------------------

_STR = {"type": "string"}
_BLOCK = {"type": "string", "description": "A block id, as returned by get_graph/add_block."}


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _json_preview(value: Any, limit: int = 6000) -> Any:
    """A JSON-safe copy of `value`, or a truncated JSON string when big."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return repr(value)[:limit]
    if len(text) > limit:
        return {"truncated_json": text[:limit] + "..."}
    return json.loads(text)


def summarize_value(value: Any, with_stats: bool = True) -> dict[str, Any]:
    """What the model sees of one output value. Dataframes: schema, roles,
    row count and redacted summary statistics -- never rows. Everything
    else (metrics, model summaries, simulation results) is already an
    aggregate and goes through as truncated JSON; images only by size."""
    if isinstance(value, DataFramePacket):
        packet = value.compute_summary() if with_stats else value
        summary = packet.summary or {}
        columns = []
        for name, meta in packet.schema_meta.items():
            info = column_info_for_llm(name, meta, summary.get(name) if with_stats else None)
            entry = _strip_none(
                {
                    "name": info.name,
                    "dtype": info.dtype,
                    "role": info.role,
                    "nulls": info.null_count or None,
                    "distinct": info.n_unique,
                    "mean": info.mean,
                    "std": info.std,
                    "min": info.min,
                    "max": info.max,
                }
            )
            if meta.tags:
                entry["tags"] = meta.tags
            columns.append(entry)
        return {"type": "dataframe", "row_count": packet.data.height, "columns": columns}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "image", "bytes": len(value), "note": "images aren't shown to the build"}
    return {"type": type(value).__name__, "value": _json_preview(value)}


_FEATURE_PARAM = re.compile(r"^(features?|cols?|columns?|risk_factors|by|on|x|y)$|_cols?$|_columns?$")


def check_excluded(b: AgentBuild, params: dict[str, Any], where: str) -> None:
    excluded = b.excluded_columns()
    if not excluded:
        return
    for name, value in (params or {}).items():
        if not _FEATURE_PARAM.search(name):
            continue
        values = value if isinstance(value, list) else [value]
        hits = sorted({v for v in values if isinstance(v, str) and v in excluded})
        if hits:
            raise ToolError(
                f"{where}: param {name!r} names {hits}, which the user tagged 'excluded' -- excluded columns "
                "must not be used. Leave them out."
            )


def check_code_excluded(b: AgentBuild, code: str) -> None:
    hits = sorted(c for c in b.excluded_columns() if f'"{c}"' in code or f"'{c}'" in code)
    if hits:
        raise ToolError(f"custom code references {hits}, which the user tagged 'excluded' -- leave them out")


def resolve_lane(b: AgentBuild, lane: str) -> str:
    """A lane argument may be an existing lane id, an existing lane's name
    (case-insensitive), a plan lane key, or a new name (creates a lane)."""
    graph = b.session.graph
    lane_map = b.lane_map
    if lane in lane_map and lane_map[lane] in graph.lanes:
        return lane_map[lane]
    if lane in graph.lanes:
        return lane
    for lane_id, l in graph.lanes.items():
        if l.name.strip().lower() == lane.strip().lower():
            return lane_id
    plan_lane = next((l for l in (b.plan or {}).get("lanes", []) if l.get("key") == lane), None)
    name = plan_lane["name"] if plan_lane else lane
    lane_id = new_id("lane")
    with b.session.edit():
        b.session.set_lane(lane_id, name, next_lane_order(graph))
    lane_map[lane] = lane_id
    b.owned_lanes.add(lane_id)
    b.log("lane", lane=lane_id, name=name)
    return lane_id


def _block_brief(b: AgentBuild, bid: str) -> dict[str, Any]:
    block = b.session.graph.blocks[bid]
    runner = b.session.runner
    status = runner.status(bid)
    st = runner.state.get(bid)
    out = {
        "id": bid,
        "name": block.name,
        "category": "custom" if block.is_custom else block.category,
        "lane": block.lane,
        "status": status,
        "params": block.params,
        "inputs": [f"{p.name}:{p.type}" for p in block.inputs],
        "outputs": [f"{p.name}:{p.type}" for p in block.outputs],
        "owned_by_this_build": b.is_owned(bid),
    }
    if bid in b.anchors:
        out["anchor"] = True
    if bid in b.approved_changes:
        out["approved_change"] = [c.get("change") for c in b.approved_changes[bid]]
    if block.column_role_overrides:
        out["column_roles"] = block.column_role_overrides
    if status == "red" and st and st.last_error:
        out["error"] = st.last_error[-600:]
    if status == "orange":
        out["stale_reason"] = runner.stale_reason(bid)
    return out


# ---- read-only tools -----------------------------------------------------


@tool(
    "list_block_types",
    "List the registry blocks you can add: category, what it does, and its ports. Call describe_block_type "
    "for a block's parameters before adding it. Optionally filter by group (e.g. modelling, tests, quality).",
    {"group": _STR},
    [],
    READ,
)
def list_block_types(b: AgentBuild, group: str | None = None) -> dict[str, Any]:
    blocks = catalogue.list_block_types(group)
    return {"groups": sorted({e["group"] for e in blocks}), "blocks": blocks}


@tool(
    "describe_block_type",
    "Full documentation for one registry block: docstring, parameters (types, defaults, which auto-fill "
    "from a column role), and ports.",
    {"category": _STR},
    ["category"],
    READ,
)
def describe_block_type(b: AgentBuild, category: str) -> dict[str, Any]:
    try:
        return catalogue.describe_block_type(category)
    except KeyError as e:
        raise ToolError(str(e)) from None


@tool(
    "get_graph",
    "The current graph: lanes, every block (id, name, category, lane, status, params, ports, whether this "
    "build created it, errors) and every wire. Anchors are the blocks the user asked you to build from.",
    {},
    [],
    READ,
)
def get_graph(b: AgentBuild) -> dict[str, Any]:
    graph = b.session.graph
    return {
        "lanes": [
            {"id": lid, "name": l.name, "order": l.order}
            for lid, l in sorted(graph.lanes.items(), key=lambda kv: kv[1].order)
        ],
        "blocks": [_block_brief(b, bid) for bid in graph.topo_order()],
        "wires": [
            {
                "id": wid,
                "from": f"{w.from_block}.{w.from_port}",
                "to": f"{w.to_block}.{w.to_port}",
                "valid": wire_is_valid(graph, w),
                "owned_by_this_build": wid in b.owned_wires,
            }
            for wid, w in graph.wires.items()
        ],
    }


@tool(
    "get_output_summary",
    "Summary of a block's last successful output on one port: for a dataframe the columns (dtype, role, "
    "null count, distinct count, mean/std/min/max) and row count -- never the rows themselves; for a "
    "metric or model the value. Defaults to the block's first output port.",
    {"block": _BLOCK, "port": _STR},
    ["block"],
    READ,
)
def get_output_summary(b: AgentBuild, block: str, port: str | None = None) -> dict[str, Any]:
    port, value = b.current_output(block, port)
    status = b.session.runner.status(block)
    out = {"block": block, "port": port, "status": status, **summarize_value(value)}
    if status == "orange":
        out["note"] = "this output is stale (the block or its upstream changed since it ran) -- run_to it for a current one"
    return out


@tool(
    "get_block_error",
    "A block's last error, with its params -- and its code, for a custom block.",
    {"block": _BLOCK},
    ["block"],
    READ,
)
def get_block_error(b: AgentBuild, block: str) -> dict[str, Any]:
    blk = b.require_block(block)
    st = b.session.runner.state.get(block)
    out = {"status": b.session.runner.status(block), "error": st.last_error if st else None, "params": blk.params}
    if blk.is_custom:
        out["code"] = blk.code
        out["metadata_transform"] = blk.metadata_transform
    return out


# ---- planning ------------------------------------------------------------------

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One or two sentences: what you'll build."},
        "assumptions": {"type": "array", "items": _STR},
        "questions": {
            "type": "array",
            "items": _STR,
            "description": "Anything you need the user to decide before building. Non-empty blocks approval until answered.",
        },
        "lanes": {
            "type": "array",
            "description": "New lanes to create (existing lanes can be referenced by id without listing them).",
            "items": {
                "type": "object",
                "properties": {"key": _STR, "name": _STR, "purpose": _STR},
                "required": ["key", "name"],
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Plan-local id, e.g. s1."},
                    "category": {"type": "string", "description": "A registry category, or 'custom'."},
                    "instruction": {"type": "string", "description": "For custom steps: what the code will do."},
                    "lane": {"type": "string", "description": "A plan lane key or an existing lane id."},
                    "name": _STR,
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "port": _STR,
                                "from": {"type": "string", "description": "A step ref or an existing block id."},
                                "from_port": _STR,
                            },
                            "required": ["port", "from", "from_port"],
                        },
                    },
                    "params": {"type": "object"},
                    "why": _STR,
                },
                "required": ["ref", "category", "lane", "name", "inputs", "why"],
            },
        },
        "changes_to_existing": {
            "type": "array",
            "description": "Changes to blocks that already exist (params, or wiring into their inputs). Each needs user approval.",
            "items": {
                "type": "object",
                "properties": {"block": _STR, "change": _STR, "why": _STR},
                "required": ["block", "change", "why"],
            },
        },
    },
    "required": ["summary", "steps"],
}


def _step_ports(step: dict[str, Any]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """(inputs, outputs) as {port: type} for a plan step, or None when not
    knowable (custom steps' inputs are whatever they declare)."""
    if step.get("category") == "custom":
        return None, {"out": "dataframe"}
    spec = BLOCK_REGISTRY[step["category"]]
    return {p.name: p.type for p in spec.inputs}, {p.name: p.type for p in spec.outputs}


def validate_plan(b: AgentBuild, plan: dict[str, Any]) -> list[str]:
    graph = b.session.graph
    errors: list[str] = []
    lane_keys = {l.get("key") for l in plan.get("lanes", [])}
    steps = plan.get("steps") or []
    if not steps:
        errors.append("plan has no steps")
    refs: dict[str, dict[str, Any]] = {}
    for i, step in enumerate(steps):
        where = f"step {step.get('ref') or i}"
        ref = step.get("ref")
        if not ref or ref in refs or ref in graph.blocks:
            errors.append(f"{where}: ref must be unique and not an existing block id")
        cat = step.get("category")
        if cat != "custom" and cat not in BLOCK_REGISTRY:
            errors.append(f"{where}: unknown category {cat!r}")
            refs[ref] = step
            continue
        if cat in catalogue.AGENT_DISALLOWED:
            errors.append(f"{where}: {cat} can't be added by an AI build ({catalogue.AGENT_DISALLOWED[cat]})")
        if cat == "custom" and not step.get("instruction"):
            errors.append(f"{where}: custom steps need an instruction")
        lane = step.get("lane")
        if lane not in lane_keys and lane not in graph.lanes:
            errors.append(f"{where}: lane {lane!r} is neither a plan lane key nor an existing lane id")
        in_ports, _ = _step_ports(step)
        for inp in step.get("inputs") or []:
            src = inp.get("from")
            if src in refs:
                src_step = refs[src]
                valid_src = src_step.get("category") == "custom" or src_step.get("category") in BLOCK_REGISTRY
                src_out = _step_ports(src_step)[1] if valid_src else None
            elif src in graph.blocks:
                src_out = {p.name: p.type for p in graph.blocks[src].outputs}
            else:
                errors.append(f"{where}: input from {src!r} is neither an earlier step ref nor an existing block id")
                continue
            if src_out is not None and inp.get("from_port") not in src_out:
                errors.append(f"{where}: {src!r} has no output port {inp.get('from_port')!r} (has {sorted(src_out)})")
                continue
            if in_ports is not None and inp.get("port") not in in_ports:
                errors.append(f"{where}: {cat} has no input port {inp.get('port')!r} (has {sorted(in_ports)})")
                continue
            if src_out is not None and in_ports is not None:
                a, bt = src_out[inp["from_port"]], in_ports[inp["port"]]
                if a != bt and "any" not in (a, bt):
                    errors.append(f"{where}: can't wire {a} output {src}.{inp['from_port']} into {bt} input {inp['port']}")
        refs[ref] = step
    for change in plan.get("changes_to_existing") or []:
        if change.get("block") not in graph.blocks:
            errors.append(f"changes_to_existing: no such block {change.get('block')!r}")
    return errors


def layout_plan(b: AgentBuild, plan: dict[str, Any]) -> None:
    """Attach ghost positions (see §4.2): each step's x/y, and the bands of
    lanes the plan would create -- computed by the same Placer the build
    phase uses."""
    graph = b.session.graph
    new_lanes = [(l["key"], l["name"]) for l in plan.get("lanes", []) if l.get("key") not in graph.lanes]
    placer = Placer(graph, new_lanes)
    plan["layout"] = {}
    for step in plan["steps"]:
        x, y = placer.place(step["lane"])
        plan["layout"][step["ref"]] = {"lane": step["lane"], "x": x, "y": y}
    plan["lane_layout"] = {
        key: {"name": g.name, "top": g.top, "height": g.height}
        for key, g in placer.lanes.items()
        if key not in graph.lanes
    }


@tool(
    "submit_plan",
    "Submit your build plan for the user's review. This ends the planning phase: after it succeeds, end "
    "your turn. If it returns errors, fix them and submit again.",
    {"plan": _PLAN_SCHEMA},
    ["plan"],
    frozenset({PLANNING}),
)
def submit_plan(b: AgentBuild, plan: dict[str, Any]) -> dict[str, Any]:
    if b.plan is not None and b.phase != PLANNING:
        raise ToolError("a plan was already submitted -- end your turn")
    errors = validate_plan(b, plan)
    if errors:
        raise ToolError("plan has problems, fix and resubmit:\n- " + "\n- ".join(errors))
    layout_plan(b, plan)
    b.plan = plan
    b.set_phase(AWAITING_APPROVAL)
    b.turn_over = True
    return {"ok": True, "message": "Plan submitted for the user's review. End your turn now."}


# ---- building ------------------------------------------------------------------


def _finish_add(b: AgentBuild, block, plan_step: str | None) -> dict[str, Any]:
    block.provenance = b.provenance_for(plan_step)
    b.owned_blocks.add(block.id)
    return {
        "block": block.id,
        "name": block.name,
        "inputs": [f"{p.name}:{p.type}" for p in block.inputs],
        "outputs": [f"{p.name}:{p.type}" for p in block.outputs],
        "next": "connect its inputs, then run_to it",
    }


@tool(
    "add_block",
    "Add a registry block. It's placed automatically in the given lane. Params you leave out keep their "
    "defaults; params that auto-fill from a column role (e.g. target_col) can usually be left out.",
    {
        "category": _STR,
        "lane": {"type": "string", "description": "An existing lane id or name, or a plan lane key."},
        "name": _STR,
        "params": {"type": "object"},
        "plan_step": {"type": "string", "description": "The plan step ref this implements, if any."},
    },
    ["category", "lane"],
    WRITE,
)
def add_block(
    b: AgentBuild, category: str, lane: str, name: str | None = None, params: dict | None = None, plan_step: str | None = None
) -> dict[str, Any]:
    if category not in BLOCK_REGISTRY:
        raise ToolError(f"unknown category {category!r} -- use add_custom_block for custom code")
    if category in catalogue.AGENT_DISALLOWED:
        raise ToolError(f"{category} can't be added by an AI build: {catalogue.AGENT_DISALLOWED[category]}")
    params = params or {}
    known = {p["name"] for p in catalogue.block_params(BLOCK_REGISTRY[category])}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ToolError(f"{category} has no params {unknown}; its params are {sorted(known)}")
    check_excluded(b, params, category)
    lane_id = resolve_lane(b, lane)
    x, y = Placer(b.session.graph).place(lane_id)
    with b.session.edit():
        block = b.session.add_block(category, name=name, lane=lane_id, x=x, y=y, params=params)
    return _finish_add(b, block, plan_step)


def _validate_custom(b: AgentBuild, code: str, inputs: list[str], metadata_transform: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compile the code and check it against the custom-block contract;
    returns (function name, its extra params' defaults)."""
    try:
        fn = _compile_code_to_fn(code)
    except SyntaxError as e:
        raise ToolError(f"code doesn't parse: {e}") from None
    except ValueError as e:
        raise ToolError(str(e)) from None
    sig = inspect.signature(fn)
    names = list(sig.parameters)
    if names[: len(inputs)] != inputs:
        raise ToolError(f"the function's first parameters must be the input ports {inputs} in order; got {names}")
    if fn.__name__ in BLOCK_REGISTRY:
        raise ToolError(f"function name {fn.__name__!r} clashes with a registry block -- pick another")
    if metadata_transform.get("kind") not in ("passthrough", "narrow", "declared"):
        raise ToolError("metadata_transform.kind must be passthrough, narrow or declared")
    try:
        resolve_metadata_transform(metadata_transform)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"bad metadata_transform: {e}") from None
    defaults = {
        n: p.default
        for n, p in sig.parameters.items()
        if n not in inputs and p.default is not inspect.Parameter.empty
    }
    check_code_excluded(b, code)
    return fn.__name__, defaults


_CUSTOM_PROPS = {
    "code": {
        "type": "string",
        "description": "One top-level polars function (see the custom block contract in your instructions).",
    },
    "metadata_transform": {
        "type": "object",
        "description": "{kind: passthrough|narrow|declared, base?, drops?, adds?: [{name, dtype, role}]}",
    },
    "params": {"type": "object", "description": "Values for the function's extra parameters."},
}


@tool(
    "add_custom_block",
    "Add a block with your own polars code, for steps no registry block covers. The function's first "
    "parameters are its dataframe inputs (named as in `inputs`, default ['df']); it returns one dataframe "
    "on output port 'out'.",
    {
        "lane": _STR,
        "name": _STR,
        "inputs": {"type": "array", "items": _STR},
        **_CUSTOM_PROPS,
        "plan_step": _STR,
    },
    ["lane", "name", "code", "metadata_transform"],
    WRITE,
)
def add_custom_block(
    b: AgentBuild,
    lane: str,
    name: str,
    code: str,
    metadata_transform: dict[str, Any],
    inputs: list[str] | None = None,
    params: dict | None = None,
    plan_step: str | None = None,
) -> dict[str, Any]:
    limit = b.options.limits.custom_blocks
    if b.counters["custom_blocks"] >= limit:
        raise ToolError(f"this build already has {limit} custom blocks (the limit) -- use a registry block or ask_user")
    inputs = inputs or ["df"]
    fn_name, defaults = _validate_custom(b, code, inputs, metadata_transform)
    if any(blk.category == fn_name for blk in b.session.graph.blocks.values()):
        fn_name_hint = f"{fn_name}_2"
        raise ToolError(f"a block with function name {fn_name!r} already exists -- rename the function, e.g. {fn_name_hint}")
    params = {**defaults, **(params or {})}
    check_excluded(b, params, name)
    lane_id = resolve_lane(b, lane)
    x, y = Placer(b.session.graph).place(lane_id)
    with b.session.edit():
        block = b.session.add_block(
            fn_name,
            block_type="standard",
            name=name,
            lane=lane_id,
            x=x,
            y=y,
            params=params,
            code=code,
            inputs=[{"name": n, "type": "dataframe"} for n in inputs],
            outputs=[{"name": "out", "type": "dataframe"}],
            metadata_transform=metadata_transform,
        )
    b.counters["custom_blocks"] += 1
    return _finish_add(b, block, plan_step)


@tool(
    "update_custom_block",
    "Replace a custom block's code, metadata_transform and/or params -- e.g. to fix an error. The input "
    "ports stay as they are.",
    {"block": _BLOCK, **_CUSTOM_PROPS},
    ["block"],
    WRITE,
)
def update_custom_block(
    b: AgentBuild, block: str, code: str | None = None, metadata_transform: dict | None = None, params: dict | None = None
) -> dict[str, Any]:
    blk = b.require_modifiable(block, "update")
    if not blk.is_custom:
        raise ToolError("not a custom block -- use set_params")
    inputs = [p.name for p in blk.inputs]
    new_code = code or blk.code or ""
    new_transform = metadata_transform or blk.metadata_transform or {"kind": "passthrough"}
    fn_name, defaults = _validate_custom(b, new_code, inputs, new_transform)
    new_params = {**{k: v for k, v in defaults.items() if k not in blk.params}, **blk.params, **(params or {})}
    check_excluded(b, new_params, blk.name)
    with b.session.edit():
        b.session.update_block(block, actor="agent", code=new_code, metadata_transform=new_transform, params=new_params)
        blk.category = fn_name
    b.record_approved_change(block, "updated custom code")
    b.failures.pop(block, None)
    return {"ok": True, "block": block}


@tool(
    "connect",
    "Wire an output port into an input port. An input port takes one wire; connecting replaces any "
    "existing wire into it.",
    {"from_block": _BLOCK, "from_port": _STR, "to_block": _BLOCK, "to_port": _STR},
    ["from_block", "from_port", "to_block", "to_port"],
    WRITE,
)
def connect(b: AgentBuild, from_block: str, from_port: str, to_block: str, to_port: str) -> dict[str, Any]:
    src = b.require_block(from_block)
    dst = b.require_modifiable(to_block, "wire into")
    src_port = next((p for p in src.outputs if p.name == from_port), None)
    dst_port = next((p for p in dst.inputs if p.name == to_port), None)
    if src_port is None:
        raise ToolError(f"{src.name!r} has no output port {from_port!r}; outputs: {[p.name for p in src.outputs]}")
    if dst_port is None:
        raise ToolError(f"{dst.name!r} has no input port {to_port!r}; inputs: {[p.name for p in dst.inputs]}")
    if src_port.type != dst_port.type and "any" not in (src_port.type, dst_port.type):
        raise ToolError(f"type mismatch: {from_port} is {src_port.type}, {to_port} expects {dst_port.type}")
    try:
        with b.session.edit():
            wire = b.session.add_wire(from_block, from_port, to_block, to_port)
    except ValueError as e:
        raise ToolError(str(e)) from None
    b.owned_wires.add(wire.id)
    b.record_approved_change(to_block, f"wired {src.name}.{from_port} into {to_port}")
    return {"wire": wire.id}


@tool("disconnect", "Remove a wire this build created.", {"wire": _STR}, ["wire"], WRITE)
def disconnect(b: AgentBuild, wire: str) -> dict[str, Any]:
    w = b.session.graph.wires.get(wire)
    if w is None:
        raise ToolError(f"no such wire {wire!r}")
    if wire not in b.owned_wires:
        b.require_modifiable(w.to_block, "unwire")
    with b.session.edit():
        b.session.delete_wire(wire)
    b.owned_wires.discard(wire)
    b.record_approved_change(w.to_block, f"removed wire into {w.to_port}")
    return {"ok": True}


@tool("delete_block", "Delete a block this build created (and its wires).", {"block": _BLOCK}, ["block"], WRITE)
def delete_block(b: AgentBuild, block: str) -> dict[str, Any]:
    blk = b.require_block(block)
    if not b.is_owned(block):
        raise ToolError(f"{blk.name!r} belongs to the user -- an AI build can only delete blocks it created")
    with b.session.edit():
        b.session.delete_block(block)
    b.owned_blocks.discard(block)
    b.owned_wires &= set(b.session.graph.wires)
    return {"ok": True}


@tool(
    "set_params",
    "Set some of a block's params (merged into the existing ones; a null value resets that param to its "
    "default).",
    {"block": _BLOCK, "params": {"type": "object"}},
    ["block", "params"],
    WRITE,
)
def set_params(b: AgentBuild, block: str, params: dict[str, Any]) -> dict[str, Any]:
    blk = b.require_modifiable(block, "change params of")
    if not blk.is_custom:
        known = {p["name"] for p in catalogue.block_params(BLOCK_REGISTRY[blk.category])}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ToolError(f"{blk.category} has no params {unknown}; its params are {sorted(known)}")
    merged = {**blk.params, **params}
    merged = {k: v for k, v in merged.items() if v is not None}
    check_excluded(b, merged, blk.name)
    with b.session.edit():
        b.session.update_block(block, actor="agent", params=merged)
    b.record_approved_change(block, f"set params {sorted(params)}")
    b.failures.pop(block, None)
    return {"ok": True, "params": merged}


@tool(
    "set_column_role",
    "Tag a column's role on a block this build created (id, target, weight, feature, date, segment, "
    "excluded, unassigned). The user's own role tags are read-only.",
    {"block": _BLOCK, "column": _STR, "role": _STR},
    ["block", "column", "role"],
    WRITE,
)
def set_column_role(b: AgentBuild, block: str, column: str, role: str) -> dict[str, Any]:
    blk = b.require_block(block)
    if not b.is_owned(block):
        raise ToolError(f"{blk.name!r} belongs to the user, and the user's role tags are read-only")
    try:
        with b.session.edit():
            b.session.set_column_role(block, column, role, actor="agent")
    except ValueError as e:
        raise ToolError(str(e)) from None
    return {"ok": True}


@tool(
    "run_to",
    "Run a block, and whatever upstream of it isn't current. Returns its status, a short summary of its "
    "outputs when it succeeded, and the error when it failed.",
    {"block": _BLOCK},
    ["block"],
    WRITE,
)
def run_to(b: AgentBuild, block: str) -> dict[str, Any]:
    blk = b.require_block(block)
    if blk.block_type == "input":
        raise ToolError("input blocks are the user's to (re)read -- run_to a block downstream of it instead")
    runner = b.session.runner
    try:
        status = b.run_slot.run_sync(lambda: runner.run_to_here(block))
    except RunBusy:
        raise ToolError("another run is in progress -- try again shortly") from None
    except (RunFailed, RuntimeError) as e:
        raise ToolError(f"couldn't run: {e}") from None
    out: dict[str, Any] = {"block": block, "status": status}
    if status == "green":
        b.failures.pop(block, None)
        outputs = {}
        for p in blk.outputs:
            try:
                _, value = b.current_output(block, p.name)
            except ToolError:
                continue
            s = summarize_value(value, with_stats=False)
            if s["type"] == "dataframe":
                s = {"type": "dataframe", "row_count": s["row_count"], "columns": [c["name"] for c in s["columns"]]}
            outputs[p.name] = s
        out["outputs"] = outputs
        return out
    st = runner.state.get(block)
    out["error"] = (st.last_error or "")[-1500:] if st else None
    upstream_red = [
        {"block": a, "name": b.session.graph.blocks[a].name, "error": (runner.state[a].last_error or "")[-400:]}
        for a in b.session.graph.ancestors([block]) - {block}
        if runner.status(a) == "red" and a in runner.state
    ]
    if upstream_red:
        out["upstream_failures"] = upstream_red
    if b.is_owned(block) and status == "red":
        b.failures[block] = b.failures.get(block, 0) + 1
        if b.failures[block] >= b.options.limits.consecutive_failures_per_block:
            out["must_ask_user"] = (
                f"{blk.name!r} has failed {b.failures[block]} times in a row. Stop retrying: call ask_user "
                "with what you tried and what you need."
            )
    return out


@tool(
    "note_deviation",
    "Record a small deviation from the approved plan (e.g. a param changed to make a fit converge, or an "
    "extra cleaning step). Structural changes need ask_user instead.",
    {"plan_step": _STR, "what": _STR, "why": _STR},
    ["what", "why"],
    WRITE,
)
def note_deviation(b: AgentBuild, what: str, why: str, plan_step: str | None = None) -> dict[str, Any]:
    b.deviations.append(_strip_none({"plan_step": plan_step, "what": what, "why": why}))
    return {"ok": True}


@tool(
    "ask_user",
    "Pause the build and ask the user something you can't decide yourself (a structural change to the "
    "plan, a change to one of their blocks that wasn't approved, or repeated failures). After calling "
    "it, end your turn -- you'll be resumed with their answer.",
    {"question": _STR},
    ["question"],
    WRITE,
)
def ask_user(b: AgentBuild, question: str) -> dict[str, Any]:
    b.pending_question = question
    b.set_phase(AWAITING_INPUT)
    b.turn_over = True
    return {"ok": True, "message": "Question sent to the user. End your turn now."}


@tool(
    "finish",
    "Finish the build. `report` is Markdown for the user: what you built, deviations, key results and "
    "concerns. `key_outputs` lists the blocks/ports holding the headline metrics, so they can be refreshed "
    "after the final full-data run. After calling it, end your turn.",
    {
        "report": _STR,
        "key_outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"block": _STR, "port": _STR, "label": _STR},
                "required": ["block"],
            },
        },
    },
    ["report"],
    WRITE,
)
def finish(b: AgentBuild, report: str, key_outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    for k in key_outputs or []:
        b.require_block(k["block"])
    b.report = report
    b.key_outputs = list(key_outputs or [])
    b.finished = True
    b.turn_over = True
    return {"ok": True, "message": "Build finished. End your turn now."}
