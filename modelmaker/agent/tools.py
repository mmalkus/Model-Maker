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

import polars as pl

from ..blocks.base import BLOCK_REGISTRY
from ..graph import _compile_code_to_fn
from ..llm.redact import column_info_for_llm
from ..metadata_transforms import resolve_metadata_transform
from ..packet import ColumnRole, DataFramePacket
from ..runslot import RunBusy, RunFailed
from ..session import new_id, wire_is_valid
from . import catalogue, leakage
from .build import AWAITING_APPROVAL, AWAITING_INPUT, BUILDING, PLANNING, STAGE_ACTIVE, STAGE_DONE, STAGE_SKIPPED, AgentBuild, ToolError
from .hints import decision_hints
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


# The tools that write custom code -- left out when a build has custom
# blocks turned off (BuildOptions.allow_custom_blocks).
CUSTOM_TOOLS = frozenset({"add_custom_block", "update_custom_block"})


def tools_for_phase(phase: str, allow_custom: bool = True) -> list[Tool]:
    return [t for t in TOOLS.values() if phase in t.phases and (allow_custom or t.name not in CUSTOM_TOOLS)]


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


# A statistics table's rows (see BlockSpec.aggregate_outputs): at most this
# many, and at most this much JSON -- a bins table for 30 features is a
# few hundred rows.
TABLE_ROW_LIMIT = 200
TABLE_CHAR_LIMIT = 20_000


# What run_to (and so plan_stage, per step) reports of a block that ran:
# enough to check the step without a get_output_summary round trip, which
# re-sends the whole conversation. A statistics table comes whole when its
# rows fit REPORT_TABLE_CHARS; a dataframe brings its target rate and the
# statistics of the columns the block added (a split's samples, WoE
# columns, a score) -- not all of them, which get_output_summary has.
REPORT_TABLE_CHARS = 6_000
REPORT_NEW_COLUMNS = 8


def _input_columns(b: AgentBuild, block_id: str) -> set[str] | None:
    """The columns of the dataframes wired into a block; None when one
    can't be read (then nothing counts as new)."""
    cols: set[str] = set()
    for wire in b.session.graph.input_wires(block_id).values():
        try:
            _, value = b.current_output(wire.from_block, wire.from_port)
        except ToolError:
            return None
        if isinstance(value, DataFramePacket):
            cols.update(value.data.columns)
    return cols


def _dataframe_report(b: AgentBuild, block_id: str, packet: DataFramePacket) -> dict[str, Any]:
    columns = list(packet.data.columns)
    out: dict[str, Any] = {"type": "dataframe", "row_count": packet.data.height, "columns": columns}
    targets = [n for n, m in packet.schema_meta.items() if m.role == ColumnRole.TARGET and n in columns]
    seen = _input_columns(b, block_id)
    new = [] if seen is None else [c for c in columns if c not in seen]
    if not new and not targets:
        return out
    full = summarize_value(packet)
    by_name = {c["name"]: c for c in full["columns"]}
    for t in targets:
        if "mean" in by_name.get(t, {}):
            out[f"{t}_mean"] = by_name[t]["mean"]
    if new:
        out["new_columns"] = [by_name[c] for c in new[:REPORT_NEW_COLUMNS] if c in by_name]
        if len(new) > REPORT_NEW_COLUMNS:
            out["new_columns_note"] = f"{len(new) - REPORT_NEW_COLUMNS} more new columns -- get_output_summary has them"
    return out


def run_report(b: AgentBuild, block_id: str) -> dict[str, Any]:
    """Each output of a block that just ran green (see REPORT_TABLE_CHARS)."""
    blk = b.session.graph.blocks[block_id]
    outputs: dict[str, Any] = {}
    for p in blk.outputs:
        try:
            _, value = b.current_output(block_id, p.name)
        except ToolError:
            continue
        if isinstance(value, DataFramePacket) and is_statistics_table(b, block_id, p.name):
            table = table_rows(value, char_limit=REPORT_TABLE_CHARS)
            if table.get("rows_shown"):
                table = {
                    "type": "statistics_table",
                    "row_count": table["row_count"],
                    "columns": table["columns"],
                    "note": "too big to show here -- get_output_summary shows its rows",
                }
            outputs[p.name] = table
        elif isinstance(value, DataFramePacket):
            outputs[p.name] = _dataframe_report(b, block_id, value)
        else:
            outputs[p.name] = summarize_value(value, with_stats=False)
    return outputs


def is_statistics_table(b: AgentBuild, block_id: str, port: str) -> bool:
    """Whether a block's output port is one its registry spec declares a
    statistics table. Never true for a custom block: the build writes
    those, so it can't vouch for their rows."""
    block = b.session.graph.blocks.get(block_id)
    if block is None or block.is_custom:
        return False
    spec = BLOCK_REGISTRY.get(block.category)
    return spec is not None and port in spec.aggregate_outputs


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.6g}")
    return value


def table_rows(packet: DataFramePacket, char_limit: int | None = None) -> dict[str, Any]:
    """A statistics table as the model sees it: its rows, rounded, within
    TABLE_ROW_LIMIT and char_limit (default TABLE_CHAR_LIMIT) -- id-role
    columns left out even here."""
    char_limit = TABLE_CHAR_LIMIT if char_limit is None else char_limit
    df = packet.data
    keep = [n for n, m in packet.schema_meta.items() if m.role != ColumnRole.ID and n in df.columns]
    rows = [{k: _round(v) for k, v in r.items()} for r in df.select(keep).head(TABLE_ROW_LIMIT).iter_rows(named=True)]
    while rows and len(json.dumps(rows, default=str)) > char_limit:
        rows = rows[: len(rows) * 3 // 4]
    rows = json.loads(json.dumps(rows, default=str))
    out: dict[str, Any] = {"type": "statistics_table", "row_count": df.height, "columns": keep, "rows": rows}
    if len(rows) < df.height:
        out["rows_shown"] = f"the first {len(rows)} of {df.height} rows -- the rest are left out"
    return out


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


SUSPICIOUS_BAND = "suspicious"  # fit_binning's iv_band for IV >= 0.5


def block_hints(b: AgentBuild, block_id: str) -> list[str]:
    """Decision hints (see hints.py) for a block that just ran green."""
    blk = b.session.graph.blocks[block_id]
    if blk.is_custom:
        return []
    outputs = {}
    for p in blk.outputs:
        try:
            outputs[p.name] = b.current_output(block_id, p.name)[1]
        except ToolError:
            continue
    return decision_hints(blk.category, outputs)


def suspicious_features(b: AgentBuild) -> dict[str, float | None]:
    """{feature: IV} for every feature a fit_binning in the graph banded
    'suspicious' -- an IV that high usually means the column is partly the
    outcome (leakage)."""
    out: dict[str, float | None] = {}
    for bid, blk in b.session.graph.blocks.items():
        if blk.is_custom or blk.category != "fit_binning":
            continue
        try:
            _, value = b.current_output(bid, "summary")
        except ToolError:
            continue
        if not isinstance(value, DataFramePacket) or not {"feature", "iv_band"} <= set(value.data.columns):
            continue
        for row in value.data.filter(pl.col("iv_band") == SUSPICIOUS_BAND).iter_rows(named=True):
            out[row["feature"]] = row.get("iv")
    return out


def flag_suspicious(b: AgentBuild, category: str, params: dict[str, Any], where: str) -> str | None:
    """When a model-fitting block's features include one fit_binning banded
    'suspicious' (directly or as its _woe column), record a concern the
    user will see at the stage review and in the report -- whatever the
    model says about it -- and return a warning for the tool result."""
    spec = BLOCK_REGISTRY.get(category)
    if spec is None or "regression" not in spec.tags:
        return None
    suspicious = suspicious_features(b)
    if not suspicious:
        return None
    used = set()
    for name, value in (params or {}).items():
        if _FEATURE_PARAM.search(name):
            used |= {v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)}
    hits = sorted({base for v in used for base in (v, v.removesuffix("_woe")) if base in suspicious})
    if not hits:
        return None
    for feature in hits:
        iv = suspicious[feature]
        b.add_concern(
            f"{where} uses {feature}{f' (IV {iv:.2f})' if isinstance(iv, float) else ''}, which fit_binning banded "
            "'suspicious' -- an IV that high usually means leakage."
        )
    return (
        f"{', '.join(hits)} {'was' if len(hits) == 1 else 'were'} banded 'suspicious' by fit_binning (IV >= 0.5), which "
        "usually means leakage. This is recorded as a concern for the user. Unless they have already agreed to it, "
        "ask_user before keeping it in the model."
    )


def check_code_excluded(b: AgentBuild, code: str) -> None:
    hits = sorted(c for c in b.excluded_columns() if f'"{c}"' in code or f"'{c}'" in code)
    if hits:
        raise ToolError(f"custom code references {hits}, which the user tagged 'excluded' -- leave them out")


def resolve_lane(b: AgentBuild, lane: str) -> str:
    """A lane argument may be an existing lane id, an existing lane's name
    (case-insensitive), a plan stage key, or a new name (creates a lane)."""
    graph = b.session.graph
    lane_map = b.lane_map
    if lane in lane_map and lane_map[lane] in graph.lanes:
        return lane_map[lane]
    if lane in graph.lanes:
        return lane
    for lane_id, l in graph.lanes.items():
        if l.name.strip().lower() == lane.strip().lower():
            return lane_id
    stage = next((st for st in (b.plan or {}).get("stages", []) if st.get("key") == lane), None)
    name = stage["name"] if stage else lane
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
    "List the registry blocks with a given tag (e.g. pd, regression, calibration): category and a "
    "one-sentence summary. Without a tag, lists the tags. Call describe_block_type for a block's ports "
    "and parameters before adding it.",
    {"tag": _STR},
    [],
    READ,
)
def list_block_types(b: AgentBuild, tag: str | None = None) -> dict[str, Any]:
    if not tag:
        return {"tags": catalogue.list_tags(), "next": "call list_block_types with one of these tags"}
    try:
        return {"tag": tag, "blocks": catalogue.list_blocks_for_tag(tag)}
    except KeyError as e:
        raise ToolError(str(e.args[0])) from None


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
    "statistics table (e.g. fit_binning's summary, compare_samples' table -- one row per feature, bin or "
    "sample) its rows; for a metric or model the value. Defaults to the block's first output port.",
    {"block": _BLOCK, "port": _STR},
    ["block"],
    READ,
)
def get_output_summary(b: AgentBuild, block: str, port: str | None = None) -> dict[str, Any]:
    port, value = b.current_output(block, port)
    status = b.session.runner.status(block)
    if isinstance(value, DataFramePacket) and is_statistics_table(b, block, port):
        summary = table_rows(value)
    else:
        summary = summarize_value(value)
    out = {"block": block, "port": port, "status": status, **summary}
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
#
# Two levels (see /agent-builder-proposal.md §4.2): the plan phase submits
# an outline of stages for the user to approve (submit_plan); the build
# then plans each stage's blocks (plan_stage) just before building it,
# with the earlier stages' real results in hand, and reports back when the
# stage is done (complete_stage).

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
        "stages": {
            "type": "array",
            "description": "The modelling stages, in build order. Each is built into one lane.",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Plan-local id, e.g. prep, est, val."},
                    "name": {"type": "string", "description": "The lane name, e.g. Estimation."},
                    "goal": {
                        "type": "string",
                        "description": "What this stage does and produces, with the decisions that shape it "
                        "(e.g. split design, model family, which metrics). Not individual blocks.",
                    },
                    "lane": {"type": "string", "description": "An existing lane id to build into, instead of a new lane."},
                    "blocks": {
                        "type": "array",
                        "items": _STR,
                        "description": "The registry categories this stage will most likely use -- their docs are "
                        "handed to the build when the stage starts.",
                    },
                },
                "required": ["key", "name", "goal", "blocks"],
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
    "required": ["summary", "stages"],
}

_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "description": "Plan-local id, e.g. s1 -- unique across the whole build."},
        "category": {"type": "string", "description": "A registry category, or 'custom'."},
        "instruction": {"type": "string", "description": "For custom steps: what the code will do."},
        "lane": {"type": "string", "description": "A stage key or an existing lane id. Defaults to this stage's lane."},
        "name": _STR,
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "port": _STR,
                    "from": {"type": "string", "description": "A step ref of this stage or an existing block id."},
                    "from_port": _STR,
                },
                "required": ["port", "from", "from_port"],
            },
        },
        "params": {"type": "object"},
        "why": _STR,
    },
    "required": ["ref", "category", "name", "inputs", "why"],
}


def _step_ports(step: dict[str, Any]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """(inputs, outputs) as {port: type} for a plan step, or None when not
    knowable (custom steps' inputs are whatever they declare)."""
    if step.get("category") == "custom":
        return None, {"out": "dataframe"}
    spec = BLOCK_REGISTRY[step["category"]]
    return {p.name: p.type for p in spec.inputs}, {p.name: p.type for p in spec.outputs}


def validate_plan(b: AgentBuild, plan: dict[str, Any]) -> list[str]:
    """The outline: stages with unique keys, each a new lane or an existing
    one; changes_to_existing naming real blocks."""
    graph = b.session.graph
    errors: list[str] = []
    stages = plan.get("stages") or []
    if not stages:
        errors.append("plan has no stages")
    keys: set[str] = set()
    for i, stage in enumerate(stages):
        key = stage.get("key")
        where = f"stage {key or i}"
        if not key or key in keys or key in graph.lanes:
            errors.append(f"{where}: key must be unique and not an existing lane id")
        keys.add(key)
        if not (stage.get("goal") or "").strip():
            errors.append(f"{where}: needs a goal")
        if not stage.get("blocks"):
            errors.append(f"{where}: list the registry blocks it will most likely use in `blocks`")
        unknown = [c for c in stage.get("blocks") or [] if c not in BLOCK_REGISTRY or c in catalogue.AGENT_DISALLOWED]
        if unknown:
            hints = "; ".join(f"{c!r}{catalogue.unknown_category_hint(c)}" for c in unknown)
            errors.append(f"{where}: blocks {unknown} aren't registry blocks an AI build can add ({hints})")
        lane = stage.get("lane")
        if lane is not None and lane not in graph.lanes:
            errors.append(f"{where}: lane {lane!r} isn't an existing lane id -- leave it out to create a new lane")
    errors += leakage.outline_order_problems(stages, BLOCK_REGISTRY)
    for change in plan.get("changes_to_existing") or []:
        if change.get("block") not in graph.blocks:
            errors.append(f"changes_to_existing: no such block {change.get('block')!r}")
    return errors


def validate_steps(b: AgentBuild, steps: list[dict[str, Any]]) -> list[str]:
    """One stage's steps: real categories and ports, lanes that exist (or a
    stage key), inputs from an earlier step of this stage or an existing
    block, refs unique across the whole build. Fills in a missing lane
    with the current stage's."""
    graph = b.session.graph
    errors: list[str] = []
    stage = b.current_stage
    stage_keys = {st["key"] for st in b.stages}
    taken = {
        s["ref"] for st in b.stages if st is not stage and st.get("plan") for s in st["plan"]["steps"]
    }
    if not steps:
        errors.append("the stage plan has no steps")
    refs: dict[str, dict[str, Any]] = {}
    for i, step in enumerate(steps):
        where = f"step {step.get('ref') or i}"
        ref = step.get("ref")
        if not ref or ref in refs or ref in graph.blocks:
            errors.append(f"{where}: ref must be unique and not an existing block id")
        elif ref in taken:
            errors.append(f"{where}: ref {ref!r} was used by an earlier stage -- pick a new one")
        cat = step.get("category")
        if cat != "custom" and cat not in BLOCK_REGISTRY:
            errors.append(f"{where}: unknown category {cat!r}{catalogue.unknown_category_hint(cat)}")
            refs[ref] = step
            continue
        if cat in catalogue.AGENT_DISALLOWED:
            errors.append(f"{where}: {cat} can't be added by an AI build ({catalogue.AGENT_DISALLOWED[cat]})")
        if cat == "custom" and not b.options.allow_custom_blocks:
            errors.append(f"{where}: custom blocks are turned off for this build -- use registry blocks")
        elif cat == "custom" and not step.get("instruction"):
            errors.append(f"{where}: custom steps need an instruction")
        if not step.get("lane") and stage is not None:
            step["lane"] = stage["key"]
        lane = step.get("lane")
        if lane not in stage_keys and lane not in graph.lanes:
            errors.append(f"{where}: lane {lane!r} is neither a stage key nor an existing lane id")
        in_ports, _ = _step_ports(step)
        for inp in step.get("inputs") or []:
            src = inp.get("from")
            if src in refs:
                src_step = refs[src]
                valid_src = src_step.get("category") == "custom" or src_step.get("category") in BLOCK_REGISTRY
                src_out = _step_ports(src_step)[1] if valid_src else None
            elif src in graph.blocks:
                src_out = {p.name: p.type for p in graph.blocks[src].outputs}
            elif src in taken:
                errors.append(f"{where}: {src!r} is an earlier stage's step -- wire from the block it built (by block id)")
                continue
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
    if not errors:
        errors += _plan_leaks(b, steps)
    return errors


def _leak_graph(
    b: AgentBuild, steps: list[dict[str, Any]] | None = None, wire: tuple[str, str, str, str] | None = None
) -> tuple[dict[str, leakage.Node], list[leakage.Edge], dict[str, str]]:
    """The graph as leakage.find_leaks sees it -- the real one, plus a
    stage's planned steps (each step's planned inputs replacing whatever
    its built block is wired to) or one wire about to be connected.
    Returns (nodes, edges, node id -> step ref)."""
    graph = b.session.graph
    nodes = {
        bid: leakage.Node(bid, blk.name, blk.category, {p.name: p.type for p in blk.inputs}, {p.name: p.type for p in blk.outputs})
        for bid, blk in graph.blocks.items()
    }
    edges = [leakage.Edge(w.from_block, w.from_port, w.to_block, w.to_port) for w in graph.wires.values()]
    planned: dict[str, str] = {}
    if steps:
        built = _step_blocks(b)
        for step in steps:
            cat = step.get("category")
            if cat != "custom" and cat not in BLOCK_REGISTRY:
                continue
            nid = built.get(step["ref"], step["ref"])
            planned[nid] = step["ref"]
            if nid not in nodes:
                ins, outs = _step_ports(step)
                nodes[nid] = leakage.Node(nid, step.get("name") or step["ref"], cat, ins or {}, outs or {})
            edges = [e for e in edges if e.dst != nid]
            for inp in step.get("inputs") or []:
                edges.append(leakage.Edge(built.get(inp["from"], inp["from"]), inp["from_port"], nid, inp["port"]))
    if wire is not None:
        src, src_port, dst, dst_port = wire
        edges = [e for e in edges if not (e.dst == dst and e.dst_port == dst_port)]
        edges.append(leakage.Edge(src, src_port, dst, dst_port))
    return nodes, edges, planned


def _plan_leaks(b: AgentBuild, steps: list[dict[str, Any]]) -> list[str]:
    """Holdout leakage a stage plan would create (see agent/leakage.py)."""
    nodes, edges, planned = _leak_graph(b, steps=steps)
    out = []
    for leak in leakage.find_leaks(nodes, edges):
        step = planned.get(leak.block) or planned.get(leak.fitter)
        if step:
            out.append(f"step {step}: {leak.message} (If the user wants it anyway, they can wire it themselves.)")
    return out


def flag_leaks(b: AgentBuild) -> None:
    """Record any holdout leakage left in the graph -- e.g. among the
    user's own blocks -- as a concern for the stage summary and report."""
    nodes, edges, _ = _leak_graph(b)
    for leak in leakage.find_leaks(nodes, edges):
        b.add_concern(leak.message)


def layout_plan(b: AgentBuild, plan: dict[str, Any]) -> None:
    """Attach the ghost bands (see §4.2) of the lanes the outline would
    create, one per stage that doesn't name an existing lane."""
    graph = b.session.graph
    new_lanes = [(st["key"], st["name"]) for st in plan["stages"] if not st.get("lane")]
    placer = Placer(graph, new_lanes)
    plan["lane_layout"] = {
        key: {"name": g.name, "top": g.top, "height": g.height}
        for key, g in placer.lanes.items()
        if key not in graph.lanes
    }


def layout_steps(b: AgentBuild, steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ghost positions for one stage's steps -- computed by the same Placer
    add_block uses, after whatever the earlier stages built, so a real
    block lands where its ghost was."""
    placer = Placer(b.session.graph)
    layout = {}
    for step in steps:
        lane_id = b.lane_map.get(step["lane"], step["lane"])
        x, y = placer.place(lane_id)
        layout[step["ref"]] = {"lane": lane_id, "x": x, "y": y}
    return layout


@tool(
    "submit_plan",
    "Submit your outline plan -- the stages, in order -- for the user's review. This ends the planning "
    "phase: after it succeeds, end your turn. If it returns errors, fix them and submit again.",
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


def _require_active_stage(b: AgentBuild) -> dict[str, Any]:
    stage = b.current_stage
    if stage is None:
        raise ToolError("this build has no stages to plan")
    if stage["status"] != STAGE_ACTIVE:
        raise ToolError(f"stage {stage['key']!r} is already complete -- end your turn")
    return stage


@tool(
    "plan_stage",
    "Plan the current stage's blocks, now that you can see what the earlier stages produced -- and, by "
    "default, build them: the app adds each registry step in order, wires it as planned, runs it and "
    "reports back, stopping at the first step that fails or needs custom code. Steps wire from earlier "
    "steps of this stage or from existing blocks by id (including blocks built in earlier stages). Call it "
    "again if the stage's steps change: steps already built are kept (with any changed params applied), new "
    "ones are built. build=false only plans.",
    {"steps": {"type": "array", "items": _STEP_SCHEMA}, "build": {"type": "boolean"}},
    ["steps"],
    WRITE,
)
def plan_stage(b: AgentBuild, steps: list[dict[str, Any]], build: bool = True) -> dict[str, Any]:
    stage = _require_active_stage(b)
    errors = validate_steps(b, steps)
    if errors:
        raise ToolError("stage plan has problems, fix and resubmit:\n- " + "\n- ".join(errors))
    stage["plan"] = {"steps": steps, "layout": layout_steps(b, steps)}
    b.log("stage", key=stage["key"], name=stage["name"], status="planned", steps=len(steps))
    if not build:
        return {"ok": True, "next": "build the steps in order (build_stage, or add_block with plan_step, connect, run_to)"}
    return _build_stage_plan(b, stage, apply_params=True)


@tool(
    "build_stage",
    "Build (the rest of) the current stage's plan: adds, wires and runs every step not built yet in plan "
    "order, and re-runs any built step that isn't green -- e.g. after you fixed a failed step's params, or "
    "added a custom step with add_custom_block(plan_step=...). Stops at the first step that fails or needs "
    "custom code.",
    {},
    [],
    WRITE,
)
def build_stage(b: AgentBuild) -> dict[str, Any]:
    stage = _require_active_stage(b)
    if not stage.get("plan"):
        raise ToolError("call plan_stage for this stage first")
    return _build_stage_plan(b, stage)


def _sync_plan_params(b: AgentBuild, block, params: dict[str, Any]) -> None:
    """A fix to a planned step's params (set_params) goes into the stage
    plan too, so the plan stays what was built -- and build_stage never
    puts the old values back."""
    ref = (block.provenance or {}).get("plan_step")
    plan = (b.current_stage or {}).get("plan")
    if not ref or not plan:
        return
    for step in plan["steps"]:
        if step["ref"] == ref:
            merged = {**(step.get("params") or {}), **params}
            step["params"] = {k: v for k, v in merged.items() if v is not None}


def _step_blocks(b: AgentBuild) -> dict[str, str]:
    """Plan step ref -> the block this build created for it."""
    out = {}
    for bid in b.owned_blocks:
        block = b.session.graph.blocks.get(bid)
        ref = (block.provenance or {}).get("plan_step") if block is not None else None
        if ref:
            out[ref] = bid
    return out


def _build_stage_plan(b: AgentBuild, stage: dict[str, Any], apply_params: bool = False) -> dict[str, Any]:
    """Build the stage's plan in order -- the same add_block / connect /
    run_to a model would call, in one tool call instead of three or four
    per step. Each step's outcome is logged as a `step` event and reported
    back briefly; the first failure (or custom step) stops the run and
    hands back to the model. `apply_params` (a re-plan): a built step whose
    planned params changed gets them set."""
    built = _step_blocks(b)
    report: list[dict[str, Any]] = []

    def stop(entry: dict[str, Any], **extra: Any) -> dict[str, Any]:
        report.append(entry)
        b.log("step", ref=entry["ref"], block=entry.get("block"), status=entry["status"], error=entry.get("error"))
        return {"ok": False, "stopped_at": entry["ref"], "steps": report, **extra}

    for step in stage["plan"]["steps"]:
        b.check_stop()
        ref = step["ref"]
        entry: dict[str, Any] = {"ref": ref}
        bid = built.get(ref)
        if bid is None:
            if step["category"] == "custom":
                entry["status"] = "needs_code"
                return stop(
                    entry,
                    next=f"write it with add_custom_block (plan_step {ref!r}, lane {step['lane']!r}) and connect its "
                    "inputs, then call build_stage to build the rest",
                )
            try:
                added = add_block(b, step["category"], step["lane"], step.get("name"), step.get("params"), ref)
                bid = built[ref] = added["block"]
                entry["block"] = bid
                if added.get("warning"):
                    entry["warning"] = added["warning"]
                for inp in step.get("inputs") or []:
                    connect(b, built.get(inp["from"], inp["from"]), inp["from_port"], bid, inp["port"])
            except ToolError as e:
                entry.update(status="error", error=str(e))
                return stop(entry, next="fix it (set_params / connect / delete_block), then call build_stage")
        entry["block"] = bid
        block = b.session.graph.blocks[bid]
        changed = {k: v for k, v in (step.get("params") or {}).items() if block.params.get(k) != v}
        if apply_params and changed and not block.is_custom:
            # A re-plan changed a built step's params: apply them.
            try:
                set_params(b, bid, changed)
            except ToolError as e:
                entry.update(status="error", error=str(e))
                return stop(entry, next="fix the step's params, then call plan_stage again")
        if b.session.runner.status(bid) != "green":
            try:
                ran = run_to(b, bid)
            except ToolError as e:
                entry.update(status="error", error=str(e))
                return stop(entry, next="fix it, then call build_stage")
            entry["status"] = ran["status"]
            if ran["status"] != "green":
                entry.update({k: ran[k] for k in ("error", "upstream_failures", "must_ask_user") if k in ran})
                return stop(entry, next="read the error, fix the step (set_params, or plan_stage with changed steps), then call build_stage")
            entry["outputs"] = ran.get("outputs")
            if ran.get("next"):
                entry["next"] = ran["next"]
        else:
            entry["status"] = "green"
        report.append(entry)
        b.log("step", ref=ref, block=bid, status=entry["status"])
    flag_leaks(b)
    return {
        "ok": True,
        "steps": report,
        "next": "check the results above (get_output_summary only for what they leave out), then complete_stage -- or finish on the last stage",
    }


@tool(
    "complete_stage",
    "Report the current stage as built and checked. `summary` is Markdown for the user: what you built and "
    "what the results show (headline numbers, and decisions you made from them, e.g. features dropped for "
    "low IV). After calling it, end your turn -- the user reviews the stage before the next one starts. On "
    "the last stage, call finish instead.",
    {"summary": _STR},
    ["summary"],
    WRITE,
)
def complete_stage(b: AgentBuild, summary: str) -> dict[str, Any]:
    stage = _require_active_stage(b)
    if b.is_last_stage():
        raise ToolError("this is the last stage -- call finish instead")
    if not stage.get("plan"):
        raise ToolError("call plan_stage for this stage (and build it) first")
    stage["status"] = STAGE_DONE
    stage["summary"] = summary
    b.log("stage", key=stage["key"], name=stage["name"], status="done", summary=summary)
    b.turn_over = True
    return {"ok": True, "message": "Stage complete. End your turn now -- you'll be resumed for the next stage."}


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
        custom = " (or add_custom_block for custom code)" if b.options.allow_custom_blocks else ""
        raise ToolError(f"unknown category {category!r}{catalogue.unknown_category_hint(category)}{custom}")
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
    out = _finish_add(b, block, plan_step)
    warning = flag_suspicious(b, category, params, block.name)
    if warning:
        out["warning"] = warning
    return out


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
    before = {(lk.block, lk.fitter, lk.split) for lk in leakage.find_leaks(*_leak_graph(b)[:2])}
    new = [
        lk
        for lk in leakage.find_leaks(*_leak_graph(b, wire=(from_block, from_port, to_block, to_port))[:2])
        if (lk.block, lk.fitter, lk.split) not in before
    ]
    if new:
        raise ToolError(new[0].message + " (If the user wants it anyway, they can wire it themselves.)")
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
    _sync_plan_params(b, blk, params)
    out = {"ok": True, "params": merged}
    warning = flag_suspicious(b, blk.category, merged, blk.name) if not blk.is_custom else None
    if warning:
        out["warning"] = warning
    return out


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
        out["outputs"] = run_report(b, block)
        if b.options.decision_hints:
            hints = block_hints(b, block)
            if hints:
                out["next"] = hints
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
    "Record a small deviation from the stage plan or the approved outline (e.g. a param changed to make a "
    "fit converge, or an extra cleaning step). Structural changes need ask_user instead.",
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
    "approved outline, a change to one of their blocks that wasn't approved, or repeated failures). After calling "
    "it, end your turn -- you'll be resumed with their answer.",
    {"question": _STR},
    ["question"],
    WRITE,
)
def ask_user(b: AgentBuild, question: str) -> dict[str, Any]:
    b.pending_question = question
    b.last_question = question
    b.set_phase(AWAITING_INPUT)
    b.turn_over = True
    return {"ok": True, "message": "Question sent to the user. End your turn now."}


@tool(
    "finish",
    "Finish the build, once the last stage is built and checked. `report` is Markdown for the user: what "
    "you built, deviations, key results and concerns. `key_outputs` lists the blocks/ports holding the "
    "headline metrics, so they can be refreshed after the final full-data run. If the user agreed (through "
    "ask_user) to drop the remaining stages, say why in `dropped_stages_reason`. After calling it, end your turn.",
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
        "dropped_stages_reason": _STR,
    },
    ["report"],
    WRITE,
)
def finish(
    b: AgentBuild, report: str, key_outputs: list[dict[str, Any]] | None = None, dropped_stages_reason: str | None = None
) -> dict[str, Any]:
    stage = b.current_stage
    remaining = b.stages[b.stage_index + 1 :] if stage is not None else []
    if remaining and not dropped_stages_reason:
        names = ", ".join(st["name"] for st in remaining)
        raise ToolError(
            f"stages still to build after this one: {names}. Finish this stage with complete_stage -- or, if the "
            "user agreed to drop them, pass dropped_stages_reason."
        )
    for k in key_outputs or []:
        b.require_block(k["block"])
    if stage is not None:
        stage["status"] = STAGE_DONE
        b.log("stage", key=stage["key"], name=stage["name"], status="done")
    for st in remaining:
        st["status"] = STAGE_SKIPPED
        b.deviations.append({"what": f"dropped stage {st['name']}", "why": dropped_stages_reason})
        b.log("stage", key=st["key"], name=st["name"], status="skipped")
    b.report = report
    b.key_outputs = list(key_outputs or [])
    b.finished = True
    b.turn_over = True
    return {"ok": True, "message": "Build finished. End your turn now."}
