"""The block catalogue an AI build chooses from -- what each registry block
does, its ports, and its parameters, derived from the registry itself
(BlockSpec + the block function's signature and docstring) so there's no
second list to keep in sync."""

from __future__ import annotations

import inspect
from typing import Any

from ..blocks.base import BLOCK_REGISTRY, BlockSpec
from ..packet import ColumnRole
from ..util import ROLE_PARAM_NAMES, find_role_param

# Params the runner injects itself (see Runner.run_block) -- never set by
# hand, so never shown to the model.
INJECTED_PARAMS = frozenset({"output_dir", "block_id", "sample_rows", "_iteration_index"})

# Blocks an AI build may not add, with the reason shown if it tries:
# anything that touches the world outside the project (files, databases,
# env vars) stays the user's to set up -- the build starts from inputs the
# user prepared -- and the fan-out pair is left out of v1 (pairing by
# block id is easy to get subtly wrong, and neither compiles yet).
AGENT_DISALLOWED: dict[str, str] = {
    "read_csv": "input blocks are prepared by the user, not the AI build",
    "read_parquet": "input blocks are prepared by the user, not the AI build",
    "read_json": "input blocks are prepared by the user, not the AI build",
    "read_excel": "input blocks are prepared by the user, not the AI build",
    "read_sql": "input blocks are prepared by the user, not the AI build",
    "write_csv": "writing files is left to the user",
    "iterate": "fan-out (iterate/collect) isn't supported in AI builds yet",
    "collect": "fan-out (iterate/collect) isn't supported in AI builds yet",
}


def ensure_blocks_registered() -> None:
    """Import every block library module so BLOCK_REGISTRY is complete --
    the API server does this at import time; the agent may also run from
    the MCP server or tests, which don't import api.py."""
    from ..blocks import (  # noqa: F401
        data_quality,
        feature_analysis,
        library,
        modelling,
        stat_tests,
        stochastic,
    )


def _summary_line(doc: str) -> str:
    first = doc.strip().split("\n\n", 1)[0]
    return " ".join(first.split())


def _annotation(param: inspect.Parameter) -> str | None:
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        return None
    return ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def block_params(spec: BlockSpec) -> list[dict[str, Any]]:
    """The block's configurable parameters: everything in its function's
    signature except the dataframe/value arguments its input ports feed
    and the runner-injected ones."""
    port_names = {p.name for p in spec.inputs}
    role_bound = {
        role_param: role.value
        for role in ROLE_PARAM_NAMES
        if (role_param := find_role_param(spec.fn, role)) is not None
    }
    out = []
    for name, param in inspect.signature(spec.fn).parameters.items():
        if name in port_names or name in INJECTED_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        entry: dict[str, Any] = {"name": name, "type": _annotation(param)}
        if param.default is inspect.Parameter.empty:
            entry["required"] = True
        else:
            entry["default"] = _jsonable(param.default)
        if name in role_bound:
            entry["auto_fills_from_role"] = role_bound[name]
        out.append(entry)
    return out


def catalogue_entry(spec: BlockSpec, detail: bool = False) -> dict[str, Any]:
    doc = inspect.getdoc(spec.fn) or ""
    entry: dict[str, Any] = {
        "category": spec.category,
        "display_name": spec.display_name,
        "group": spec.group or spec.block_type,
        "block_type": spec.block_type,
        "summary": _summary_line(doc),
        "inputs": [{"name": p.name, "type": p.type, "required": p.required} for p in spec.inputs],
        "outputs": [{"name": p.name, "type": p.type} for p in spec.outputs],
    }
    if spec.category in AGENT_DISALLOWED:
        entry["agent_can_add"] = False
        entry["why_not"] = AGENT_DISALLOWED[spec.category]
    if detail:
        entry["doc"] = doc
        entry["params"] = block_params(spec)
    return entry


def list_block_types(group: str | None = None, include_disallowed: bool = False) -> list[dict[str, Any]]:
    ensure_blocks_registered()
    out = []
    for spec in BLOCK_REGISTRY.values():
        if not include_disallowed and spec.category in AGENT_DISALLOWED:
            continue
        if group is not None and (spec.group or spec.block_type) != group:
            continue
        out.append(catalogue_entry(spec))
    return out


def describe_block_type(category: str) -> dict[str, Any]:
    ensure_blocks_registered()
    spec = BLOCK_REGISTRY.get(category)
    if spec is None:
        raise KeyError(f"unknown block category {category!r}")
    return catalogue_entry(spec, detail=True)


ROLE_NOTES = {
    ColumnRole.TARGET.value: "the modelling target; params named target/target_col auto-fill from it",
    ColumnRole.PREDICTED.value: "set automatically on model outputs; params named score_col/predicted_col auto-fill from it",
    ColumnRole.EXCLUDED.value: "must never be used as a model feature",
}
