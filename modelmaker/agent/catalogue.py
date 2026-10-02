"""The block catalogue an AI build chooses from -- what each registry block
does, its ports, and its parameters, derived from the registry itself
(BlockSpec + the block function's signature and docstring) so there's no
second list to keep in sync."""

from __future__ import annotations

import inspect
import re
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
    "demo_credit_data": "input blocks are prepared by the user, not the AI build",
    "write_csv": "writing files is left to the user",
    "iterate": "fan-out (iterate/collect) isn't supported in AI builds yet",
    "collect": "fan-out (iterate/collect) isn't supported in AI builds yet",
}


# The tags an AI build browses the catalogue by, in the order the compact
# system prompt lists them. Each block declares its own tags (BlockSpec.tags):
# what it does, and which risk model it's for. list_block_types(tag=...) lists
# one tag's blocks with a one-sentence summary, so a small local model never
# has to read the whole catalogue at once. Every block an AI build may add
# needs at least one tag; keep each tag to ~10 blocks so its listing stays
# well inside a local model's tool-result cap (both checked by
# tests/test_agent_foundations.py).
TAGS: dict[str, str] = {
    "data_prep": "row/column transformations: derive, filter, select, join, aggregate, encode, treat missings, exclusions",
    "sampling": "train/test and out-of-time splits",
    "data_quality": "profiling, data quality rules, missing values, exclusion waterfalls, target over time",
    "binning_woe": "coarse classing and weight-of-evidence encoding",
    "feature_selection": "univariate screens, multicollinearity, stepwise selection",
    "regression": "fitting, applying and calibrating regression models",
    "scorecard": "points-based scorecards built on binned/WoE features",
    "rating_scale": "master scales, rating grades and grade-level checks",
    "performance": "discrimination and accuracy: AUC/Gini, KS, ROC, sample comparison, continuous accuracy",
    "calibration": "calibration tests and adjustments: Hosmer-Lemeshow, backtests, long-run average, MoC",
    "stability": "population and characteristic stability, target trend",
    "pd": "probability of default models: fit, calibrate, grade, backtest",
    "lgd": "loss given default: recoveries, LGD target, fractional logit, validation",
    "ccf_ead": "credit conversion factor and exposure at default",
    "capital_simulation": "economic capital: ASRF, Monte Carlo credit/op-risk simulation, aggregation, VaR/TVaR",
    "distributions": "fit and sample parametric distributions, dependency structures, risk measures",
    "proxy_models": "surrogate valuation functions: fit, evaluate, validate, var-covar",
    "output": "tables, values and charts to report",
}


def _tagged(tag: str) -> list[BlockSpec]:
    return [
        spec for spec in BLOCK_REGISTRY.values() if tag in spec.tags and spec.category not in AGENT_DISALLOWED
    ]


def list_tags() -> list[dict[str, Any]]:
    ensure_blocks_registered()
    return [{"tag": tag, "blocks": len(_tagged(tag)), "about": about} for tag, about in TAGS.items()]


def ensure_blocks_registered() -> None:
    """Import every block library module so BLOCK_REGISTRY is complete --
    the API server does this at import time; the agent may also run from
    the MCP server or tests, which don't import api.py."""
    from ..blocks import (  # noqa: F401
        binning,
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


def _first_sentence(summary: str, limit: int = 220) -> str:
    """The first sentence of a summary line, for tag listings -- the whole
    first paragraph runs to ~1k chars on some blocks."""
    sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\s+--\s+", summary, maxsplit=1)[0]
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + "..."
    return sentence


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
        "tags": list(spec.tags),
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


def list_blocks_for_tag(tag: str) -> list[dict[str, Any]]:
    """One tag's blocks, each with a one-sentence summary -- ports and
    params come from describe_block_type."""
    if tag not in TAGS:
        raise KeyError(f"unknown tag {tag!r}; tags are: {', '.join(TAGS)}")
    ensure_blocks_registered()
    return [
        {"category": spec.category, "summary": _first_sentence(_summary_line(inspect.getdoc(spec.fn) or ""))}
        for spec in _tagged(tag)
    ]


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
