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


# Tags an AI build browses the catalogue by: list_block_types(tag=...) lists
# one tag's blocks with a one-sentence summary, so a small local model never
# has to read the whole catalogue (or a broad palette group) at once. A block
# can carry several tags -- what it does, and which risk model it's for. Every
# block an AI build may add needs at least one; keep each tag to ~10 blocks so
# its listing stays well inside a local model's tool-result cap (both are
# checked by tests/test_agent_foundations.py).
AGENT_TAGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "data_prep": (
        "row/column transformations: derive, filter, select, join, aggregate, encode, treat missings, exclusions",
        ("derive_columns", "filter", "select", "join", "groupby_agg", "one_hot_encode",
         "missing_value_treatment", "apply_exclusions"),
    ),
    "sampling": (
        "train/test and out-of-time splits",
        ("train_test_split", "time_split"),
    ),
    "data_quality": (
        "profiling, data quality rules, missing values, exclusion waterfalls, target over time",
        ("data_profile", "data_quality_rules", "missing_value_treatment", "apply_exclusions", "target_trend"),
    ),
    "binning_woe": (
        "coarse classing and weight-of-evidence encoding",
        ("fit_binning", "apply_binning", "woe_transform", "bin_chart"),
    ),
    "feature_selection": (
        "univariate screens, multicollinearity, stepwise selection",
        ("iv_table", "correlation_matrix", "stepwise_selection", "characteristic_stability", "fit_binning"),
    ),
    "regression": (
        "fitting, applying and calibrating regression models",
        ("logistic_regression", "glm_fit", "lgd_regression", "stepwise_selection", "predict", "calibrate_model"),
    ),
    "scorecard": (
        "points-based scorecards built on binned/WoE features",
        ("scorecard_scale", "scorecard_table", "fit_binning", "apply_binning", "woe_transform", "logistic_regression"),
    ),
    "rating_scale": (
        "master scales, rating grades and grade-level checks",
        ("fit_master_scale", "assign_rating_grade", "rating_summary", "grade_backtest"),
    ),
    "performance": (
        "discrimination and accuracy: AUC/Gini, KS, ROC, sample comparison, continuous accuracy",
        ("auc_gini", "ks_test", "roc_curve", "compare_samples", "continuous_accuracy"),
    ),
    "calibration": (
        "calibration tests and adjustments: Hosmer-Lemeshow, backtests, long-run average, MoC",
        ("calibration_test", "bucketed_calibration", "calibrate_model", "grade_backtest",
         "margin_of_conservatism", "long_run_average"),
    ),
    "stability": (
        "population and characteristic stability, target trend",
        ("psi_test", "characteristic_stability", "target_trend", "compare_samples"),
    ),
    "pd": (
        "probability of default models: fit, calibrate, grade, backtest",
        ("logistic_regression", "calibrate_model", "calibration_test", "grade_backtest", "fit_master_scale",
         "assign_rating_grade", "rating_summary", "target_trend", "margin_of_conservatism", "long_run_average"),
    ),
    "lgd": (
        "loss given default: recoveries, LGD target, fractional logit, validation",
        ("lgd_regression", "discount_recoveries", "compute_lgd", "long_run_average", "margin_of_conservatism",
         "calibrate_model", "bucketed_calibration", "continuous_accuracy"),
    ),
    "ccf_ead": (
        "credit conversion factor and exposure at default",
        ("compute_ccf", "compute_ead", "lgd_regression", "long_run_average", "margin_of_conservatism",
         "bucketed_calibration", "continuous_accuracy"),
    ),
    "capital_simulation": (
        "economic capital: ASRF, Monte Carlo credit/op-risk simulation, aggregation, VaR/TVaR",
        ("simulate_credit_portfolio", "simulate_op_risk_lda", "aggregate_simulation", "asrf_economic_capital",
         "risk_measures", "build_dependency", "var_covar_aggregate"),
    ),
    "distributions": (
        "fit and sample parametric distributions, dependency structures, risk measures",
        ("fit_distribution", "sample_distribution", "build_dependency", "risk_measures"),
    ),
    "proxy_models": (
        "surrogate valuation functions: fit, evaluate, validate, var-covar",
        ("fit_proxy", "evaluate_proxy", "validate_proxy", "var_covar_aggregate"),
    ),
    "output": (
        "tables, values and charts to report",
        ("display_table", "display_value", "generate_image", "bin_chart", "roc_curve"),
    ),
}


def tags_for(category: str) -> list[str]:
    return [tag for tag, (_, cats) in AGENT_TAGS.items() if category in cats]


def list_tags() -> list[dict[str, Any]]:
    return [{"tag": tag, "blocks": len(cats), "about": about} for tag, (about, cats) in AGENT_TAGS.items()]


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
        "tags": tags_for(spec.category),
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
    if tag not in AGENT_TAGS:
        raise KeyError(f"unknown tag {tag!r}; tags are: {', '.join(AGENT_TAGS)}")
    ensure_blocks_registered()
    out = []
    for category in AGENT_TAGS[tag][1]:
        doc = inspect.getdoc(BLOCK_REGISTRY[category].fn) or ""
        out.append({"category": category, "summary": _first_sentence(_summary_line(doc))})
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
