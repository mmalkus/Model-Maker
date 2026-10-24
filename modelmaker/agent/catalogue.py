"""The block catalogue an AI build chooses from -- what each built-in block
does, its ports, and its parameters, derived from the block definitions
themselves (BLOCKS entry + the function's signature and docstring) so
there's no second list to keep in sync."""

from __future__ import annotations

import inspect
import re
from typing import Any

from ..block_library import INJECTED_PARAMS, load_builtin
from ..blocks.base import BLOCK_REGISTRY, BlockDefinition
from ..packet import ColumnRole
from ..util import ROLE_PARAM_NAMES, find_role_param

__all__ = ["INJECTED_PARAMS"]  # re-exported: the runner-injected params, never shown to the model

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
# system prompt lists them. Each block declares its own tags (BLOCKS "tags"):
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


# The pseudo-tag listing every block in the user's own block library --
# blocks they wrote, or that earlier AI builds created (drafts included).
# Library blocks also appear under whatever real tags they declare.
LIBRARY_TAG = "library"
LIBRARY_ABOUT = "the user's own block library: blocks they (or earlier AI builds) wrote -- check here before creating a new block"


def library_blocks() -> dict[str, BlockDefinition]:
    """Library blocks an AI build may add: valid, requirements installed,
    not an input block (inputs are the user's to prepare). Keyed by id;
    their `origin` is their library file."""
    from .. import block_library, library_store

    out: dict[str, BlockDefinition] = {}
    for f in library_store.list_files():
        if f.parsed.file_errors:
            continue
        for d in f.parsed.definitions:
            if f.parsed.errors.get(d.id) or d.block_type == "input" or d.id in BLOCK_REGISTRY or d.id in out:
                continue
            if block_library.check_requirements(d):
                continue
            out[d.id] = d
    return out


def available(include_library: bool = True) -> dict[str, BlockDefinition]:
    """Every block an AI build can know about: the built-ins, plus (unless
    the build turned it off) the user's library blocks."""
    ensure_blocks_registered()
    return {**BLOCK_REGISTRY, **(library_blocks() if include_library else {})}


def lookup(category: str, include_library: bool = True) -> BlockDefinition | None:
    ensure_blocks_registered()
    if category in BLOCK_REGISTRY:
        return BLOCK_REGISTRY[category]
    return library_blocks().get(category) if include_library else None


def _tagged(tag: str, include_library: bool = True) -> list[BlockDefinition]:
    if tag == LIBRARY_TAG:
        return list(library_blocks().values()) if include_library else []
    return [
        spec
        for spec in available(include_library).values()
        if tag in spec.tags and spec.category not in AGENT_DISALLOWED
    ]


def list_tags(include_library: bool = True) -> list[dict[str, Any]]:
    ensure_blocks_registered()
    tags = [{"tag": tag, "blocks": len(_tagged(tag, include_library)), "about": about} for tag, about in TAGS.items()]
    n = len(_tagged(LIBRARY_TAG, include_library))
    if n:
        tags.append({"tag": LIBRARY_TAG, "blocks": n, "about": LIBRARY_ABOUT})
    return tags


def tag_block_names(tag: str, include_library: bool = True) -> list[str]:
    """A tag's addable block categories, by name only."""
    ensure_blocks_registered()
    return [spec.category for spec in _tagged(tag, include_library)]


def unknown_category_hint(category: str, include_library: bool = True) -> str:
    """Why a category isn't a block, and what to use instead: a tag name
    (a common mix-up) lists its blocks; anything else, the closest names."""
    import difflib

    ensure_blocks_registered()
    if category in TAGS or category == LIBRARY_TAG:
        return f" -- that's a tag, not a block; its blocks are: {', '.join(tag_block_names(category, include_library))}"
    names = [c for c in available(include_library) if c not in AGENT_DISALLOWED]
    close = difflib.get_close_matches(str(category), names, n=3, cutoff=0.5)
    return f" -- did you mean {', '.join(close)}?" if close else " -- list_block_types(tag=...) lists the blocks"


def ensure_blocks_registered() -> None:
    """Make sure BLOCK_REGISTRY holds every built-in (a no-op once
    `import modelmaker` has loaded them, which it always does)."""
    load_builtin()


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


def block_params(spec: BlockDefinition) -> list[dict[str, Any]]:
    """The block's configurable parameters: everything in its function's
    signature except the dataframe/value arguments its input ports feed
    and the runner-injected ones."""
    port_names = {p.name for p in spec.inputs}
    role_bound = {
        role_param: role.value
        for role in ROLE_PARAM_NAMES
        if (role_param := find_role_param(spec.signature_fn, role)) is not None
    }
    out = []
    for name, param in inspect.signature(spec.signature_fn).parameters.items():
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


def catalogue_entry(spec: BlockDefinition, detail: bool = False) -> dict[str, Any]:
    doc = spec.description
    entry: dict[str, Any] = {
        "category": spec.category,
        "display_name": spec.display_name,
        "group": spec.group or spec.block_type,
        "block_type": spec.block_type,
        "tags": list(spec.tags),
        "summary": _summary_line(doc),
        "inputs": [{"name": p.name, "type": p.type, "required": p.required} for p in spec.inputs],
        "outputs": [
            {"name": p.name, "type": p.type, **({"statistics_table": True} if p.name in spec.aggregate_outputs else {})}
            for p in spec.outputs
        ],
    }
    if spec.source == "library":
        entry["library_file"] = spec.origin
        entry["version"] = spec.version
    if spec.category in AGENT_DISALLOWED:
        entry["agent_can_add"] = False
        entry["why_not"] = AGENT_DISALLOWED[spec.category]
    if detail:
        entry["doc"] = doc
        entry["params"] = block_params(spec)
    return entry


def list_block_types(
    group: str | None = None, include_disallowed: bool = False, include_library: bool = True
) -> list[dict[str, Any]]:
    ensure_blocks_registered()
    out = []
    for spec in available(include_library).values():
        if not include_disallowed and spec.category in AGENT_DISALLOWED:
            continue
        if group is not None and (spec.group or spec.block_type) != group:
            continue
        out.append(catalogue_entry(spec))
    return out


def list_blocks_for_tag(tag: str, include_library: bool = True) -> list[dict[str, Any]]:
    """One tag's blocks, each with a one-sentence summary -- ports and
    params come from describe_block_type."""
    if tag not in TAGS and not (tag == LIBRARY_TAG and include_library):
        raise KeyError(f"unknown tag {tag!r}; tags are: {', '.join(t['tag'] for t in list_tags(include_library))}")
    ensure_blocks_registered()
    return [
        {
            "category": spec.category,
            "summary": _first_sentence(spec.summary or spec.display_name),
            **({"library": True} if spec.source == "library" else {}),
        }
        for spec in _tagged(tag, include_library)
    ]


def describe_block_type(category: str, include_library: bool = True) -> dict[str, Any]:
    ensure_blocks_registered()
    spec = lookup(category, include_library)
    if spec is None:
        raise KeyError(f"unknown block category {category!r}")
    return catalogue_entry(spec, detail=True)


ROLE_NOTES = {
    ColumnRole.TARGET.value: "the modelling target; params named target/target_col auto-fill from it",
    ColumnRole.PREDICTED.value: "set automatically on model outputs; params named score_col/predicted_col auto-fill from it",
    ColumnRole.EXCLUDED.value: "must never be used as a model feature",
}
