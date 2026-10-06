"""The phases of a PD build and, per phase, its menu of blocks. Each menu
option says when it's on offer (from facts, never the model's say), and how
its parameters are decided -- typed questions, each with the rule the
decision-hint policy would apply (agent/hints.py thresholds).

Phase 0 decides the model type; phases 1-2 are built for PD so far."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from ...blocks.base import BLOCK_REGISTRY
from .. import hints as h
from .core import Ask, Decider, choice, noul
from .workspace import (
    Workspace,
    column_facts,
    dataset_facts,
    history_months,
    status_candidates,
)

DONE = "DONE"


@dataclass
class Option:
    key: str
    describe: str  # the menu criterion for this option, one line
    offered: Callable[[Workspace], bool]
    run: Callable[[Workspace, Decider], dict[str, Any]]  # -> {"block", "params", ...}
    # Required options must run before the phase may end: DONE isn't on
    # the menu while one is on offer. Optional ones are judgement calls.
    required: bool = True


@dataclass
class Phase:
    key: str
    title: str
    done_describe: str
    options: list[Option]
    extra_facts: Callable[[Workspace], dict[str, Any]] = lambda ws: {}


def run_block(category: str, *inputs: Any, **params: Any) -> Any:
    from ..catalogue import ensure_blocks_registered

    ensure_blocks_registered()
    return BLOCK_REGISTRY[category].fn(*inputs, **params)


def _done(ws: Workspace, key: str) -> bool:
    return key in ws.resolved


# ---- phase 0: model type --------------------------------------------------------------

MODEL_TYPE_Q = choice(
    "Which credit-risk model does this data support?",
    {
        "pd": "probability of default: a binary default target, or a panel with a 0/1 default status to build one",
        "lgd": "loss given default: a continuous loss-rate target on defaulted facilities",
        "ccf_ead": "credit conversion factor / exposure at default: a continuous drawn-vs-limit target",
    },
)


def model_type_rule(f: dict[str, Any]) -> str:
    if f["target"] == "binary" or (f["target"] == "missing" and f.get("status_candidates")):
        return "pd"
    return "lgd"


# ---- phase 1: data quality & preparation ------------------------------------------------

SCREEN_Q = choice(
    "Credit-risk PD model data prep. Should this input column be a candidate feature?",
    {
        "use": "a normal, reasonably filled, varying column",
        "exclude_id": "an identifier: text or integer with nearly every value distinct (95%+)",
        "exclude_date": "a date: not a feature itself",
        "exclude_empty": "mostly empty, fill rate below 50%",
        "exclude_constant": "one value covers 95%+ of rows",
        "exclude_post_outcome": "filled only for defaulted rows (or only for non-defaulted): recorded after the outcome, leaks the target",
    },
)


def screen_rule(f: dict[str, Any]) -> str:
    fe, fn = f.get("fill_events"), f.get("fill_non_events")
    if fe is not None and fn is not None and abs(fe - fn) >= 0.85:
        return "exclude_post_outcome"
    if f.get("date_like"):
        return "exclude_date"
    if f["distinct_share"] >= h.ID_LIKE_DISTINCT and (f["dtype"] == "String" or f["dtype"].startswith(("Int", "UInt"))):
        return "exclude_id"
    if f["fill_rate"] < h.LOW_FILL_RATE:
        return "exclude_empty"
    if f.get("top_value_share", 0) >= h.DOMINANT_SHARE:
        return "exclude_constant"
    return "use"


def run_profile(ws: Workspace, d: Decider) -> dict[str, Any]:
    ws.artifacts["profile"] = run_block("data_profile", ws.df)
    return {"block": "data_profile", "params": {}}


def run_screen(ws: Workspace, d: Decider) -> dict[str, Any]:
    cols = ws.cols("unassigned", "feature")
    asks = [
        Ask(
            f"screen:{c}",
            column_facts(ws, c),
            {"screen": SCREEN_Q},
            {"screen": screen_rule},
        )
        for c in cols
    ]
    excluded = {}
    for c, ans in zip(cols, d.ask_many(asks)):
        verdict = ans["screen"]
        ws.roles[c] = "feature" if verdict == "use" else "excluded"
        if verdict != "use":
            excluded[c] = verdict
    return {
        "block": None,
        "action": "set column roles",
        "params": {"excluded": excluded},
    }


NON_NEGATIVE_Q = noul("Should this column never be negative (an amount, count, ratio, age or score)?")
UNIT_Q = noul("Is this column a share or rate that must lie between 0 and 1?")


def run_dq_rules(ws: Workspace, d: Decider) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    id_col, date_col, target = ws.col("id"), ws.col("date"), ws.col("target")
    if id_col:
        key = [id_col, date_col] if dataset_facts(ws)["panel"] and date_col else [id_col]
        rules.append({"name": "unique key", "rule": "unique", "columns": key})
    if target:
        rules.append({"name": "target not null", "rule": "not_null", "column": target})
    numeric = [c for c in ws.cols("feature") if ws.df[c].dtype.is_numeric()]
    asks = [
        Ask(
            f"dq:{c}",
            {
                k: v
                for k, v in column_facts(ws, c).items()
                if k in ("column", "dtype", "min", "max", "negative_share", "distinct_share", "binary")
            },
            {"non_negative": NON_NEGATIVE_Q, "unit_interval": UNIT_Q},
            # A few negatives (under 5%) are bad values, not a signed column.
            {
                "non_negative": lambda f: f.get("negative_share", 0) < 0.05,
                # A 0/1 flag is in [0,1] but isn't a share.
                "unit_interval": lambda f: f.get("min", -1) >= 0 and f.get("max", 2) <= 1 and not f.get("binary"),
            },
        )
        for c in numeric
    ]
    for c, ans in zip(numeric, d.ask_many(asks)):
        if ans["unit_interval"]:
            rules.append(
                {
                    "name": f"{c} in [0,1]",
                    "rule": "between",
                    "column": c,
                    "min": 0,
                    "max": 1,
                    "severity": "warning",
                }
            )
        elif ans["non_negative"]:
            rules.append(
                {
                    "name": f"{c} >= 0",
                    "rule": "between",
                    "column": c,
                    "min": 0,
                    "severity": "warning",
                }
            )
    result = run_block("data_quality_rules", ws.df, rules=rules)
    ws.artifacts["dq"] = result
    failed = [r["name"] for r in result["results"] if not r["passed"]]
    return {
        "block": "data_quality_rules",
        "params": {"rules": rules},
        "result": {"passed": result["passed"], "failed": failed},
    }


def exclusion_templates(ws: Workspace) -> list[dict[str, Any]]:
    """Standard exclusions that apply to this data, with the share each drops."""
    out = []
    target = ws.col("target")
    n = max(ws.df.height, 1)
    if target and ws.df[target].null_count():
        out.append(
            {
                "name": "target missing",
                "expr": f'"{target}" IS NOT NULL',
                "drops": ws.df[target].null_count() / n,
            }
        )
    for r in (ws.artifacts.get("dq") or {}).get("results", []):
        col = r["detail"].removeprefix("column=")
        if r["rule"] == "between" and not r["passed"] and col in ws.df.columns:
            lo = 0
            hi = 1 if r["name"].endswith("[0,1]") else None
            expr = f'"{col}" IS NULL OR ("{col}" >= {lo}' + (f' AND "{col}" <= {hi})' if hi is not None else ")")
            out.append(
                {
                    "name": f"{col} out of range",
                    "expr": expr,
                    "drops": r["violations"] / n,
                }
            )
    return out


APPLY_EXCLUSION_Q = noul(
    "Apply this exclusion to the modelling population? Usual practice: yes for invalid or out-of-scope rows, "
    "unless it drops 20%+ of the population (then the data needs a look first)."
)


def run_exclusions(ws: Workspace, d: Decider) -> dict[str, Any]:
    templates = exclusion_templates(ws)
    asks = [
        Ask(
            f"exclude:{t['name']}",
            {"exclusion": t["name"], "share_of_rows_dropped": t["drops"]},
            {"apply": APPLY_EXCLUSION_Q},
            {"apply": lambda f: f["share_of_rows_dropped"] < h.BIG_EXCLUSION},
        )
        for t in templates
    ]
    chosen = [{"name": t["name"], "expr": t["expr"]} for t, a in zip(templates, d.ask_many(asks)) if a["apply"]]
    if not chosen:
        return {"block": None, "action": "declined all exclusions", "params": {}}
    ws.df, summary = run_block("apply_exclusions", ws.df, rules=chosen)
    return {
        "block": "apply_exclusions",
        "params": {"rules": chosen},
        "result": {"dropped": summary["total_dropped"]},
    }


MISSING_Q = choice(
    "How should this feature's missing values be treated before WoE binning?",
    {
        "flag": "leave as missing: binning gives missing its own bin (the default when 1%+ is missing)",
        "median": "fill with the median: numeric, only a handful missing (under 1%)",
        "mode": "fill with the most common value: text, only a handful missing (under 1%)",
        "drop_rows": "drop the rows: missing means the record is unusable",
    },
)
INDICATOR_Q = noul("Add a was-missing 0/1 indicator column (worth it when 5%+ is missing and the values get filled)?")


def missing_rule(f: dict[str, Any]) -> str:
    if 1 - f["fill_rate"] >= 0.01:
        return "flag"
    return "median" if f["dtype"].startswith(("Int", "UInt", "Float")) else "mode"


def run_missing(ws: Workspace, d: Decider) -> dict[str, Any]:
    cols = [c for c in ws.cols("feature") if ws.df[c].null_count()]
    asks = [
        Ask(
            f"missing:{c}",
            {k: v for k, v in column_facts(ws, c).items() if k in ("column", "dtype", "fill_rate")},
            {"method": MISSING_Q, "indicator": INDICATOR_Q},
            {
                "method": missing_rule,
                "indicator": lambda f: 1 - f["fill_rate"] >= 0.05 and missing_rule(f) != "flag",
            },
        )
        for c in cols
    ]
    strategies = {}
    for c, a in zip(cols, d.ask_many(asks)):
        method = a["method"]
        if method in ("median",) and not ws.df[c].dtype.is_numeric():
            method = "mode"  # a model's slip the block would reject
        strategies[c] = {
            "method": method,
            "add_indicator": bool(a["indicator"]) and method != "flag",
        }
    ws.df = run_block("missing_value_treatment", ws.df, strategies=strategies)
    for c in ws.df.columns:
        ws.roles.setdefault(c, "feature")  # *_was_missing indicators
    return {"block": "missing_value_treatment", "params": {"strategies": strategies}}


def phase1_facts(ws: Workspace) -> dict[str, Any]:
    feats = ws.cols("feature")
    return {
        "features": len(feats) if _done(ws, "screen") else None,
        "features_with_nulls": sum(1 for c in feats if ws.df[c].null_count()) if _done(ws, "screen") else None,
        "dq_failed_rules": [r["name"] for r in (ws.artifacts.get("dq") or {}).get("results", []) if not r["passed"]],
    }


PHASE1 = Phase(
    "data_prep",
    "Data quality & preparation",
    "the data is profiled, every column screened, quality rules checked, applicable exclusions decided, "
    "and no kept feature has untreated missing values",
    [
        Option(
            "profile",
            "data_profile: profile every column (fill, distinct, dominant value); always first",
            lambda ws: not _done(ws, "profile"),
            run_profile,
        ),
        Option(
            "screen",
            "column screen: decide per column whether it's a candidate feature (needs the profile)",
            lambda ws: _done(ws, "profile") and not _done(ws, "screen"),
            run_screen,
        ),
        Option(
            "dq_rules",
            "data_quality_rules: assert key uniqueness, target not null, value ranges (needs the screen)",
            lambda ws: _done(ws, "screen") and not _done(ws, "dq_rules"),
            run_dq_rules,
        ),
        Option(
            "exclusions",
            "apply_exclusions: drop invalid rows (missing target, out-of-range values) with a waterfall",
            lambda ws: _done(ws, "screen") and not _done(ws, "exclusions") and bool(exclusion_templates(ws)),
            run_exclusions,
            required=False,
        ),
        Option(
            "missing",
            "missing_value_treatment: decide per feature how its missing values are handled",
            lambda ws: (
                _done(ws, "screen")
                and not _done(ws, "missing")
                and any(ws.df[c].null_count() for c in ws.cols("feature"))
            ),
            run_missing,
        ),
    ],
    phase1_facts,
)


# ---- phase 2: target & sampling (PD) -----------------------------------------------------

HORIZON_Q = choice(
    "PD default horizon: how far ahead does the target look for a default?",
    {
        "12m": "12 months: the regulatory (IRB / IFRS 9 stage 1) PD",
        "24m": "24 months",
        "36m": "36 months",
    },
)
PERFORMING_Q = noul("Keep only accounts performing (not in default) at the observation date, the usual PD population?")
INCOMPLETE_Q = choice(
    "Rows whose outcome window isn't complete yet (history ends before the horizon, no default seen):",
    {
        "drop": "drop them: their outcome is unknown (usual)",
        "null": "keep them with a missing target",
    },
)


def default_col_question(cands: list[str]) -> dict[str, Any]:
    return choice(
        "Which 0/1 column is the account's monthly default status (1 = in default that month)?",
        {c: f"column {c}" for c in cands},
    )


def run_forward_flag(ws: Workspace, d: Decider) -> dict[str, Any]:
    cands = status_candidates(ws)
    state = {
        "candidates": {
            c: {
                "share_of_ones": float(ws.df[c].mean() or 0),
                "ever_one_then_back_to_zero": _reverts(ws, c),
            }
            for c in cands
        }
    }
    questions = {
        "horizon": HORIZON_Q,
        "performing": PERFORMING_Q,
        "incomplete": INCOMPLETE_Q,
    }
    rules = {
        "horizon": lambda f: "12m",
        "performing": lambda f: True,
        "incomplete": lambda f: "drop",
    }
    if len(cands) > 1:
        questions["default_col"] = default_col_question(cands)
        # A default status rarely flips back to 0; a product flag does.
        rules["default_col"] = lambda f: min(
            f["candidates"],
            key=lambda c: (f["candidates"][c]["ever_one_then_back_to_zero"], c),
        )
    a = d.ask("forward_flag", state, questions, rules)
    default_col = a.get("default_col", cands[0])
    params = {
        "id_col": ws.col("id"),
        "date_col": ws.col("date"),
        "default_col": default_col,
        "horizon": int(a["horizon"].removesuffix("m")),
        "performing_only": bool(a["performing"]),
        "incomplete": a["incomplete"],
    }
    ws.df = run_block("forward_default_flag", ws.df, **params)
    flag = f"default_{params['horizon']}m"
    ws.roles[default_col] = "excluded"  # the status is the outcome, never a feature
    ws.roles[flag] = "target"
    return {"block": "forward_default_flag", "params": params}


def _reverts(ws: Workspace, col: str) -> float:
    """Share of accounts whose `col` goes 1 -> 0 at some point."""
    id_col, date_col = ws.col("id"), ws.col("date")
    if not (id_col and date_col):
        return 0.0
    g = (
        ws.df.select(id_col, date_col, col)
        .sort(id_col, date_col)
        .with_columns((pl.col(col).shift(1).over(id_col) == 1).and_(pl.col(col) == 0).alias("__back"))
        .group_by(id_col)
        .agg(pl.col("__back").any())
    )
    return float(g["__back"].mean() or 0.0)


PERIOD_Q = choice(
    "Period for the target-over-time check:",
    {
        "month": "monthly: under 3 years of history",
        "quarter": "quarterly: 3-10 years of history",
        "year": "yearly: over 10 years",
    },
)


def period_rule(f: dict[str, Any]) -> str:
    m = f.get("history_months") or 0
    return "month" if m < 36 else "quarter" if m <= 120 else "year"


def run_trend(ws: Workspace, d: Decider) -> dict[str, Any]:
    f = {"history_months": history_months(ws), "rows": ws.df.height}
    period = d.ask("trend_period", f, {"period": PERIOD_Q}, {"period": period_rule})["period"]
    table = run_block(
        "target_trend",
        ws.df,
        date_col=ws.col("date"),
        target_col=ws.col("target"),
        period=period,
    )
    ws.artifacts["trend"] = table
    ws.artifacts["trend_period"] = period
    return {
        "block": "target_trend",
        "params": {"period": period},
        "result": immature_tail(ws),
    }


def immature_tail(ws: Workspace) -> dict[str, Any] | None:
    """The trend's trailing periods that are thin or whose default rate
    collapses (outcomes not yet seen) -- the rows to consider dropping."""
    t = ws.artifacts.get("trend")
    if t is None or t.height < 4:
        return None
    t = t.sort("period") if "period" in t.columns else t
    label = t.columns[0]
    median_n = float(t["n"].median())
    median_rate = float(t["target_mean"].median())
    tail = []
    for row in reversed(t.to_dicts()):
        thin = row["n"] < h.IMMATURE_LAST_PERIOD * median_n
        collapsed = (row["target_mean"] or 0) < 0.5 * median_rate
        if not (thin or collapsed):
            break
        tail.append(row)
    if not tail:
        return None
    first = tail[-1][label]
    return {
        "from_period": first,
        "periods": len(tail),
        "rows": int(sum(r["n"] for r in tail)),
        "rate": float(sum(r["n"] * (r["target_mean"] or 0) for r in tail) / max(sum(r["n"] for r in tail), 1)),
        "median_rate": median_rate,
    }


def _period_start(label: str, period: str) -> str:
    y = int(label[:4])
    if period == "month":
        return f"{label}-01"
    if period == "quarter":
        return date(y, 3 * (int(label[-1]) - 1) + 1, 1).isoformat()
    return f"{y}-01-01"


DROP_IMMATURE_Q = noul(
    "The latest periods look immature (few rows, or a default rate far below usual because outcomes haven't "
    "had time to show). Drop them from the modelling data?"
)


def run_drop_immature(ws: Workspace, d: Decider) -> dict[str, Any]:
    tail = immature_tail(ws)
    state = {k: tail[k] for k in ("periods", "rows", "rate", "median_rate")} | {
        "share_of_rows": tail["rows"] / ws.df.height
    }
    if not d.ask(
        "drop_immature",
        state,
        {"drop": DROP_IMMATURE_Q},
        {"drop": lambda f: f["share_of_rows"] < 0.3},
    )["drop"]:
        return {"block": None, "action": "kept the latest periods", "params": {}}
    cutoff = _period_start(str(tail["from_period"]), ws.artifacts["trend_period"])
    expr = f"\"{ws.col('date')}\" < '{cutoff}'"
    ws.df = run_block("filter", ws.df, expr=expr)
    return {"block": "filter", "params": {"expr": expr}}


def oot_windows(ws: Workspace) -> dict[str, dict[str, Any]]:
    """Candidate out-of-time windows (the last 6/12/24 months), with their
    events and the share of rows they take."""
    d, target = ws.dates(), ws.col("target")
    if d is None or target is None:
        return {}
    hi = d.max()
    out = {}
    for months in (6, 12, 24):
        y, m = divmod(hi.year * 12 + hi.month - 1 - months + 1, 12)
        cutoff = date(y, m + 1, 1)
        mask = d >= cutoff
        out[f"last_{months}m"] = {
            "cutoff": cutoff.isoformat(),
            "events": int(ws.df[target].filter(mask).sum() or 0),
            "share_of_rows": float(mask.mean() or 0),
        }
    return out


def oot_rule(f: dict[str, Any]) -> str:
    ok = [k for k, w in f["windows"].items() if w["events"] >= h.MIN_EVENTS and w["share_of_rows"] <= 0.4]
    return ok[0] if ok else "last_12m"


def run_time_split(ws: Workspace, d: Decider) -> dict[str, Any]:
    windows = oot_windows(ws)
    q = choice(
        "Out-of-time validation window: the most recent data held back. It needs 50+ defaults to be a stable test, "
        "and should leave most data (60%+) for development; prefer the shortest window that does both.",
        {
            k: f"from {w['cutoff']}: {w['events']} defaults, {w['share_of_rows']:.0%} of rows"
            for k, w in windows.items()
        },
    )
    state = {"windows": {k: {"events": w["events"], "share_of_rows": w["share_of_rows"]} for k, w in windows.items()}}
    pick = d.ask("oot_window", state, {"window": q}, {"window": oot_rule})["window"]
    params = {"date_col": ws.col("date"), "cutoff": windows[pick]["cutoff"]}
    ws.df, ws.artifacts["out_of_time"] = run_block("time_split", ws.df, **params)
    return {
        "block": "time_split",
        "params": params,
        "result": {"oot_rows": ws.artifacts["out_of_time"].height},
    }


TEST_SIZE_Q = choice(
    "Share of the development data held out as the test sample:",
    {
        "0.2": "20%: enough when the test sample still gets 50+ defaults",
        "0.3": "30%: when 20% would leave under 50 defaults",
    },
)
STRATIFY_Q = noul(
    "Stratify the split on the target, so train and test get the same default rate (usual when defaults are rare)?"
)


def run_train_test(ws: Workspace, d: Decider) -> dict[str, Any]:
    target = ws.col("target")
    f = {
        "rows": ws.df.height,
        "events": int(ws.df[target].sum() or 0),
        "event_rate": float(ws.df[target].mean() or 0),
    }
    a = d.ask(
        "train_test",
        f,
        {"test_size": TEST_SIZE_Q, "stratify": STRATIFY_Q},
        {
            "test_size": lambda f: "0.3" if f["events"] * 0.2 < h.MIN_EVENTS else "0.2",
            "stratify": lambda f: f["event_rate"] < 0.2,
        },
    )
    params = {
        "test_size": float(a["test_size"]),
        "seed": 0,
        "stratify_col": target if a["stratify"] else None,
    }
    ws.df, ws.artifacts["test"] = run_block("train_test_split", ws.df, **params)
    return {"block": "train_test_split", "params": params}


def _has_target(ws: Workspace) -> bool:
    return ws.col("target") is not None


def phase2_facts(ws: Workspace) -> dict[str, Any]:
    tail = immature_tail(ws) if _done(ws, "trend") and not _done(ws, "drop_immature") else None
    windows = oot_windows(ws) if ws.col("date") and _has_target(ws) and not _done(ws, "time_split") else {}
    return {
        "target_trend_checked": _done(ws, "trend") if ws.col("date") and _has_target(ws) else None,
        "immature_latest_periods": bool(tail),
        "oot_window_with_50_defaults": any(w["events"] >= h.MIN_EVENTS for w in windows.values()) if windows else None,
        "validation_sample": "out_of_time" in ws.artifacts or "test" in ws.artifacts,
    }


PHASE2 = Phase(
    "target_sampling",
    "Target & sampling",
    "a target exists, its trend over time is checked (when there's a date), immature periods are dealt with, "
    "and development/validation samples are cut",
    [
        Option(
            "forward_flag",
            "forward_default_flag: build the 12m-ahead default target from a monthly panel's default status",
            lambda ws: (
                not _has_target(ws)
                and dataset_facts(ws)["panel"]
                and bool(ws.col("date"))
                and bool(status_candidates(ws))
                and not _done(ws, "forward_flag")
            ),
            run_forward_flag,
        ),
        Option(
            "trend",
            "target_trend: default rate over time, spots drift and immature recent periods",
            lambda ws: _has_target(ws) and bool(ws.col("date")) and not _done(ws, "trend"),
            run_trend,
        ),
        Option(
            "drop_immature",
            "filter: drop immature latest periods the trend flagged",
            lambda ws: (
                _done(ws, "trend")
                and not _done(ws, "drop_immature")
                and not _done(ws, "time_split")
                and immature_tail(ws) is not None
            ),
            run_drop_immature,
            required=False,
        ),
        Option(
            "time_split",
            "time_split: hold back the latest period as out-of-time validation, before any train/test split",
            lambda ws: (
                _has_target(ws)
                and bool(ws.col("date"))
                and not _done(ws, "time_split")
                # Out-of-time comes off first: after a train/test split it
                # would only cut the training sample.
                and not _done(ws, "train_test")
                and any(w["events"] >= h.MIN_EVENTS and w["share_of_rows"] <= 0.4 for w in oot_windows(ws).values())
            ),
            run_time_split,
            required=False,
        ),
        Option(
            "train_test",
            "train_test_split: random development/test split of the (development) data",
            lambda ws: _has_target(ws) and not _done(ws, "train_test"),
            run_train_test,
        ),
    ],
    phase2_facts,
)


def _pd_phases() -> list[Phase]:
    from .modelling_phases import PHASE3, PHASE4, PHASE5

    return [PHASE1, PHASE2, PHASE3, PHASE4, PHASE5]


PHASES = {"pd": _pd_phases}
