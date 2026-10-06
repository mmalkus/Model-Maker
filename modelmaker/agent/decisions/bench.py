"""Plan five scenarios built to make the menus vary, with a decision backend,
and report how far it agrees with the rule policy.

    python -m modelmaker.agent.decisions.bench --backend haiku
    python -m modelmaker.agent.decisions.bench --backend rules --scenario panel
    python -m modelmaker.agent.decisions.bench --backend laya:router

Scenarios:
- snapshot: the demo data (one row per facility), date + target tagged
- no_dates: the same with no date role -- no trend, no out-of-time split
- few_defaults: 400 rows (~45 defaults) -- no out-of-time split possible
- messy: snapshot plus missing targets and negative values -- exclusions
- panel: a monthly account panel with a default status and no target yet
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import polars as pl

from ... import demo_data
from .backends import get_backend
from .core import Decider
from .planner import plan
from .workspace import Workspace

DEMO_ROLES = {
    "application_id": "id",
    "application_date": "date",
    "default_flag": "target",
}


def panel_frame(n_accounts: int = 400, seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for a in range(n_accounts):
        start = rng.randrange(0, 60)
        life = rng.randrange(18, 84)
        score = rng.gauss(680, 60)
        util = min(max(rng.random() * 0.8, 0.0), 1.2)
        in_default = 0
        overdraft = 0
        for m in range(start, min(start + life, 84)):
            y, mo = divmod(2018 * 12 + m, 12)
            util = min(max(util + rng.gauss(0, 0.05), 0.0), 1.3)
            if not in_default:
                z = -5.2 + (650 - score) / 45 + 1.8 * util
                in_default = int(rng.random() < 1 / (1 + math.exp(-z)))
            overdraft = int(rng.random() < 0.15) if rng.random() < 0.3 else overdraft
            rows.append(
                {
                    "account_id": f"ACC{a:05d}",
                    "month": date(y, mo + 1, 1).isoformat(),
                    "credit_score": round(score),
                    "utilization": round(util, 3),
                    "balance": round(5000 * util + rng.gauss(0, 300), 2),
                    "has_overdraft": overdraft,
                    "in_default": in_default,
                }
            )
    return pl.DataFrame(rows)


def scenarios() -> dict[str, tuple[pl.DataFrame, dict[str, str]]]:
    full = demo_data.to_frame(5000)
    rng = random.Random(3)
    messy = full.with_columns(
        pl.Series(
            "default_flag",
            [None if rng.random() < 0.04 else v for v in full["default_flag"]],
        ),
        pl.Series("dti", [-abs(v) if rng.random() < 0.02 else v for v in full["dti"]]),
    )
    return {
        "snapshot": (full, dict(DEMO_ROLES)),
        "no_dates": (full, {"application_id": "id", "default_flag": "target"}),
        "few_defaults": (demo_data.to_frame(400), dict(DEMO_ROLES)),
        "messy": (messy, dict(DEMO_ROLES)),
        "panel": (panel_frame(), {"account_id": "id", "month": "date"}),
    }


def model_line(result) -> str:
    """The fitted model in one line: features and Gini per sample."""
    fit = next((s for s in result.steps if s["option"] == "fit" and "params" in s), None)
    ev = next((s for s in result.steps if s["option"] == "evaluate" and "result" in s), None)
    if not fit or not ev:
        return "(no model)"
    ginis = ", ".join(f"{k[5:]} {v:.2f}" for k, v in ev["result"].items() if k.startswith("gini_"))
    feats = [f.removesuffix("_woe") for f in fit["params"]["features"]]
    return f"{len(feats)} features [{', '.join(feats)}]; Gini {ginis}; verdict {ev['result']['verdict']}"


def kind(name: str) -> str:
    """Decision family for the report: 'screen:age' -> 'screen'."""
    return name.split(":", 1)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rules")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=None, help="write the full decision log as JSON here")
    args = ap.parse_args(argv)

    backend = get_backend(args.backend)
    all_scen = scenarios()
    names = list(all_scen) if args.scenario == "all" else args.scenario.split(",")
    report: dict[str, dict] = {}
    totals = defaultdict(lambda: [0, 0, 0])  # kind -> [agree, total, low-confidence]
    for name in names:
        df, roles = all_scen[name]
        decider = Decider(backend, max_tokens=args.max_tokens)
        rules_decider = Decider(get_backend("rules"), max_tokens=args.max_tokens)
        t0 = time.time()
        result = plan(Workspace(df, dict(roles)), decider)
        elapsed = time.time() - t0
        rule_result = plan(Workspace(df, dict(roles)), rules_decider)
        path = [f"{s['option']}" for s in result.steps]
        rule_path = [f"{s['option']}" for s in rule_result.steps]
        recs = decider.records
        agree = sum(r.agrees for r in recs)
        print(f"\n=== {name}  ({df.height} rows, {elapsed:.0f}s) ===")
        print(f"model type: {result.model_type}   (rules: {rule_result.model_type})")
        print(f"path : {' -> '.join(path)}")
        print(f"rules: {' -> '.join(rule_path)}" + ("   [same]" if path == rule_path else "   [DIFFERENT]"))
        print(
            f"decisions: {len(recs)}, agree with rules {agree}/{len(recs)}, "
            f"low-confidence (to user) {sum(r.needs_user for r in recs)}, max tokens {max(r.tokens for r in recs)}"
        )
        for r in recs:
            k = kind(r.name)
            totals[k][0] += r.agrees
            totals[k][1] += 1
            totals[k][2] += r.needs_user
            if not r.agrees:
                print(
                    f"  differs  {r.name:34s} {r.qid:13s} got={r.answer.value!s:22s} rules={r.rule_answer!s:22s} conf={r.answer.confidence:.2f}"
                )
        print(f"model: {model_line(result)}")
        if rule_result is not result:
            print(f"rules: {model_line(rule_result)}")
        for s in result.steps:
            if s.get("error"):
                print(f"  ERROR in {s['option']}: {s['error']}")
        report[name] = {
            "plan": result.to_dict(),
            "rule_plan": rule_result.to_dict(),
            "decisions": [r.to_dict() for r in recs],
            "seconds": elapsed,
        }

    print("\n=== by decision type ===")
    for k, (a, n, low) in sorted(totals.items()):
        print(f"  {k:16s} agree {a:3d}/{n:<3d} ({a / n:4.0%})   low-confidence {low}")
    a = sum(v[0] for v in totals.values())
    n = sum(v[1] for v in totals.values())
    print(f"  {'ALL':16s} agree {a:3d}/{n:<3d} ({a / n:4.0%})")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
        print(f"\nfull log: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
