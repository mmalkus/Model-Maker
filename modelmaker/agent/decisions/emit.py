"""From a decision plan to the canvas: the planner decides on the anchor
block's actual output, then this lays the same steps out as real, wired
registry blocks, one lane per phase -- so the model is an ordinary graph
the person can inspect, edit, run and compile, with an AI badge on every
block and the decision log as its build report.

All of it is one session transaction: one Undo removes the whole build."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from ...packet import ColumnRole, DataFramePacket
from ..layout import Placer, next_lane_order
from .core import Decider
from .planner import PlanResult
from .workspace import Workspace

PHASE_LANES = {
    "data_prep": "1 · Data prep",
    "target_sampling": "2 · Target & sampling",
    "binning": "3 · Binning & WoE",
    "selection": "4 · Feature selection",
    "fit": "5 · Fit",
    "calibration": "6 · Calibration & rating",
    "validation": "7 · Validation",
}
ROLES = {r.value for r in ColumnRole} - {ColumnRole.PREDICTED.value, ColumnRole.UNASSIGNED.value}


def anchor_output(session, anchor: str) -> tuple[str, DataFramePacket]:
    """The anchor's (port, dataframe packet), running it first if needed."""
    block = session.graph.blocks[anchor]
    st = session.runner.state.get(anchor)
    if not (st and st.last_successful_key and session.runner.cache.get(st.last_successful_key)):
        session.runner.run_to_here(anchor)
        st = session.runner.state.get(anchor)
    entry = session.runner.cache.get(st.last_successful_key) if st and st.last_successful_key else None
    if entry is None:
        raise ValueError(f"{block.name!r} has no output to build from -- run it first")
    for port in block.outputs:
        value = entry.outputs.get(port.name)
        if isinstance(value, DataFramePacket):
            return port.name, value
    raise ValueError(f"{block.name!r} has no dataframe output")


def workspace_from(packet: DataFramePacket) -> Workspace:
    roles = {c: m.role.value for c, m in packet.schema_meta.items() if m.role.value in ROLES}
    return Workspace(packet.data, roles)


class Emitter:
    def __init__(self, session, build_id: str, backend_name: str) -> None:
        self.session = session
        self.build_id = build_id
        self.backend = backend_name
        self.at = datetime.now(UTC).isoformat()
        self.blocks: list[str] = []
        self.lanes: dict[str, str] = {}
        self.placer: Placer | None = None

    def lane(self, phase: str) -> str:
        if phase not in self.lanes:
            lane_id = f"dec_{self.build_id}_{phase}"
            self.session.set_lane(lane_id, PHASE_LANES.get(phase, phase), next_lane_order(self.session.graph))
            self.lanes[phase] = lane_id
            self.placer = Placer(self.session.graph)
        return self.lanes[phase]

    def add(self, phase: str, category: str, name: str, inputs: dict[str, tuple[str, str]], **params: Any) -> str:
        lane = self.lane(phase)
        x, y = self.placer.place(lane)
        block = self.session.add_block(category, name=name, lane=lane, x=x, y=y, params=params)
        block.provenance = {
            "source": "agent",
            "build_id": self.build_id,
            "at": self.at,
            "goal": "Decision-based PD build",
            "plan_llm": f"decisions ({self.backend})",
            "build_llm": f"decisions ({self.backend})",
            "plan_step": phase,
            "stage": phase,
            "modified_by_user": False,
        }
        for port, (src, src_port) in inputs.items():
            self.session.add_wire(src, src_port, block.id, port)
        self.blocks.append(block.id)
        return block.id


def emit(session, anchor: str, port: str, ws: Workspace, result: PlanResult, build_id: str, backend: str) -> dict:
    """Lay the plan out on the canvas. Returns {"blocks", "sinks", "report_block"}."""
    e = Emitter(session, build_id, backend)
    main = (anchor, port)
    samples: dict[str, tuple[str, str]] = {}
    woe: dict[str, tuple[str, str]] = {}
    sinks: list[str] = []
    feats: list[str] = []
    model = binning = None
    graded: dict[str, tuple[str, str]] = {}
    steps = {s["option"]: s for s in result.steps if "error" not in s}

    def side(phase, category, name, inputs, **params):
        bid = e.add(phase, category, name, inputs, **params)
        sinks.append(bid)
        return bid

    target = ws.col("target")  # the final one: forward_default_flag may have made it
    date_col = ws.col("date")
    for s in result.steps:
        opt, ph, p, r = s["option"], s["phase"], s.get("params", {}), s.get("result", {})
        if "error" in s or opt == "DONE":
            continue
        if opt == "profile":
            side(ph, "data_profile", "Data profile", {"df": main})
        elif opt == "screen":
            bid = e.add(ph, "select", "Column screen", {"df": main}, cols=_kept_columns(result, main, session, p))
            for col in p["excluded"]:
                if col in session.graph.blocks[bid].params["cols"]:
                    session.set_column_role(bid, col, "excluded", actor="agent")
            main = (bid, "out")
        elif opt == "dq_rules":
            side(ph, "data_quality_rules", "Data quality rules", {"df": main}, rules=p["rules"])
        elif opt in ("exclusions", "missing", "drop_immature") and s.get("block"):
            category = {
                "exclusions": "apply_exclusions",
                "missing": "missing_value_treatment",
                "drop_immature": "filter",
            }[opt]
            name = {"exclusions": "Exclusions", "missing": "Missing values", "drop_immature": "Drop immature periods"}[
                opt
            ]
            main = (e.add(ph, category, name, {"df": main}, **p), "out")
        elif opt == "forward_flag":
            bid = e.add(ph, "forward_default_flag", "Default flag (12m ahead)", {"df": main}, **p)
            session.set_column_role(bid, p["default_col"], "excluded", actor="agent")
            main = (bid, "out")
        elif opt == "trend":
            side(ph, "target_trend", "Default rate over time", {"df": main}, date_col=date_col, target_col=target, **p)
        elif opt == "time_split":
            bid = e.add(ph, "time_split", "Out-of-time split", {"df": main}, **p)
            main, samples["out_of_time"] = (bid, "development"), (bid, "out_of_time")
        elif opt == "train_test":
            bid = e.add(ph, "train_test_split", "Train / test split", {"df": main}, **p)
            main, samples["test"] = (bid, "train"), (bid, "test")
        elif opt == "binning":
            binning = e.add(ph, "fit_binning", "Binning", {"df": main}, features=_features(ws, steps), **p)
            sinks.append(binning)
        elif opt == "woe":
            feats = p["features"]
            for name, src in (("train", main), *samples.items()):
                woe[name] = (
                    e.add(
                        ph,
                        "apply_binning",
                        f"WoE · {name}",
                        {"df": src, "binning": (binning, "binning")},
                        output="woe",
                        features=feats,
                    ),
                    "out",
                )
            feats = [f"{f}_woe" for f in feats]
        elif opt == "correlation":
            side(ph, "correlation_matrix", "Correlation / VIF", {"df": woe["train"]}, features=feats)
            feats = [f for f in feats if f not in r["dropped"]]
        elif opt == "stability":
            against = p["against"]
            side(
                ph,
                "characteristic_stability",
                f"Feature stability vs {against}",
                {"expected": woe["train"], "actual": woe[against]},
                features=feats,
            )
            feats = [f for f in feats if f not in r["dropped"]]
        elif opt == "stepwise":
            side(
                ph, "stepwise_selection", "Stepwise selection", {"df": woe["train"]}, target=target, features=feats, **p
            )
        elif opt == "fit":
            model = e.add(
                ph,
                "logistic_regression",
                "Logistic PD model",
                {"df": woe["train"]},
                target=target,
                features=p["features"],
            )
        elif opt == "calibrate":
            model = e.add(
                ph,
                "calibrate_model",
                "Calibrate to long-run DR",
                {"df": (model, "predictions"), "model": (model, "model")},
                central_tendency=p["central_tendency"],
            )
        elif opt == "master_scale":
            pred = e.add(ph, "predict", "Score · train", {"df": woe["train"], "model": (model, "model")})
            master = e.add(
                ph, "fit_master_scale", "Master scale", {"df": (pred, "predictions")}, score_col="predicted_proba",
                target_col=target, **p,
            )  # fmt: skip
            for name in ("train", *samples):
                src = (pred, "predictions") if name == "train" else None
                if src is None:
                    src = (
                        e.add(ph, "predict", f"Score · {name}", {"df": woe[name], "model": (model, "model")}),
                        "predictions",
                    )
                graded[name] = (
                    e.add(
                        ph,
                        "assign_rating_grade",
                        f"Grades · {name}",
                        {"df": src, "master_scale": (master, "master_scale")},
                    ),
                    "out",
                )
        elif opt == "discrimination":
            for name, src in graded.items():
                side(ph, "auc_gini", f"Gini · {name}", {"df": src}, score_col="predicted_proba", target_col=target)
        elif opt == "calibration_test":
            for name in samples:
                side(ph, "calibration_test", f"Hosmer-Lemeshow · {name}", {"df": graded[name]},
                     score_col="predicted_proba", target_col=target)  # fmt: skip
        elif opt == "backtest":
            side(ph, "grade_backtest", f"Grade back-test · {p['sample']}", {"df": graded[p["sample"]]},
                 grade_col="grade", target_col=target, pd_col="grade_pd")  # fmt: skip
        elif opt == "score_stability":
            side(ph, "psi_test", "Score PSI · out-of-time", {"expected": graded["train"], "actual": graded["out_of_time"]},
                 col="predicted_proba")  # fmt: skip
    report_block = sinks[-1] if sinks else (e.blocks[-1] if e.blocks else anchor)
    return {"blocks": e.blocks, "sinks": sinks, "report_block": report_block, "lanes": list(e.lanes.values())}


def _features(ws: Workspace, steps: dict[str, dict]) -> list[str]:
    binning = steps.get("iv_screen")
    kept = (binning or {}).get("params", {}).get("kept", [])
    dropped = list((binning or {}).get("params", {}).get("dropped", {}))
    return kept + dropped  # every candidate the binning saw


def _kept_columns(result: PlanResult, main: tuple[str, str], session, params: dict) -> list[str]:
    """The screen keeps every column but the excluded ones -- except a
    panel's status column, which the target is built from later."""
    keep_anyway = {s["params"]["default_col"] for s in result.steps if s["option"] == "forward_flag" and "params" in s}
    return [c for c in session_columns(session, main) if c not in params["excluded"] or c in keep_anyway]


def session_columns(session, ref: tuple[str, str]) -> list[str]:
    st = session.runner.state.get(ref[0])
    entry = session.runner.cache.get(st.last_successful_key) if st and st.last_successful_key else None
    value = entry.outputs.get(ref[1]) if entry else None
    return list(value.data.columns) if isinstance(value, DataFramePacket) else []


def report(result: PlanResult, decider: Decider, backend: str, build_id: str) -> str:
    """The build report: the sign-off first, then each phase's steps and
    every decision, with what the rule policy would have said."""
    recs = decider.records
    sign = next((s for s in result.steps if s["option"] == "sign_off"), None)
    lines = [f"# Decision-based PD build {build_id}", ""]
    lines.append(f"Decisions by **{backend}**: {len(recs)} in all, {sum(r.agrees for r in recs)} agree with the rule policy, "
                 f"{sum(r.needs_user for r in recs)} below the confidence threshold (for a person to confirm).")  # fmt: skip
    if sign:
        res = sign["result"]
        lines += ["", f"## Sign-off: **{res['verdict'].replace('_', ' ')}**", ""]
        lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in res.items() if k != "verdict"]
    by_phase: dict[str, list[dict]] = {}
    for s in result.steps:
        by_phase.setdefault(s["phase"], []).append(s)
    for phase, steps in by_phase.items():
        lines += ["", f"## {PHASE_LANES.get(phase, phase)}", ""]
        for s in steps:
            what = s.get("block") or s.get("action") or s["option"]
            detail = s.get("error") or s.get("result") or s.get("skipped") or ""
            lines.append(f"- **{s['option']}** ({what}) {_short(detail)}")
    flagged = [r for r in recs if r.needs_user or not r.agrees]
    if flagged:
        lines += [
            "",
            "## Decisions to review",
            "",
            "| decision | answer | confidence | rule policy |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r.name} · {r.qid} | {r.answer.value} | {r.answer.confidence:.2f} | {r.rule_answer} |" for r in flagged
        ]
    return "\n".join(lines)


def _short(value: Any, limit: int = 300) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def new_build_id() -> str:
    return "dec" + secrets.token_hex(3)
