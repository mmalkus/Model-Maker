"""Phases 6-7 of a PD build: calibration & rating scale, then validation
ending in a sign-off -- the overall verdict the person sees first.

Every sample is scored with the (calibrated) model and graded on the
master scale the same way (scored()), so the tests read like-for-like."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import polars as pl

from .. import hints as h
from .core import Decider, choice, noul, score
from .modelling_phases import DISCRIMINATION_Q, SAMPLES, _target, discrimination_rule
from .phases import Option, Phase, _done, run_block
from .workspace import Workspace


def scored(ws: Workspace, name: str) -> pl.DataFrame | None:
    """Sample `name` ("train", "test", "out_of_time") with predicted_proba,
    and grade / grade_pd once the master scale exists."""
    df = ws.df if name == "train" else ws.artifacts.get(name)
    if df is None:
        return None
    out = run_block("predict", df, ws.artifacts["model"])
    if "master_scale" in ws.artifacts:
        out = run_block("assign_rating_grade", out, ws.artifacts["master_scale"])
    return out


def _verdicts(ws: Workspace) -> dict[str, str]:
    return ws.artifacts.setdefault("verdicts", {})


def _validation_samples(ws: Workspace) -> list[str]:
    return [s for s in SAMPLES if s in ws.artifacts]


# ---- phase 6: calibration & rating scale -----------------------------------------------------

CENTRAL_Q = choice(
    "PD calibration: which long-run average default rate is the central tendency?",
    {
        "time_weighted": "the average of the yearly default rates, each year counting the same (usual)",
        "default_weighted": "the pooled rate over all years, busier years counting more",
        "higher": "the higher of the two: conservative, when they differ by 10% or more",
    },
)


def central_rule(f: dict[str, Any]) -> str:
    return "higher" if f["relative_gap"] > h.LRA_GAP else "time_weighted"


def run_calibrate(ws: Workspace, d: Decider) -> dict[str, Any]:
    target, date_col = _target(ws), ws.col("date")
    frames = [ws.df, *(ws.artifacts[s] for s in _validation_samples(ws))]
    cols = [c for c in (target, date_col) if c]
    pooled = pl.concat([f.select(cols) for f in frames], how="vertical_relaxed")
    if date_col:
        _, lra = run_block("long_run_average", pooled, target_col=target, date_col=date_col, period="year")
        dw, tw = lra["default_weighted"], lra["time_weighted"]
        f = {
            "n_periods": lra["n_periods"],
            "time_weighted": tw,
            "default_weighted": dw,
            "relative_gap": abs(tw - dw) / dw if dw else 0.0,
        }
        pick = d.ask("central_tendency", f, {"basis": CENTRAL_Q}, {"basis": central_rule})["basis"]
        ct = max(dw, tw) if pick == "higher" else tw if pick == "time_weighted" else dw
        basis = {"basis": pick, "n_periods": lra["n_periods"]}
    else:
        ct = float(pooled[target].mean())
        basis = {"basis": "pooled sample rate (no date: no long-run average)"}
    preds = run_block("predict", ws.df, ws.artifacts["model"])
    ws.artifacts["model"] = run_block("calibrate_model", preds, ws.artifacts["model"], central_tendency=ct)
    shift = ws.artifacts["model"]["calibration"]["intercept_shift"]
    return {
        "block": "calibrate_model",
        "params": {"central_tendency": ct, **basis},
        "result": {"intercept_shift": shift},
    }


GRADES_Q = choice(
    "Master scale: how many rating grades? Each grade needs enough defaults to back-test (roughly 40+ in "
    "development): fewer defaults, fewer grades.",
    {"5": "5 grades: under 280 defaults", "7": "7 grades: 280-400 defaults", "10": "10 grades: 400+ defaults"},
)
ALGORITHM_Q = choice(
    "How should the grade boundaries be set?",
    {
        "monotonic_default_rate": "so observed default rates rise grade by grade (usual for a rating scale)",
        "quantile": "equal-sized grades by score",
    },
)
MIN_SHARE_Q = noul("Require every grade to hold at least 3% of the population (no thin grades)?")


def grades_rule(f: dict[str, Any]) -> str:
    return "10" if f["events"] >= 400 else "7" if f["events"] >= 280 else "5"


def run_master_scale(ws: Workspace, d: Decider) -> dict[str, Any]:
    target = _target(ws)
    f = {"rows": ws.df.height, "events": int(ws.df[target].sum() or 0)}
    a = d.ask(
        "master_scale",
        f,
        {"grades": GRADES_Q, "algorithm": ALGORITHM_Q, "min_share": MIN_SHARE_Q},
        {"grades": grades_rule, "algorithm": lambda f: "monotonic_default_rate", "min_share": lambda f: True},
    )
    params = {
        "n_grades": int(a["grades"]),
        "algorithm": a["algorithm"],
        "min_grade_share": h.THIN_GRADE if a["min_share"] else 0.0,
    }
    preds = run_block("predict", ws.df, ws.artifacts["model"])
    ms = run_block("fit_master_scale", preds, score_col="predicted_proba", target_col=target, **params)
    ws.artifacts["master_scale"] = ms
    rates = [g["observed_rate"] for g in ms["grades"]]
    monotonic = all(b >= a for a, b in pairwise(rates))
    return {
        "block": "fit_master_scale",
        "params": params,
        "result": {"grades": len(ms["grades"]), "monotonic": monotonic},
    }


PHASE6 = Phase(
    "calibration",
    "Calibration & rating scale",
    "the model is calibrated to the long-run default rate and a master scale grades it",
    [
        Option(
            "calibrate",
            "calibrate_model: shift the PDs to the long-run average default rate",
            lambda ws: "model" in ws.artifacts and not _done(ws, "calibrate"),
            run_calibrate,
        ),
        Option(
            "master_scale",
            "fit_master_scale: group the calibrated PDs into rating grades (after calibration)",
            lambda ws: _done(ws, "calibrate") and not _done(ws, "master_scale"),
            run_master_scale,
        ),
    ],
    lambda ws: {"calibrated": _done(ws, "calibrate"), "graded": _done(ws, "master_scale")},
)


# ---- phase 7: validation ------------------------------------------------------------------------


def run_discrimination(ws: Workspace, d: Decider) -> dict[str, Any]:
    target = _target(ws)
    f = {}
    for name in ("train", *_validation_samples(ws)):
        f[f"gini_{name}"] = run_block("auc_gini", scored(ws, name), score_col="predicted_proba", target_col=target)[
            "gini"
        ]
    verdict = d.ask("discrimination", f, {"verdict": DISCRIMINATION_Q}, {"verdict": discrimination_rule})["verdict"]
    _verdicts(ws)["discrimination"] = verdict
    return {"block": "auc_gini", "params": {"samples": list(f)}, "result": {**f, "verdict": verdict}}


CALIBRATED_Q = noul(
    "Does the PD calibration hold on the validation samples? Yes when every Hosmer-Lemeshow p-value is 0.05+ and "
    "the mean PD is within 20% (relative) of the observed default rate."
)


def calibration_rule(f: dict[str, Any]) -> bool:
    return all(v >= h.CALIBRATION_P for k, v in f.items() if k.startswith("hl_p_")) and all(
        v <= h.CALIBRATION_GAP for k, v in f.items() if k.startswith("mean_gap_")
    )


def run_calibration_test(ws: Workspace, d: Decider) -> dict[str, Any]:
    target = _target(ws)
    f = {}
    for name in _validation_samples(ws):
        df = scored(ws, name)
        f[f"hl_p_{name}"] = run_block("calibration_test", df, score_col="predicted_proba", target_col=target)["p_value"]
        observed = float(df[target].mean() or 0)
        f[f"mean_gap_{name}"] = abs(float(df["predicted_proba"].mean()) - observed) / observed if observed else 0.0
    ok = d.ask("calibration", f, {"holds": CALIBRATED_Q}, {"holds": calibration_rule})["holds"]
    _verdicts(ws)["calibration"] = "holds" if ok else "miscalibrated"
    return {"block": "calibration_test", "params": {}, "result": {**f, "holds": ok}}


BACKTEST_Q = choice(
    "Grade PD back-test on the validation sample (binomial test per grade; red = PD too low for the observed "
    "default rate). Verdict:",
    {
        "pass": "no red or amber grades, portfolio green, default rates rise grade by grade",
        "findings": "an amber or a single red grade, or a non-monotonic step: usable, to be noted",
        "fail": "portfolio red, or two or more red grades",
    },
)


def backtest_rule(f: dict[str, Any]) -> str:
    if f["portfolio"] == "red" or f["red_grades"] >= 2:
        return "fail"
    if f["red_grades"] or f["amber_grades"] or not f["monotonic"]:
        return "findings"
    return "pass"


def run_backtest(ws: Workspace, d: Decider) -> dict[str, Any]:
    name = "out_of_time" if "out_of_time" in ws.artifacts else "test"
    bt = run_block("grade_backtest", scored(ws, name), grade_col="grade", target_col=_target(ws), pd_col="grade_pd")
    lights = [g["traffic_light"] for g in bt["grades"]]
    f = {
        "sample": name,
        "grades": len(lights),
        "red_grades": lights.count("red"),
        "amber_grades": sum(1 for x in lights if x not in ("green", "red")),
        "portfolio": bt["portfolio"]["traffic_light"],
        "monotonic": bool(bt.get("monotonic")),
    }
    verdict = d.ask("backtest", f, {"verdict": BACKTEST_Q}, {"verdict": backtest_rule})["verdict"]
    _verdicts(ws)["backtest"] = verdict
    return {"block": "grade_backtest", "params": {"sample": name}, "result": {**f, "verdict": verdict}}


SCORE_STABILITY_Q = score(
    "How far has the score distribution shifted from development to out-of-time, by PSI (below 0.1 stable, "
    "0.1-0.25 shifted, above 0.25 major shift)?",
    ["stable", "shifted", "major_shift"],
)


def stability_rule(f: dict[str, Any]) -> str:
    return "stable" if f["psi"] < h.PSI_STABLE else "shifted" if f["psi"] < h.PSI_SHIFTED else "major_shift"


def run_score_stability(ws: Workspace, d: Decider) -> dict[str, Any]:
    psi = run_block("psi_test", scored(ws, "train"), scored(ws, "out_of_time"), col="predicted_proba")["psi"]
    level = d.ask("score_psi", {"psi": psi}, {"level": SCORE_STABILITY_Q}, {"level": stability_rule})["level"]
    _verdicts(ws)["score_stability"] = level
    return {"block": "psi_test", "params": {"col": "predicted_proba"}, "result": {"psi": psi, "level": level}}


SIGN_OFF_Q = choice(
    "Overall verdict on this PD model, from the validation results:",
    {
        "accept": "every test passed",
        "accept_with_findings": "usable, with findings to note: a small overfit, miscalibration, amber grades, or a "
        "shifted score",
        "reject": "weak or suspicious discrimination, or a failed back-test",
    },
)


def sign_off_rule(f: dict[str, Any]) -> str:
    if f.get("discrimination") in ("weak", "suspicious") or f.get("backtest") == "fail":
        return "reject"
    clean = {"discrimination": "good", "calibration": "holds", "backtest": "pass", "score_stability": "stable"}
    return "accept" if all(f.get(k, ok) == ok for k, ok in clean.items()) else "accept_with_findings"


def run_sign_off(ws: Workspace, d: Decider) -> dict[str, Any]:
    f = dict(_verdicts(ws))
    verdict = d.ask("sign_off", f, {"verdict": SIGN_OFF_Q}, {"verdict": sign_off_rule})["verdict"]
    ws.artifacts["sign_off"] = verdict
    return {"block": None, "action": "sign-off", "params": {}, "result": {**f, "verdict": verdict}}


def _validated(ws: Workspace) -> bool:
    return all(_done(ws, k) for k in ("discrimination", "calibration_test", "backtest"))


def _before_sign_off(ws: Workspace, key: str) -> bool:
    return "master_scale" in ws.artifacts and not _done(ws, key) and not _done(ws, "sign_off")


PHASE7 = Phase(
    "validation",
    "Validation",
    "discrimination, calibration and the grade back-test are judged, and the model is signed off",
    [
        Option(
            "discrimination",
            "auc_gini: Gini on train, test and out-of-time",
            lambda ws: _before_sign_off(ws, "discrimination"),
            run_discrimination,
        ),
        Option(
            "calibration_test",
            "calibration_test: Hosmer-Lemeshow and mean PD vs observed on each validation sample",
            lambda ws: _before_sign_off(ws, "calibration_test"),
            run_calibration_test,
        ),
        Option(
            "backtest",
            "grade_backtest: binomial test of each grade's PD on the validation sample",
            lambda ws: _before_sign_off(ws, "backtest"),
            run_backtest,
        ),
        Option(
            "score_stability",
            "psi_test: score distribution shift, development vs out-of-time",
            lambda ws: _before_sign_off(ws, "score_stability") and "out_of_time" in ws.artifacts,
            run_score_stability,
            required=False,
        ),
        Option(
            "sign_off",
            "sign-off: the overall verdict, once discrimination, calibration and the back-test are judged",
            lambda ws: _validated(ws) and not _done(ws, "sign_off"),
            run_sign_off,
        ),
    ],
    lambda ws: {"tests_judged": sorted((ws.artifacts.get("verdicts") or {}).keys())},
)
