"""Phases 3-5 of a PD build: binning & WoE, feature selection, and the fit
with its drop-and-refit loop. Same shape as phases 1-2: a menu per phase,
typed questions per block, each with the rule the decision-hint policy
would apply.

Samples: the workspace's df is the development (training) sample; the
test and out-of-time samples sit in ws.artifacts and get the same WoE
transform, so validation reads them like-for-like."""

from __future__ import annotations

from typing import Any

from .. import hints as h
from .core import Ask, Decider, choice, noul
from .phases import Option, Phase, _done, run_block
from .workspace import Workspace

SAMPLES = ("test", "out_of_time")


def _target(ws: Workspace) -> str:
    return ws.col("target")


def _events(ws: Workspace) -> int:
    return int(ws.df[_target(ws)].sum() or 0)


# ---- phase 3: binning & WoE ---------------------------------------------------------------

MAX_BINS_Q = choice(
    "Coarse classing: at most how many bins per feature? Each bin needs enough defaults to give a stable WoE "
    "(roughly 30+): fewer defaults, fewer bins.",
    {
        "4": "4 bins: under 200 defaults",
        "5": "5 bins: 200-500 defaults",
        "6": "6 bins: 500+ defaults (usual)",
        "8": "8 bins: very large samples",
    },
)
MONOTONIC_Q = noul("Force numeric features' bins to be monotonic in default rate (the scorecard convention)?")


def max_bins_rule(f: dict[str, Any]) -> str:
    return "4" if f["events"] < 200 else "5" if f["events"] < 500 else "6"


def run_binning(ws: Workspace, d: Decider) -> dict[str, Any]:
    f = {"rows": ws.df.height, "events": _events(ws), "features": len(ws.cols("feature"))}
    a = d.ask(
        "binning",
        f,
        {"max_bins": MAX_BINS_Q, "monotonic": MONOTONIC_Q},
        {"max_bins": max_bins_rule, "monotonic": lambda f: True},
    )
    params = {
        "target": _target(ws),
        "features": ws.cols("feature"),
        "max_bins": int(a["max_bins"]),
        "monotonic": bool(a["monotonic"]),
    }
    ws.artifacts["binning"], _, ws.artifacts["binning_summary"] = run_block("fit_binning", ws.df, **params)
    return {"block": "fit_binning", "params": {k: v for k, v in params.items() if k != "features"}}


IV_Q = choice(
    "PD feature screen on the binned feature's information value (IV). What should be done with it?",
    {
        "keep": "useful signal: IV 0.02 to 0.5",
        "drop": "no signal: IV below 0.02",
        "check_leakage": "IV 0.5 or more: suspiciously strong, may leak the target",
    },
)
KNOWN_AT_OBSERVATION_Q = noul(
    "This feature is suspiciously predictive. Judging by what it is, would its value be known at the observation "
    "(application / scoring) date, before the outcome -- so it's a genuine predictor, not a leak?"
)


def iv_rule(f: dict[str, Any]) -> str:
    iv = f.get("iv") or 0.0
    return "drop" if iv < h.USELESS_IV else "check_leakage" if iv >= h.SUSPICIOUS_IV else "keep"


def run_iv_screen(ws: Workspace, d: Decider) -> dict[str, Any]:
    rows = ws.artifacts["binning_summary"].to_dicts()
    keys = ("feature", "type", "iv", "gini", "monotonic", "n_bins")
    asks = [Ask(f"iv:{r['feature']}", {k: r[k] for k in keys}, {"verdict": IV_Q}, {"verdict": iv_rule}) for r in rows]
    verdicts = {r["feature"]: a["verdict"] for r, a in zip(rows, d.ask_many(asks))}
    suspects = [r for r in rows if verdicts[r["feature"]] == "check_leakage"]
    # Leakage is a judgement on what the feature *is*: the screen already
    # removed post-outcome columns, so the policy keeps the rest.
    asks = [
        Ask(
            f"leak:{r['feature']}",
            {"feature": r["feature"], "iv": r["iv"], "gini": r["gini"]},
            {"known": KNOWN_AT_OBSERVATION_Q},
            {"known": lambda f: True},
        )
        for r in suspects
    ]
    for r, a in zip(suspects, d.ask_many(asks)):
        verdicts[r["feature"]] = "keep" if a["known"] else "drop_leak"
    kept = [f for f, v in verdicts.items() if v == "keep"]
    if not kept:
        raise ValueError("no feature passed the IV screen")
    ws.artifacts["model_features"] = kept
    dropped = {f: v for f, v in verdicts.items() if v != "keep"}
    return {"block": None, "action": "IV screen", "params": {"kept": kept, "dropped": dropped}}


def run_woe(ws: Workspace, d: Decider) -> dict[str, Any]:
    feats = ws.artifacts["model_features"]
    binning = ws.artifacts["binning"]
    ws.df = run_block("apply_binning", ws.df, binning, output="woe", features=feats)
    for s in SAMPLES:
        if s in ws.artifacts:
            ws.artifacts[s] = run_block("apply_binning", ws.artifacts[s], binning, output="woe", features=feats)
    ws.artifacts["woe_features"] = [f"{f}_woe" for f in feats]
    return {"block": "apply_binning", "params": {"output": "woe", "features": feats, "samples": ["train", *SAMPLES]}}


def phase3_facts(ws: Workspace) -> dict[str, Any]:
    return {
        "events": _events(ws) if _target(ws) else None,
        "features": len(ws.cols("feature")),
        "binned": _done(ws, "binning"),
        "kept_after_iv": len(ws.artifacts.get("model_features") or []) or None,
    }


PHASE3 = Phase(
    "binning",
    "Binning & WoE",
    "features are binned, screened on IV (leaks dealt with), and WoE-transformed on every sample",
    [
        Option(
            "binning",
            "fit_binning: coarse-class every candidate feature and measure its IV",
            lambda ws: bool(_target(ws)) and not _done(ws, "binning"),
            run_binning,
        ),
        Option(
            "iv_screen",
            "IV screen: keep, drop or question each feature on its information value (needs the binning)",
            lambda ws: _done(ws, "binning") and not _done(ws, "iv_screen"),
            run_iv_screen,
        ),
        Option(
            "woe",
            "apply_binning: WoE-transform the kept features on train, test and out-of-time (needs the IV screen)",
            lambda ws: _done(ws, "iv_screen") and not _done(ws, "woe"),
            run_woe,
        ),
    ],
    phase3_facts,
)


# ---- phase 4: feature selection --------------------------------------------------------------


def _iv(ws: Workspace) -> dict[str, float]:
    return {f"{r['feature']}_woe": r["iv"] for r in ws.artifacts["binning_summary"].to_dicts()}


def _woe(ws: Workspace) -> list[str]:
    return ws.artifacts.get("woe_features") or []


def _drop(ws: Workspace, features: list[str]) -> None:
    ws.artifacts["woe_features"] = [f for f in _woe(ws) if f not in features]


def pair_question(a: str, b: str, iv: dict[str, float]) -> dict[str, Any]:
    return choice(
        "These two WoE features are highly correlated (|r| 0.7+): largely the same information. "
        "Usual practice: keep the one with the higher IV.",
        {
            "drop_first": f"drop {a} (IV {iv[a]:.3f})",
            "drop_second": f"drop {b} (IV {iv[b]:.3f})",
            "keep_both": "keep both",
        },
    )


VIF_Q = noul("Drop this feature as redundant? Usual practice: yes when its VIF is 10 or more.")


def run_correlation(ws: Workspace, d: Decider) -> dict[str, Any]:
    feats = _woe(ws)
    metric = run_block("correlation_matrix", ws.df, features=feats)
    names, m = metric["features"], metric["correlation"]
    iv = _iv(ws)
    pairs = sorted(
        (
            (abs(m[i][j]), names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if h.num(m[i][j]) and abs(m[i][j]) >= h.HIGH_CORRELATION
        ),
        reverse=True,
    )
    asks = [
        Ask(
            f"corr:{a}/{b}",
            {"r": r, "iv_first": iv[a], "iv_second": iv[b]},
            {"pick": pair_question(a, b, iv)},
            {"pick": lambda f: "drop_first" if f["iv_first"] < f["iv_second"] else "drop_second"},
        )
        for r, a, b in pairs
    ]
    dropped: dict[str, str] = {}
    for (_, a, b), ans in zip(pairs, d.ask_many(asks)):
        if a in dropped or b in dropped:
            continue  # the pair's already broken up
        if ans["pick"] != "keep_both":
            gone, kept = (a, b) if ans["pick"] == "drop_first" else (b, a)
            dropped[gone] = f"correlated with {kept}"
    high_vif = [v for v in metric.get("vif") or [] if h.num(v.get("vif")) and v["vif"] >= h.HIGH_VIF]
    high_vif = [v for v in high_vif if v["feature"] not in dropped]
    asks = [
        Ask(
            f"vif:{v['feature']}",
            {"feature": v["feature"], "vif": v["vif"], "iv": iv[v["feature"]]},
            {"drop": VIF_Q},
            {"drop": lambda f: f["vif"] >= h.HIGH_VIF},
        )
        for v in high_vif
    ]
    for v, ans in zip(high_vif, d.ask_many(asks)):
        if ans["drop"]:
            dropped[v["feature"]] = f"VIF {v['vif']:.1f}"
    _drop(ws, list(dropped))
    return {"block": "correlation_matrix", "params": {}, "result": {"dropped": dropped}}


STABILITY_Q = noul(
    "This feature's distribution shifted between development and the validation sample (PSI 0.1-0.25: shifted, "
    "0.25+: major shift). Drop it from the model? Usual practice: drop on a major shift."
)


def _validation(ws: Workspace) -> str | None:
    return "out_of_time" if "out_of_time" in ws.artifacts else "test" if "test" in ws.artifacts else None


def run_stability(ws: Workspace, d: Decider) -> dict[str, Any]:
    sample = _validation(ws)
    table = run_block("characteristic_stability", ws.df, ws.artifacts[sample], features=_woe(ws))
    shifted = [r for r in table.to_dicts() if h.num(r["psi"]) and r["psi"] >= h.PSI_STABLE]
    iv = _iv(ws)
    asks = [
        Ask(
            f"psi:{r['feature']}",
            {"feature": r["feature"], "psi": r["psi"], "iv": iv[r["feature"]], "against": sample},
            {"drop": STABILITY_Q},
            {"drop": lambda f: f["psi"] >= h.PSI_SHIFTED},
        )
        for r in shifted
    ]
    dropped = {r["feature"]: f"PSI {r['psi']:.2f}" for r, a in zip(shifted, d.ask_many(asks)) if a["drop"]}
    if len(dropped) == len(_woe(ws)):
        dropped = {}  # never drop every feature on stability alone
    _drop(ws, list(dropped))
    return {"block": "characteristic_stability", "params": {"against": sample}, "result": {"dropped": dropped}}


DIRECTION_Q = choice(
    "Stepwise selection direction:",
    {
        "both": "forward, re-checking earlier picks after each addition (usual)",
        "forward": "add the best feature one at a time",
        "backward": "start from all, remove the weakest one at a time",
    },
)
P_ENTER_Q = choice(
    "Significance a feature needs to enter the model:",
    {"0.05": "p < 0.05: the usual level", "0.01": "p < 0.01: stricter, for large samples (20,000+ rows)"},
)


def run_stepwise(ws: Workspace, d: Decider) -> dict[str, Any]:
    f = {"rows": ws.df.height, "events": _events(ws), "candidates": len(_woe(ws))}
    a = d.ask(
        "stepwise",
        f,
        {"direction": DIRECTION_Q, "p_enter": P_ENTER_Q},
        {"direction": lambda f: "both", "p_enter": lambda f: "0.01" if f["rows"] >= 20000 else "0.05"},
    )
    params = {"target": _target(ws), "features": _woe(ws), "direction": a["direction"], "p_enter": float(a["p_enter"])}
    metric = run_block("stepwise_selection", ws.df, **params)
    if not metric["selected"]:
        raise ValueError("stepwise selection kept no feature")
    ws.artifacts["woe_features"] = metric["selected"]
    return {
        "block": "stepwise_selection",
        "params": {k: v for k, v in params.items() if k not in ("target", "features")},
        "result": {"selected": metric["selected"]},
    }


def phase4_facts(ws: Workspace) -> dict[str, Any]:
    return {
        "woe_features": len(_woe(ws)),
        "validation_sample": _validation(ws),
        "stepwise_done": _done(ws, "stepwise"),
    }


PHASE4 = Phase(
    "selection",
    "Feature selection",
    "redundant features are removed and stepwise selection has picked the model's features",
    [
        Option(
            "correlation",
            "correlation_matrix: find highly correlated pairs and high VIFs, drop the weaker",
            lambda ws: bool(_woe(ws)) and not _done(ws, "correlation") and not _done(ws, "stepwise"),
            run_correlation,
        ),
        Option(
            "stability",
            "characteristic_stability: drop features whose distribution shifts in the validation sample",
            lambda ws: (
                bool(_woe(ws))
                and _validation(ws) is not None
                and not _done(ws, "stability")
                and not _done(ws, "stepwise")
            ),
            run_stability,
            required=False,
        ),
        Option(
            "stepwise",
            "stepwise_selection: pick the model's features by significance (after the redundancy check)",
            lambda ws: _done(ws, "correlation") and not _done(ws, "stepwise"),
            run_stepwise,
        ),
    ],
    phase4_facts,
)


# ---- phase 5: fit ------------------------------------------------------------------------------

COEF_Q = choice(
    "Logistic PD model on WoE features. WoE = ln(goods / bads) is higher where defaults are rarer, so every "
    "coefficient should be NEGATIVE. What should be done with this coefficient?",
    {
        "keep": "negative (expected sign) and significant (p < 0.05)",
        "drop_insignificant": "p-value 0.05 or higher",
        "drop_wrong_sign": "positive: the wrong sign, usually collinearity",
    },
)
MAX_REFITS = 5


def coef_rule(f: dict[str, Any]) -> str:
    if f["coef"] > 0:
        return "drop_wrong_sign"
    return "drop_insignificant" if f["p_value"] >= h.MAX_P_VALUE else "keep"


def run_fit(ws: Workspace, d: Decider) -> dict[str, Any]:
    feats = list(_woe(ws))
    rounds = []
    for _ in range(MAX_REFITS):
        _, model = run_block("logistic_regression", ws.df, target=_target(ws), features=feats)
        stats = model["statistics"]
        asks = [
            Ask(
                f"coef:{f}",
                {"feature": f, "coef": stats[f]["estimate"], "p_value": stats[f]["p_value"]},
                {"action": COEF_Q},
                {"action": coef_rule},
            )
            for f in feats
        ]
        drop = {f: a["action"] for f, a in zip(feats, d.ask_many(asks)) if a["action"] != "keep"}
        rounds.append({"features": len(feats), "dropped": drop})
        if not drop or len(drop) == len(feats):
            break
        feats = [f for f in feats if f not in drop]
    ws.artifacts["model"] = model
    ws.artifacts["woe_features"] = feats
    return {
        "block": "logistic_regression",
        "params": {"features": feats},
        "result": {"refits": len(rounds) - 1, "rounds": rounds},
    }


DISCRIMINATION_Q = choice(
    "PD model validation: judge its discrimination from the Gini on each sample.",
    {
        "good": "validation Gini 0.2+, below 0.85, and within 0.05 of train",
        "weak": "validation Gini below 0.2",
        "overfit": "validation Gini more than 0.05 below train",
        "suspicious": "Gini 0.85+: likely leakage",
    },
)


def discrimination_rule(f: dict[str, Any]) -> str:
    ginis = [v for k, v in f.items() if k.startswith("gini_")]
    val = [v for k, v in f.items() if k.startswith("gini_") and k != "gini_train"]
    if max(ginis) >= h.SUSPICIOUS_GINI:
        return "suspicious"
    if min(val) < h.WEAK_GINI:
        return "weak"
    if f["gini_train"] - min(val) > h.OVERFIT_GINI_DROP:
        return "overfit"
    return "good"


def run_evaluate(ws: Workspace, d: Decider) -> dict[str, Any]:
    model, target = ws.artifacts["model"], _target(ws)
    ginis = {}
    for name in ("train", *SAMPLES):
        df = ws.df if name == "train" else ws.artifacts.get(name)
        if df is None:
            continue
        preds = run_block("predict", df, model)
        ginis[f"gini_{name}"] = run_block("auc_gini", preds, score_col="predicted_proba", target_col=target)["gini"]
    verdict = d.ask("discrimination", ginis, {"verdict": DISCRIMINATION_Q}, {"verdict": discrimination_rule})
    ws.artifacts["verdict"] = verdict["verdict"]
    return {"block": "auc_gini", "params": {"samples": list(ginis)}, "result": {**ginis, **verdict}}


PHASE5 = Phase(
    "fit",
    "Model fit",
    "the logistic model is fitted with every coefficient sound",
    [
        Option(
            "fit",
            "logistic_regression: fit, then drop wrong-sign / insignificant coefficients and refit",
            lambda ws: _done(ws, "stepwise") and not _done(ws, "fit"),
            run_fit,
        ),
    ],
    lambda ws: {"model_features": len(_woe(ws)), "fitted": _done(ws, "fit")},
)
