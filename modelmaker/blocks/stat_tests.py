"""Model-validation test blocks: KS statistic, AUC/Gini, and Population
Stability Index (PSI) -- the standard discrimination and stability checks
for a PD-style model. Each produces a scalar_metric artifact (a plain
dict), not a dataframe, so it terminates a branch of the pipeline the way
generate_image/display_table do. Extra libraries (scipy, sklearn, numpy)
are imported locally inside each fn -- see generate_image in library.py for
why: the function body is inlined verbatim into the compiled script, which
only guarantees `polars as pl` at module level.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..agent import hints as h
from ..metadata_transforms import infer_dtypes
from .base import BlockSpec, FieldSpec, PortSpec, register_block


def ks_test(df: pl.DataFrame, score_col: str, target_col: str) -> dict:
    """Kolmogorov-Smirnov discrimination test: two-sample KS between the
    `score_col` distributions of goods (`target_col` == 0) and bads
    (`target_col` == 1). Emits a scalar_metric dict on the `metric` port:
    {"kind": "ks_test", "ks_statistic", "p_value"} -- higher KS means
    better separation. `target_col` must be binary 0/1 with both values
    present. `score_col` auto-fills from the input's role=predicted column
    and `target_col` from its role=target column when left unset."""
    from scipy.stats import ks_2samp

    scores = df[score_col].to_numpy()
    target = df[target_col].to_numpy()
    good = scores[target == 0]
    bad = scores[target == 1]
    if len(good) == 0 or len(bad) == 0:
        raise ValueError(f"'{target_col}' must contain both 0 and 1 values to run a KS test")
    result = ks_2samp(good, bad)
    return {"kind": "ks_test", "ks_statistic": float(result.statistic), "p_value": float(result.pvalue)}


def _ks_test_hints(outputs: dict[str, Any]) -> list[str]:
    ks = h.metric(outputs).get("ks_statistic")
    if not h.num(ks):
        return []
    if ks >= h.SUSPICIOUS_KS:
        return [f"KS {ks:.2f} is unusually high for a credit model -- check the features for leakage."]
    if ks < h.WEAK_KS:
        return [f"KS {ks:.2f} is weak -- the score separates goods and bads poorly."]
    return [f"KS {ks:.2f} is in the usual range."]


register_block(
    BlockSpec(
        category="ks_test",
        block_type="output",
        group="tests",
        display_name="KS test",
        tags=("performance",),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=ks_test,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("score_col", "Score column", "column", auto_role="predicted"),
            FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
        ),
        hints=_ks_test_hints,
    )
)


def auc_gini(df: pl.DataFrame, score_col: str, target_col: str) -> dict:
    """Area under the ROC curve of `score_col` against a binary 0/1
    `target_col`, plus the Gini coefficient (2*AUC - 1). Emits a
    scalar_metric dict on the `metric` port: {"kind": "auc_gini", "auc",
    "gini"}. Assumes a higher score means higher probability of
    target == 1 (e.g. a predicted PD); a score where higher = safer gives
    AUC < 0.5. `score_col` auto-fills from the input's role=predicted
    column and `target_col` from its role=target column when left unset."""
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(df[target_col].to_numpy(), df[score_col].to_numpy()))
    return {"kind": "auc_gini", "auc": auc, "gini": 2 * auc - 1}


def _gini_value(gini: Any) -> list[str]:
    if not h.num(gini):
        return []
    if gini < 0:
        return ["Gini is negative -- the score runs the wrong way (higher = safer). Check score_col is the predicted PD."]
    if gini >= h.SUSPICIOUS_GINI:
        return [f"Gini {gini:.2f} is unusually high for a credit model -- check the features for leakage before going on."]
    if gini < h.WEAK_GINI:
        return [f"Gini {gini:.2f} is weak -- revisit feature selection before validating further."]
    return [f"Gini {gini:.2f} is in the usual range."]


def _auc_gini_hints(outputs: dict[str, Any]) -> list[str]:
    return _gini_value(h.metric(outputs).get("gini"))


register_block(
    BlockSpec(
        category="auc_gini",
        block_type="output",
        group="tests",
        display_name="AUC / Gini",
        tags=("performance",),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=auc_gini,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("score_col", "Score column", "column", auto_role="predicted"),
            FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
        ),
        hints=_auc_gini_hints,
    )
)


def psi_test(expected: pl.DataFrame, actual: pl.DataFrame, col: str, bins: int = 10) -> dict:
    """Population Stability Index: how much `col`'s distribution in `actual`
    has drifted from `expected` (e.g. current score distribution vs. the
    distribution the model was built/validated on). Numeric columns are
    bucketed on `expected`'s quantiles; a text/boolean column compares
    category shares instead. Nulls get their own bucket either way. For
    every feature at once, use characteristic_stability."""
    import numpy as np

    e_s, a_s = expected[col], actual[col]
    labels: list[str] | None = None
    if e_s.dtype.is_numeric() and e_s.dtype != pl.Boolean:
        exp_vals = e_s.cast(pl.Float64).to_numpy()
        act_vals = a_s.cast(pl.Float64).to_numpy()
        e_null, a_null = np.isnan(exp_vals), np.isnan(act_vals)
        edges = np.quantile(exp_vals[~e_null], np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
        exp_counts, _ = np.histogram(exp_vals[~e_null], bins=edges)
        act_counts, _ = np.histogram(act_vals[~a_null], bins=edges)
        if e_null.any() or a_null.any():
            exp_counts = np.append(exp_counts, e_null.sum())
            act_counts = np.append(act_counts, a_null.sum())
    else:
        e_k = e_s.cast(pl.Utf8).fill_null("__missing__").to_list()
        a_k = a_s.cast(pl.Utf8).fill_null("__missing__").to_list()
        labels = sorted(set(e_k) | set(a_k))
        exp_counts = np.array([e_k.count(k) for k in labels])
        act_counts = np.array([a_k.count(k) for k in labels])
    exp_pct = exp_counts / max(len(e_s), 1)
    act_pct = act_counts / max(len(a_s), 1)
    # Laplace-style smoothing so an empty bucket never produces ln(0).
    eps = 1e-6
    contributions = (act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps))
    buckets = [
        {"expected_pct": float(e), "actual_pct": float(a), "contribution": float(c)}
        for e, a, c in zip(exp_pct, act_pct, contributions)
    ]
    if labels is not None:
        for b, lab in zip(buckets, labels):
            b["category"] = lab
    return {"kind": "psi_test", "psi": float(contributions.sum()), "buckets": buckets}


def rating_summary(df: pl.DataFrame, grade_col: str, target_col: str) -> dict:
    """Per-grade population and observed default rate -- the standard check
    that a rating scale (see modelling.fit_master_scale/assign_rating_grade)
    is monotonic: default rate should rise, never fall, from the
    lowest-risk grade to the highest."""
    stats = df.group_by(grade_col).agg(n=pl.len(), default_rate=pl.col(target_col).mean())
    stats = stats.sort(pl.col(grade_col).cast(pl.Int64, strict=False), nulls_last=True)
    rows = stats.to_dicts()
    rates = [r["default_rate"] for r in rows]
    monotonic = all(a <= b for a, b in zip(rates, rates[1:]))
    return {"kind": "rating_summary", "grades": rows, "monotonic": monotonic}


def _rating_summary_hints(outputs: dict[str, Any]) -> list[str]:
    metric = h.metric(outputs)
    if "monotonic" not in metric:
        return []
    if metric["monotonic"]:
        return ["Default rates rise grade by grade, as they should."]
    return [
        "Default rates aren't monotonic across grades -- refit the master scale with algorithm "
        "'monotonic_default_rate' (or fewer grades)."
    ]


register_block(
    BlockSpec(
        category="rating_summary",
        block_type="output",
        group="tests",
        display_name="Rating scale summary",
        tags=("rating_scale", "pd"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=rating_summary,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("grade_col", "Grade column", "column"),
            FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
        ),
        hints=_rating_summary_hints,
    )
)


def _psi_test_hints(outputs: dict[str, Any]) -> list[str]:
    psi = h.metric(outputs).get("psi")
    if not h.num(psi):
        return []
    if psi >= h.PSI_SHIFTED:
        return [f"PSI {psi:.3f} >= {h.PSI_SHIFTED}: a significant shift -- ask_user before relying on it."]
    if psi >= h.PSI_STABLE:
        return [f"PSI {psi:.3f}: a moderate shift -- worth noting in the stage summary."]
    return [f"PSI {psi:.3f} < {h.PSI_STABLE}: stable."]


register_block(
    BlockSpec(
        category="psi_test",
        block_type="output",
        group="tests",
        display_name="PSI test",
        tags=("stability",),
        inputs=[PortSpec("expected"), PortSpec("actual")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=psi_test,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("col", "Column to compare", "column"),
            FieldSpec("bins", "Bins", "number"),
        ),
        hints=_psi_test_hints,
    )
)


def roc_curve(df: pl.DataFrame, score_col: str, target_col: str) -> dict:
    """The full ROC curve -- false/true positive rate at each distinct
    score threshold -- not just the scalar AUC that auc_gini reports, for
    plotting or a closer look at where a model's discrimination actually
    comes from."""
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve as _roc_curve

    y = df[target_col].to_numpy()
    scores = df[score_col].to_numpy()
    fpr, tpr, thresholds = _roc_curve(y, scores)
    auc = float(roc_auc_score(y, scores))
    return {
        "kind": "roc_curve",
        "auc": auc,
        # sklearn's first threshold is a synthetic max(score)+1 (not a real
        # cutoff, just "reject everything") and can come back as +inf --
        # nulled out rather than left as a non-JSON float.
        "points": [
            {"fpr": float(f), "tpr": float(t), "threshold": float(th) if th == th and th != float("inf") else None}
            for f, t, th in zip(fpr, tpr, thresholds)
        ],
    }


def _roc_curve_hints(outputs: dict[str, Any]) -> list[str]:
    auc = h.metric(outputs).get("auc")
    return _gini_value(2 * auc - 1) if h.num(auc) else []


register_block(
    BlockSpec(
        category="roc_curve",
        block_type="output",
        group="tests",
        display_name="ROC curve",
        tags=("performance", "output"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=roc_curve,
        metadata_transform=lambda *_a, **_k: {},
        hints=_roc_curve_hints,
    )
)


def calibration_test(df: pl.DataFrame, score_col: str, target_col: str, bins: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test: buckets rows into `bins`
    quantile groups by predicted probability (`score_col`), then compares
    each bucket's observed event count to its expected count (the sum of
    predicted probabilities in it). A large HL statistic (low p-value)
    means the model's predicted probabilities don't match reality well,
    even when discrimination (AUC/KS) looks fine -- calibration and
    discrimination are different things, and this is what checks the
    former."""
    from scipy.stats import chi2

    bucketed = df.with_columns(pl.col(score_col).qcut(bins, allow_duplicates=True).alias("_bucket"))
    stats = bucketed.group_by("_bucket").agg(
        n=pl.len(),
        observed=pl.col(target_col).sum(),
        expected=pl.col(score_col).sum(),
    ).sort("expected")
    rows = stats.to_dicts()

    hl_statistic = 0.0
    for r in rows:
        n, observed, expected = r["n"], float(r["observed"]), float(r["expected"])
        # A bucket with nothing expected/expected==n contributes an
        # undefined (0/0) term -- excluded rather than let it poison the
        # sum, the same way psi_test/iv_table smooth an empty bucket away
        # instead of blowing up on it.
        if expected <= 0 or expected >= n:
            continue
        hl_statistic += (observed - expected) ** 2 / (expected * (1 - expected / n))
    degrees_of_freedom = max(len(rows) - 2, 1)

    return {
        "kind": "calibration_test",
        "hl_statistic": float(hl_statistic),
        "p_value": float(chi2.sf(hl_statistic, degrees_of_freedom)),
        "degrees_of_freedom": degrees_of_freedom,
        "buckets": [
            {
                "bucket": str(r["_bucket"]),
                "n": r["n"],
                "observed_count": float(r["observed"]),
                "expected_count": float(r["expected"]),
            }
            for r in rows
        ],
    }


def _calibration_test_hints(outputs: dict[str, Any]) -> list[str]:
    p = h.metric(outputs).get("p_value")
    if not h.num(p):
        return []
    if p < h.CALIBRATION_P:
        return [
            f"Hosmer-Lemeshow p = {p:.3g} < {h.CALIBRATION_P}: predicted PDs don't match observed rates -- calibrate "
            "(calibrate_model) or explain why not in the stage summary."
        ]
    return [f"Hosmer-Lemeshow p = {p:.3g}: calibration is acceptable."]


register_block(
    BlockSpec(
        category="calibration_test",
        block_type="output",
        group="tests",
        display_name="Calibration (Hosmer-Lemeshow)",
        tags=("calibration", "pd"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=calibration_test,
        metadata_transform=lambda *_a, **_k: {},
        hints=_calibration_test_hints,
    )
)


def continuous_accuracy(df: pl.DataFrame, actual_col: str, predicted_col: str) -> dict:
    """MAE/MSE/RMSE/R^2 between a continuous bounded target (LGD, CCF, EAD)
    and its prediction -- the accuracy battery a continuous target needs
    instead of the discrimination metrics above (Gini/KS/AUC only make
    sense for a binary target; a validator will ask for this, not those,
    on an LGD or EAD model). Also reports the rank-ordering measures used
    for a continuous target's discrimination -- Spearman's rho and
    Kendall's tau-b between prediction and outcome, and the CAP-style
    accuracy ratio (the loss-share analogue of Gini: how much of the total
    realised value the prediction's ranking concentrates at the top,
    relative to a perfect ranking) -- plus the mean actual vs. mean
    predicted and their difference (`bias`, positive = over-prediction).
    `actual_col` auto-fills from the role=target column."""
    import numpy as np
    from scipy.stats import kendalltau, spearmanr

    actual = df[actual_col].to_numpy().astype(float)
    predicted = df[predicted_col].to_numpy().astype(float)

    def _cap_area(order_by: np.ndarray) -> float:
        order = np.argsort(-order_by, kind="stable")
        cum = np.cumsum(actual[order]) / actual.sum()
        curve = np.concatenate([[0.0], cum])
        return float(((curve[1:] + curve[:-1]) / 2.0).sum() / len(actual))

    accuracy_ratio = None
    if actual.sum() > 0 and len(actual) > 1:
        model_area, perfect_area = _cap_area(predicted) - 0.5, _cap_area(actual) - 0.5
        accuracy_ratio = float(model_area / perfect_area) if perfect_area > 0 else None
    rankable = len(actual) > 2 and actual.std() > 0 and predicted.std() > 0
    rho = spearmanr(actual, predicted).statistic if rankable else float("nan")
    tau = kendalltau(actual, predicted).statistic if rankable else float("nan")
    errors = predicted - actual
    mse = float(np.mean(errors**2))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "kind": "continuous_accuracy",
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "mean_actual": float(actual.mean()),
        "mean_predicted": float(predicted.mean()),
        "bias": float(predicted.mean() - actual.mean()),
        "spearman": None if np.isnan(rho) else float(rho),
        "kendall_tau": None if np.isnan(tau) else float(tau),
        "accuracy_ratio": accuracy_ratio,
    }


def _continuous_accuracy_hints(outputs: dict[str, Any]) -> list[str]:
    m = h.metric(outputs)
    hints = []
    if h.num(m.get("r2")) and m["r2"] < 0:
        hints.append("R^2 is negative -- the model predicts worse than the mean. Revisit the features.")
    if h.num(m.get("spearman")) and m["spearman"] < h.WEAK_SPEARMAN:
        hints.append(f"Spearman {m['spearman']:.2f} is weak -- the prediction barely ranks the outcome.")
    actual, bias = m.get("mean_actual"), m.get("bias")
    if h.num(actual) and h.num(bias) and actual and abs(bias) / abs(actual) > h.CALIBRATION_GAP:
        hints.append(
            f"Mean prediction is {abs(bias) / abs(actual):.0%} {'above' if bias > 0 else 'below'} the mean outcome -- "
            "calibrate the model."
        )
    if m and not hints and h.num(m.get("spearman")):
        hints.append(f"Accuracy looks reasonable (Spearman {m['spearman']:.2f}, bias {m.get('bias', 0):+.4f}).")
    return hints


register_block(
    BlockSpec(
        category="continuous_accuracy",
        block_type="output",
        group="tests",
        display_name="Continuous accuracy (MAE/MSE/RMSE)",
        tags=("performance", "lgd", "ccf_ead"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=continuous_accuracy,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("actual_col", "Actual column", "column", auto_role="target"),
            FieldSpec("predicted_col", "Predicted column", "column", auto_role="predicted"),
        ),
        hints=_continuous_accuracy_hints,
    )
)


def bucketed_calibration(df: pl.DataFrame, actual_col: str, predicted_col: str, bins: int = 10) -> dict:
    """Observed-vs-expected by bucket for a continuous bounded target (LGD,
    CCF, EAD): quantile-buckets rows by their predicted value, then compares
    each bucket's mean observed value to its mean predicted value -- the
    standard LGD/EAD validation check, distinct from calibration_test above
    (which is Hosmer-Lemeshow, built for a binary target's predicted
    probability, not a continuous one)."""
    bucketed = df.with_columns(pl.col(predicted_col).qcut(bins, allow_duplicates=True).alias("_bucket"))
    stats = bucketed.group_by("_bucket").agg(
        n=pl.len(),
        observed_mean=pl.col(actual_col).mean(),
        predicted_mean=pl.col(predicted_col).mean(),
    ).sort("predicted_mean")
    rows = stats.to_dicts()
    return {
        "kind": "bucketed_calibration",
        "buckets": [
            {
                "bucket": str(r["_bucket"]),
                "n": r["n"],
                "observed_mean": float(r["observed_mean"]),
                "predicted_mean": float(r["predicted_mean"]),
            }
            for r in rows
        ],
    }


def _bucketed_calibration_hints(outputs: dict[str, Any]) -> list[str]:
    buckets = h.metric(outputs).get("buckets") or []
    off = [
        b["bucket"]
        for b in buckets
        if h.num(b.get("observed_mean"))
        and h.num(b.get("predicted_mean"))
        and b["predicted_mean"]
        and abs(b["observed_mean"] - b["predicted_mean"]) / abs(b["predicted_mean"]) > h.CALIBRATION_GAP
    ]
    if off:
        return [
            f"{len(off)} of {len(buckets)} buckets miss their prediction by more than {h.CALIBRATION_GAP:.0%}: "
            f"{h.names(off)} -- calibrate the model, or say why in the stage summary."
        ]
    return ["Every bucket's observed mean is close to its prediction."] if buckets else []


register_block(
    BlockSpec(
        category="bucketed_calibration",
        block_type="output",
        group="tests",
        display_name="Bucketed O vs E",
        tags=("calibration", "lgd", "ccf_ead"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=bucketed_calibration,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("actual_col", "Actual column", "column", auto_role="target"),
            FieldSpec("predicted_col", "Predicted column", "column", auto_role="predicted"),
            FieldSpec("bins", "Bins", "number"),
        ),
        hints=_bucketed_calibration_hints,
    )
)


def grade_backtest(
    df: pl.DataFrame, grade_col: str, target_col: str, pd_col: str, confidence: float = 0.95
) -> dict:
    """Grade-level PD back-test -- the standard IRB/ECB validation of a
    rating scale's calibration. Per grade: count, defaults, observed
    default rate, the grade's PD (mean of `pd_col`, e.g. assign_rating_grade's
    `grade_pd` or the model's predicted PD), and two one-sided tests of
    H0 "the PD is not too low":
      - binomial: P(X >= defaults) with X ~ Binomial(n, PD);
      - Jeffreys: the Beta(defaults + 0.5, n - defaults + 0.5) posterior's
        probability that the true default rate is <= PD.
    A p-value below 1 - `confidence` flags the grade ("red"; "amber" below
    twice that): its PD underestimates the observed default rate. Also
    reports the same tests at portfolio level, the Herfindahl index of the
    grade population (concentration; ~1/n_grades is even, 1 = everything
    in one grade), and whether the observed rate is monotonic across
    grades. `target_col` auto-fills from the role=target column."""
    from scipy.stats import beta, binom

    alpha = 1.0 - confidence

    def _tests(n: int, d: int, pd_: float) -> dict:
        p_binom = float(binom.sf(d - 1, n, pd_)) if n else None
        p_jeff = float(beta.cdf(pd_, d + 0.5, n - d + 0.5)) if n else None
        worst = min(v for v in (p_binom, p_jeff) if v is not None) if n else None
        light = None if worst is None else ("red" if worst < alpha else ("amber" if worst < 2 * alpha else "green"))
        return {"binomial_p_value": p_binom, "jeffreys_p_value": p_jeff, "traffic_light": light}

    stats = (
        df.group_by(grade_col)
        .agg(n=pl.len(), defaults=pl.col(target_col).sum(), pd=pl.col(pd_col).mean())
        .sort(pl.col(grade_col).cast(pl.Int64, strict=False), pl.col(grade_col), nulls_last=True)
    )
    total = df.height
    grades = []
    for r in stats.to_dicts():
        n, d, pd_ = int(r["n"]), int(r["defaults"]), float(r["pd"])
        grades.append({"grade": r[grade_col], "n": n, "defaults": d, "observed_dr": d / n if n else None, "pd": pd_, **_tests(n, d, pd_)})
    rates = [g["observed_dr"] for g in grades]
    n_def = int(df[target_col].sum())
    portfolio_pd = float(df[pd_col].mean())
    return {
        "kind": "grade_backtest",
        "confidence": confidence,
        "grades": grades,
        "portfolio": {"n": total, "defaults": n_def, "observed_dr": n_def / total if total else None, "pd": portfolio_pd, **_tests(total, n_def, portfolio_pd)},
        "herfindahl": float(sum((g["n"] / total) ** 2 for g in grades)) if total else None,
        "monotonic": all(a <= b for a, b in zip(rates, rates[1:])),
        "n_red": sum(1 for g in grades if g["traffic_light"] == "red"),
    }


def _grade_backtest_hints(outputs: dict[str, Any]) -> list[str]:
    metric = h.metric(outputs)
    grades = metric.get("grades") or []
    if not grades:
        return []
    hints = []
    red = [str(g["grade"]) for g in grades if g.get("traffic_light") == "red"]
    if red:
        hints.append(
            f"Red grades ({h.names(red)}): their PD is too low for the observed default rate -- recalibrate, or add a "
            "margin of conservatism."
        )
    if (metric.get("portfolio") or {}).get("traffic_light") == "red":
        hints.append("The portfolio-level PD is too low -- recalibrate before anything else.")
    if metric.get("monotonic") is False:
        hints.append("Observed rates aren't monotonic across grades -- refit the master scale ('monotonic_default_rate').")
    hhi = metric.get("herfindahl")
    if h.num(hhi) and hhi > 2 / len(grades):
        hints.append(f"The population is concentrated in few grades (Herfindahl {hhi:.2f}) -- consider more even grades.")
    if not hints:
        hints.append("The back-test passes: no red grades, monotonic, evenly spread.")
    return hints


register_block(
    BlockSpec(
        category="grade_backtest",
        block_type="output",
        group="tests",
        display_name="Grade PD back-test (binomial / Jeffreys)",
        tags=("rating_scale", "calibration", "pd"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=grade_backtest,
        metadata_transform=lambda *_a, **_k: {},
        form=(
            FieldSpec("grade_col", "Grade column", "column"),
            FieldSpec("target_col", "Default flag column", "column", auto_role="target"),
            FieldSpec("pd_col", "Grade PD column (e.g. grade_pd)", "column"),
            FieldSpec("confidence", "Confidence level", "number", step=0.005),
        ),
        hints=_grade_backtest_hints,
    )
)


def compare_samples(
    sample_1: pl.DataFrame,
    target_col: str,
    score_col: str,
    sample_2: pl.DataFrame | None = None,
    sample_3: pl.DataFrame | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Side-by-side performance of one model on up to three samples --
    typically train, test and out-of-time -- in one table instead of a
    metric block per sample. One row per wired sample (named by `labels`,
    default "train"/"test"/"oot"): row count, mean outcome, mean
    prediction, and, for a binary target, AUC, Gini and KS; for a
    continuous one (LGD/CCF), RMSE, MAE, R^2 and Spearman's rho. A drop in
    Gini (or rho) from train to test/OOT is the overfitting / stability
    signal. `target_col` and `score_col` auto-fill from the role=target
    and role=predicted columns."""
    import numpy as np
    from scipy.stats import ks_2samp, spearmanr
    from sklearn.metrics import roc_auc_score

    names = labels or ["train", "test", "oot"]
    rows = []
    for i, sample in enumerate([sample_1, sample_2, sample_3]):
        if sample is None:
            continue
        y = sample[target_col].cast(pl.Float64).to_numpy()
        p = sample[score_col].cast(pl.Float64).to_numpy()
        row = {
            "sample": names[i] if i < len(names) else f"sample_{i + 1}",
            "n": len(y),
            "mean_actual": float(y.mean()) if len(y) else None,
            "mean_predicted": float(p.mean()) if len(p) else None,
        }
        if np.isin(np.unique(y), [0.0, 1.0]).all():
            both = len(np.unique(y)) == 2
            auc = float(roc_auc_score(y, p)) if both else None
            row.update(
                {
                    "auc": auc,
                    "gini": 2 * auc - 1 if auc is not None else None,
                    "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic) if both else None,
                }
            )
        else:
            err = p - y
            ss_tot = float(((y - y.mean()) ** 2).sum())
            row.update(
                {
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "mae": float(np.mean(np.abs(err))),
                    "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else None,
                    "spearman": float(spearmanr(y, p).statistic) if len(y) > 2 else None,
                }
            )
        rows.append(row)
    return pl.DataFrame(rows)


def _compare_samples_hints(outputs: dict[str, Any]) -> list[str]:
    rows = h.rows(outputs.get("table"))
    if not rows:
        return []
    hints = []
    base = rows[0]
    for key, limit in (("gini", h.OVERFIT_GINI_DROP), ("spearman", h.OVERFIT_SPEARMAN_DROP)):
        for r in rows[1:]:
            if h.num(base.get(key)) and h.num(r.get(key)) and base[key] - r[key] > limit:
                hints.append(
                    f"{key.title()} falls {base[key] - r[key]:.2f} from {base['sample']} ({base[key]:.2f}) to "
                    f"{r['sample']} ({r[key]:.2f}) -- likely overfitting: try fewer features or stronger regularisation."
                )
    for r in rows:
        actual, predicted = r.get("mean_actual"), r.get("mean_predicted")
        if h.num(actual) and h.num(predicted) and actual:
            gap = (predicted - actual) / abs(actual)
            if abs(gap) > h.CALIBRATION_GAP:
                hints.append(
                    f"On {r['sample']} the mean prediction ({predicted:.4f}) is {abs(gap):.0%} "
                    f"{'above' if gap > 0 else 'below'} the mean outcome ({actual:.4f}) -- calibrate the model."
                )
    if not hints:
        hints.append("Performance holds across the samples and predictions match outcomes on average.")
    return hints


register_block(
    BlockSpec(
        category="compare_samples",
        block_type="output",
        group="tests",
        display_name="Compare samples (train/test/OOT)",
        tags=("performance", "stability"),
        inputs=[PortSpec("sample_1"), PortSpec("sample_2", required=False), PortSpec("sample_3", required=False)],
        outputs=[PortSpec("table")],
        aggregate_outputs=("table",),
        fn=compare_samples,
        metadata_transform=infer_dtypes,
        hints=_compare_samples_hints,
    )
)
