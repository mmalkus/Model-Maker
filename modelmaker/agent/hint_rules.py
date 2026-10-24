"""Decision hints for an AI build, by rule name.

A block names the rule that judges its result with `"hints": "<rule>"` in
its BLOCKS entry; the rule reads the block's outputs ({port: value}) and
returns short "what this result means" notes (see hints.decision_hints).
Thresholds and helpers live in agent/hints.py. Each rule documents the
output shape it expects, so a library block producing the same shape (e.g.
a scalar_metric dict with a "gini" key on a `metric` port) can reuse it.
"""

from __future__ import annotations

from typing import Any, Callable



from ..packet import DataFramePacket
from . import hints as h

HintsFn = Callable[[dict[str, Any]], list[str]]


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


def _split_hints(outputs: dict[str, Any]) -> list[str]:
    samples = {}
    for port, packet in outputs.items():
        target = h.target_column(packet)
        if target is None or packet.data.height == 0:
            continue
        y = packet.data[target].drop_nulls()
        if y.len() == 0 or not y.dtype.is_numeric():
            continue
        binary = set(y.unique().to_list()) <= {0, 1}
        samples[port] = (packet.data.height, float(y.mean()), int(y.sum()) if binary else None)
    hints = []
    for port, packet in outputs.items():
        if isinstance(packet, DataFramePacket) and packet.data.height == 0:
            hints.append(f"{port} is empty -- check the split settings (test_size, cutoff, oot_end).")
    thin = [f"{p} has {ev}" for p, (_, _, ev) in samples.items() if ev is not None and ev < h.MIN_EVENTS]
    if thin:
        hints.append(f"Few events (< {h.MIN_EVENTS}): {h.names(thin)} -- metrics on it will be noisy.")
    if len(samples) == 2:
        (a, (_, ra, _)), (b, (_, rb, _)) = samples.items()
        if ra and abs(rb - ra) / abs(ra) > h.SPLIT_RATE_GAP:
            hints.append(
                f"Target rate differs between {a} ({ra:.4f}) and {b} ({rb:.4f}) -- for a random split, set "
                "stratify_col to the target."
            )
    return hints


def _fit_binning_hints(outputs: dict[str, Any]) -> list[str]:
    return h.iv_rows(h.rows(outputs.get("summary")), keep_monotonic=True)


def _data_profile_hints(outputs: dict[str, Any]) -> list[str]:
    cols = h.metric(outputs).get("columns") or []
    empty = [c["column"] for c in cols if h.num(c.get("fill_rate")) and c["fill_rate"] < h.LOW_FILL_RATE]
    flat = [
        c["column"]
        for c in cols
        if h.num(c.get("dominant_value_share")) and c["dominant_value_share"] >= h.DOMINANT_SHARE and c["column"] not in empty
    ]
    ids = [
        c["column"]
        for c in cols
        if h.num(c.get("distinct_ratio"))
        and c["distinct_ratio"] >= h.ID_LIKE_DISTINCT
        and not str(c.get("dtype", "")).startswith("Float")
    ]
    hints = []
    if empty:
        hints.append(f"Mostly empty (filled < {h.LOW_FILL_RATE:.0%}): {h.names(empty)} -- leave out, or treat the missing values.")
    if flat:
        hints.append(f"Near-constant (one value >= {h.DOMINANT_SHARE:.0%}): {h.names(flat)} -- leave out as features.")
    if ids:
        hints.append(f"Id-like (almost every value distinct): {h.names(ids)} -- never use as features.")
    if cols and not hints:
        hints.append("No empty, constant or id-like columns among those profiled.")
    return hints


def _apply_exclusions_hints(outputs: dict[str, Any]) -> list[str]:
    summary = h.metric(outputs, "summary")
    start, final = summary.get("starting_population"), summary.get("final_population")
    if not h.num(start) or not start:
        return []
    hints = []
    for step in summary.get("steps") or []:
        if step.get("rule") is None:
            continue
        if step.get("dropped") == 0:
            hints.append(f"Rule {step['step']!r} dropped nothing -- check its expression keeps the right rows.")
        elif h.num(step.get("dropped")) and step["dropped"] / start > h.BIG_EXCLUSION:
            hints.append(
                f"Rule {step['step']!r} dropped {step['dropped'] / start:.0%} of the population -- confirm that's "
                "intended (ask_user if the plan didn't say)."
            )
    if not hints and h.num(final):
        hints.append(f"Exclusions kept {final / start:.0%} of the population; no rule looks off.")
    return hints


def _data_quality_rules_hints(outputs: dict[str, Any]) -> list[str]:
    results = h.metric(outputs).get("results") or []
    errors = [r["name"] for r in results if not r.get("passed") and r.get("severity", "error") == "error"]
    warnings = [r["name"] for r in results if not r.get("passed") and r.get("severity") == "warning"]
    hints = []
    if errors:
        hints.append(f"Failed: {h.names(errors)} -- fix the data (or ask_user) before modelling on it.")
    if warnings:
        hints.append(f"Warnings: {h.names(warnings)} -- mention them in the stage summary.")
    if results and not hints:
        hints.append("Every data-quality rule passed.")
    return hints


def _iv_table_hints(outputs: dict[str, Any]) -> list[str]:
    return h.iv_rows(h.metric(outputs).get("rows") or [], keep_monotonic=False)


def _correlation_matrix_hints(outputs: dict[str, Any]) -> list[str]:
    metric = h.metric(outputs)
    features, matrix = metric.get("features") or [], metric.get("correlation") or []
    pairs = [
        f"{features[i]}/{features[j]} ({matrix[i][j]:+.2f})"
        for i in range(len(features))
        for j in range(i + 1, len(features))
        if i < len(matrix) and j < len(matrix[i]) and h.num(matrix[i][j]) and abs(matrix[i][j]) >= h.HIGH_CORRELATION
    ]
    vif = [v["feature"] for v in metric.get("vif") or [] if h.num(v.get("vif")) and v["vif"] >= h.HIGH_VIF]
    hints = []
    if pairs:
        hints.append(f"Highly correlated (|r| >= {h.HIGH_CORRELATION}): {h.names(pairs)} -- keep the stronger of each pair.")
    if vif:
        hints.append(f"VIF >= {h.HIGH_VIF:g}: {h.names(vif)} -- largely redundant given the others; drop or merge.")
    if features and not hints:
        hints.append("No redundant features (no high correlations or VIFs).")
    return hints


def _characteristic_stability_hints(outputs: dict[str, Any]) -> list[str]:
    rows = h.rows(outputs.get("table"))
    unstable = [r["feature"] for r in rows if h.num(r.get("psi")) and r["psi"] >= h.PSI_SHIFTED]
    monitor = [r["feature"] for r in rows if h.num(r.get("psi")) and h.PSI_STABLE <= r["psi"] < h.PSI_SHIFTED]
    hints = []
    if unstable:
        hints.append(f"Unstable (PSI >= {h.PSI_SHIFTED}): {h.names(unstable)} -- leave out, or ask_user before keeping.")
    if monitor:
        hints.append(f"Shifting (PSI {h.PSI_STABLE}-{h.PSI_SHIFTED}): {h.names(monitor)} -- note them in the stage summary.")
    if rows and not hints:
        hints.append(f"Every feature is stable (PSI < {h.PSI_STABLE}).")
    return hints


def _target_trend_hints(outputs: dict[str, Any]) -> list[str]:
    rows = [r for r in h.rows(outputs.get("table")) if h.num(r.get("n"))]
    if len(rows) < 2:
        return []
    hints = []
    counts = sorted(r["n"] for r in rows)
    median = counts[len(counts) // 2]
    if rows[-1]["n"] < h.IMMATURE_LAST_PERIOD * median:
        hints.append(
            f"The last period ({rows[-1]['period']}) has {rows[-1]['n']} rows against a typical {median} -- likely "
            "immature; consider leaving it out (time_split's oot_end)."
        )
    thin = [str(r["period"]) for r in rows if r["n"] < h.THIN_PERIOD]
    if thin:
        hints.append(f"Periods with < {h.THIN_PERIOD} rows: {h.names(thin)} -- their target rates are noise.")
    return hints


def _coefficient_hints(model: Any, woe_signs: bool) -> list[str]:
    if not isinstance(model, dict):
        return []
    stats = {k: v for k, v in (model.get("statistics") or {}).items() if k != "intercept"}
    coefs = model.get("coefficients") or {k: v.get("estimate") for k, v in stats.items()}
    hints = []
    if model.get("converged") is False:
        hints.append("The fit didn't converge -- try fewer features, or check for collinear or constant ones.")
    # WoE is higher where the outcome is rarer, so every WoE coefficient
    # of a PD model should be negative (logistic_regression's docstring).
    wrong = (
        [f for f, c in coefs.items() if f.endswith("_woe") and h.num(c) and c > 0] if woe_signs else []
    )
    if wrong:
        hints.append(
            f"Positive coefficient on WoE feature(s) {h.names(wrong)} -- the wrong sign, usually collinearity. "
            "Drop them (or the feature they overlap with) and refit."
        )
    weak = [f for f, s in stats.items() if (s.get("p_value") or 0.0) > h.MAX_P_VALUE and f not in wrong]
    if weak:
        hints.append(f"Not significant (p > {h.MAX_P_VALUE}): {h.names(weak)} -- consider dropping them and refitting.")
    if coefs and not hints:
        woe = woe_signs and all(f.endswith("_woe") for f in coefs)
        hints.append(
            "Coefficients look sound: signs as expected and all significant." if woe else "All coefficients are significant."
        )
    return hints


def _logistic_regression_hints(outputs: dict[str, Any]) -> list[str]:
    return _coefficient_hints(outputs.get("model"), woe_signs=True)


def _lgd_regression_hints(outputs: dict[str, Any]) -> list[str]:
    return _coefficient_hints(outputs.get("model"), woe_signs=False)


def _fit_master_scale_hints(outputs: dict[str, Any]) -> list[str]:
    grades = h.metric(outputs, "master_scale").get("grades") or []
    if not grades:
        return []
    hints = []
    rates = [g.get("observed_rate") for g in grades]
    if all(h.num(r) for r in rates) and any(b < a for a, b in zip(rates, rates[1:])):
        hints.append("Observed rates aren't monotonic across grades -- refit with algorithm 'monotonic_default_rate'.")
    total = sum(g.get("n") or 0 for g in grades)
    thin = [str(g["grade"]) for g in grades if total and (g.get("n") or 0) / total < h.THIN_GRADE]
    if thin:
        hints.append(f"Thin grades ({h.names(thin)}, < {h.THIN_GRADE:.0%} of the population) -- set min_grade_share.")
    if not hints:
        hints.append(f"{len(grades)} grades, monotonic and none too thin.")
    return hints


def _forward_default_flag_hints(outputs: dict[str, Any]) -> list[str]:
    packet = outputs.get("out")
    target = h.target_column(packet)
    if target is None:
        return []
    flag = packet.data[target]
    if flag.len() == 0:
        return ["No rows left -- check id_col/date_col/default_col and the horizon against the data's span."]
    hints = []
    nulls = flag.null_count()
    if nulls:
        hints.append(
            f"{nulls} rows have an incomplete outcome window (null {target}) -- filter them out "
            f"({target} IS NOT NULL) before fitting."
        )
    events = int(flag.sum() or 0)
    if events < h.MIN_EVENTS:
        hints.append(f"Only {events} defaults (< {h.MIN_EVENTS}) -- a PD model on this will be unstable.")
    else:
        hints.append(f"{target}: {events} defaults in {flag.len() - nulls} rows ({events / (flag.len() - nulls):.2%}).")
    return hints


def _computed_target_hints(column: str) -> Callable[[dict[str, Any]], list[str]]:
    def hints(outputs: dict[str, Any]) -> list[str]:
        packet = outputs.get("out")
        if not isinstance(packet, DataFramePacket) or column not in packet.data.columns or packet.data.height == 0:
            return []
        nulls = packet.data[column].null_count() / packet.data.height
        if nulls > h.MAX_NULL_SHARE:
            return [
                f"{nulls:.0%} of rows have a null {column} -- filter them out ({column} IS NOT NULL) before fitting."
            ]
        return []

    return hints


def _stepwise_selection_hints(outputs: dict[str, Any]) -> list[str]:
    metric = h.metric(outputs)
    selected = metric.get("selected")
    if selected is None:
        return []
    if not selected:
        return ["Nothing passed p_enter -- revisit the candidate features (or loosen p_enter) before fitting."]
    hints = [f"Use these as the model's features: {h.names(selected)}."]
    if metric.get("dropped"):
        hints.append(f"Dropped along the way: {h.names(metric['dropped'])}.")
    if metric.get("target_type") == "binary":
        wrong = [
            c["feature"] for c in metric.get("coefficients") or [] if c["feature"].endswith("_woe") and c.get("sign") == "+"
        ]
        if wrong:
            hints.append(f"Wrong (positive) sign on {h.names(wrong)} -- drop them before fitting the final model.")
    return hints


def _calibrate_model_hints(outputs: dict[str, Any]) -> list[str]:
    cal = h.metric(outputs, "model").get("calibration") or {}
    shift, before, after = cal.get("intercept_shift"), cal.get("mean_prediction_before"), cal.get("mean_prediction_after")
    if not (h.num(shift) and h.num(before) and h.num(after)):
        return []
    if abs(shift) > h.BIG_INTERCEPT_SHIFT:
        return [
            f"A large shift (log-odds {shift:+.2f}, mean {before:.4f} -> {after:.4f}) -- check the central tendency "
            "is the right one, and mention it in the stage summary."
        ]
    return [f"Calibrated: mean prediction {before:.4f} -> {after:.4f}. Use this model for prediction from here on."]


def _long_run_average_hints(outputs: dict[str, Any]) -> list[str]:
    metric = h.metric(outputs)
    hints = []
    periods = metric.get("n_periods")
    if h.num(periods) and periods < h.SHORT_HISTORY:
        hints.append(
            f"Only {periods} period(s) -- likely short of a full cycle. Say so in the stage summary; a margin of "
            "conservatism may be needed."
        )
    dw, tw = metric.get("default_weighted"), metric.get("time_weighted")
    if h.num(dw) and h.num(tw) and dw and abs(tw - dw) / abs(dw) > h.LRA_GAP:
        hints.append(
            f"Default-weighted ({dw:.4f}) and time-weighted ({tw:.4f}) averages differ by more than {h.LRA_GAP:.0%} -- "
            "the choice matters; state which you use as the central tendency and why."
        )
    if not hints and h.num(periods):
        hints.append(f"{periods} periods, and the default- and time-weighted averages agree.")
    return hints


def _ks_test_hints(outputs: dict[str, Any]) -> list[str]:
    ks = h.metric(outputs).get("ks_statistic")
    if not h.num(ks):
        return []
    if ks >= h.SUSPICIOUS_KS:
        return [f"KS {ks:.2f} is unusually high for a credit model -- check the features for leakage."]
    if ks < h.WEAK_KS:
        return [f"KS {ks:.2f} is weak -- the score separates goods and bads poorly."]
    return [f"KS {ks:.2f} is in the usual range."]


def _auc_gini_hints(outputs: dict[str, Any]) -> list[str]:
    return _gini_value(h.metric(outputs).get("gini"))


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


def _psi_test_hints(outputs: dict[str, Any]) -> list[str]:
    psi = h.metric(outputs).get("psi")
    if not h.num(psi):
        return []
    if psi >= h.PSI_SHIFTED:
        return [f"PSI {psi:.3f} >= {h.PSI_SHIFTED}: a significant shift -- ask_user before relying on it."]
    if psi >= h.PSI_STABLE:
        return [f"PSI {psi:.3f}: a moderate shift -- worth noting in the stage summary."]
    return [f"PSI {psi:.3f} < {h.PSI_STABLE}: stable."]


def _roc_curve_hints(outputs: dict[str, Any]) -> list[str]:
    auc = h.metric(outputs).get("auc")
    return _gini_value(2 * auc - 1) if h.num(auc) else []


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


def _fit_distribution_hints(outputs: dict[str, Any]) -> list[str]:
    dist = h.metric(outputs, "distribution")
    stats = dist.get("fit_stats") or {}
    p = stats.get("ks_p")
    if not dist.get("family") or not h.num(p):
        return []
    if p < h.GOOD_FIT_P:
        return [
            f"The best family ({dist['family']}) still fits poorly (KS p = {p:.3g}) -- try other families, or "
            "spliced_tail for a heavy tail."
        ]
    return [f"{dist['family']} fits best (lowest AIC) and fits well (KS p = {p:.3g})."]


def _validate_proxy_hints(outputs: dict[str, Any]) -> list[str]:
    r2 = h.metric(outputs, "diagnostics").get("out_of_sample_r2")
    if not h.num(r2):
        return []
    if r2 < h.PROXY_MIN_R2:
        return [
            f"Out-of-sample R^2 {r2:.3f} < {h.PROXY_MIN_R2} -- the proxy isn't accurate enough: fit on more scenarios "
            "or a richer basis."
        ]
    return [f"Out-of-sample R^2 {r2:.3f}: the proxy is accurate enough to use."]


RULES: dict[str, HintsFn] = {
    "apply_exclusions": _apply_exclusions_hints,
    "auc_gini": _auc_gini_hints,
    "bucketed_calibration": _bucketed_calibration_hints,
    "calibrate_model": _calibrate_model_hints,
    "calibration_test": _calibration_test_hints,
    "characteristic_stability": _characteristic_stability_hints,
    "compare_samples": _compare_samples_hints,
    "computed_ccf": _computed_target_hints("ccf"),
    "computed_lgd": _computed_target_hints("lgd"),
    "continuous_accuracy": _continuous_accuracy_hints,
    "correlation_matrix": _correlation_matrix_hints,
    "data_profile": _data_profile_hints,
    "data_quality_rules": _data_quality_rules_hints,
    "fit_binning": _fit_binning_hints,
    "fit_distribution": _fit_distribution_hints,
    "fit_master_scale": _fit_master_scale_hints,
    "forward_default_flag": _forward_default_flag_hints,
    "grade_backtest": _grade_backtest_hints,
    "iv_table": _iv_table_hints,
    "ks_test": _ks_test_hints,
    "lgd_regression": _lgd_regression_hints,
    "logistic_regression": _logistic_regression_hints,
    "long_run_average": _long_run_average_hints,
    "psi_test": _psi_test_hints,
    "rating_summary": _rating_summary_hints,
    "roc_curve": _roc_curve_hints,
    "split": _split_hints,
    "stepwise_selection": _stepwise_selection_hints,
    "target_trend": _target_trend_hints,
    "validate_proxy": _validate_proxy_hints,
}
