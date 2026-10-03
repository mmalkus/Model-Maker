"""Decision hints (BuildOptions.decision_hints): short, rule-based "what
this result means and what to do next" notes added to run_to's report
when a block that the build has to judge ran green -- data checks,
feature screens, fits, discrimination, stability and calibration tests.

They move judgement out of the model and into code, for small models
that call tools reliably but read a statistics table poorly. A hint is a
suggestion, not a guard: the model may depart from one with a reason
(note_deviation). The thresholds are the usual credit-risk rules of
thumb; they are the modelling policy a hinted build follows, so they're
kept together here, named, to be read and changed in one place.

Blocks with nothing to judge (reads, joins, transforms, plots, fitted
artifacts used only by wiring) get no hints; neither does glm_fit, whose
artifact carries no statistics to judge it by."""

from __future__ import annotations

from typing import Any, Callable

from ..packet import ColumnRole, DataFramePacket

# ---- thresholds ----------------------------------------------------------------

# Data quality.
LOW_FILL_RATE = 0.5  # a column filled below this is mostly empty
DOMINANT_SHARE = 0.95  # one value this common makes a column near-constant
ID_LIKE_DISTINCT = 0.95  # a text/integer column this distinct is an id
BIG_EXCLUSION = 0.2  # one exclusion rule dropping this share of the start
MIN_EVENTS = 50  # defaults a sample needs for a stable validation
SPLIT_RATE_GAP = 0.2  # relative event-rate gap between two samples
MAX_NULL_SHARE = 0.0  # computed LGD/CCF nulls tolerated before a fit

# Feature screens. fit_binning/iv_table IV bands (see binning._iv_band):
# below USELESS_IV a feature carries no signal; at or above SUSPICIOUS_IV
# it usually leaks the target.
USELESS_IV = 0.02
SUSPICIOUS_IV = 0.5
# Continuous target: a binned feature explaining less than this share of
# the target's variance isn't worth a model slot.
WEAK_R2 = 0.005
HIGH_CORRELATION = 0.7  # |Pearson r| of a redundant pair
HIGH_VIF = 10.0  # variance-inflation factor of a redundant feature
THIN_PERIOD = 30  # rows in a period below which its rate is noise
IMMATURE_LAST_PERIOD = 0.5  # last period's rows vs. the median period's

# Fits.
MAX_P_VALUE = 0.05  # a non-intercept coefficient this unlikely to be non-zero
BIG_INTERCEPT_SHIFT = 1.0  # calibration moving the log-odds this far (~2.7x)
THIN_GRADE = 0.03  # a grade holding less of the population than this
SHORT_HISTORY = 5  # periods a long-run average should span (about a cycle)
LRA_GAP = 0.1  # relative gap between default- and time-weighted averages

# Discrimination (binary target, on any one sample).
WEAK_GINI = 0.2
SUSPICIOUS_GINI = 0.85
WEAK_KS = 0.2
SUSPICIOUS_KS = 0.7
OVERFIT_GINI_DROP = 0.05  # train -> test/OOT, absolute
# Continuous target: rank correlation of prediction and outcome.
WEAK_SPEARMAN = 0.2
OVERFIT_SPEARMAN_DROP = 0.05

# Calibration and stability.
CALIBRATION_GAP = 0.2  # mean prediction vs. mean outcome, relative
CALIBRATION_P = 0.05  # Hosmer-Lemeshow p-value below which PDs don't fit
PSI_STABLE = 0.1
PSI_SHIFTED = 0.25

# Stochastic.
GOOD_FIT_P = 0.05  # KS p-value below which even the best family fits badly
PROXY_MIN_R2 = 0.98  # a proxy's out-of-sample R^2 to rely on it

MAX_NAMES = 12  # names listed per hint, at most


# ---- helpers ---------------------------------------------------------------------


def _names(items: list[str]) -> str:
    shown = ", ".join(str(i) for i in items[:MAX_NAMES])
    return shown + (f" (+{len(items) - MAX_NAMES} more)" if len(items) > MAX_NAMES else "")


def _rows(value: Any) -> list[dict[str, Any]]:
    return value.data.to_dicts() if isinstance(value, DataFramePacket) else []


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _metric(outputs: dict[str, Any], port: str = "metric") -> dict[str, Any]:
    value = outputs.get(port)
    return value if isinstance(value, dict) else {}


def _target(packet: Any) -> str | None:
    if not isinstance(packet, DataFramePacket):
        return None
    return next(
        (n for n, m in packet.schema_meta.items() if m.role == ColumnRole.TARGET and n in packet.data.columns), None
    )


# ---- data --------------------------------------------------------------------------


def _profile(outputs: dict[str, Any]) -> list[str]:
    cols = _metric(outputs).get("columns") or []
    empty = [c["column"] for c in cols if _num(c.get("fill_rate")) and c["fill_rate"] < LOW_FILL_RATE]
    flat = [
        c["column"]
        for c in cols
        if _num(c.get("dominant_value_share")) and c["dominant_value_share"] >= DOMINANT_SHARE and c["column"] not in empty
    ]
    ids = [
        c["column"]
        for c in cols
        if _num(c.get("distinct_ratio"))
        and c["distinct_ratio"] >= ID_LIKE_DISTINCT
        and not str(c.get("dtype", "")).startswith("Float")
    ]
    hints = []
    if empty:
        hints.append(f"Mostly empty (filled < {LOW_FILL_RATE:.0%}): {_names(empty)} -- leave out, or treat the missing values.")
    if flat:
        hints.append(f"Near-constant (one value >= {DOMINANT_SHARE:.0%}): {_names(flat)} -- leave out as features.")
    if ids:
        hints.append(f"Id-like (almost every value distinct): {_names(ids)} -- never use as features.")
    if cols and not hints:
        hints.append("No empty, constant or id-like columns among those profiled.")
    return hints


def _exclusions(outputs: dict[str, Any]) -> list[str]:
    summary = _metric(outputs, "summary")
    start, final = summary.get("starting_population"), summary.get("final_population")
    if not _num(start) or not start:
        return []
    hints = []
    for step in summary.get("steps") or []:
        if step.get("rule") is None:
            continue
        if step.get("dropped") == 0:
            hints.append(f"Rule {step['step']!r} dropped nothing -- check its expression keeps the right rows.")
        elif _num(step.get("dropped")) and step["dropped"] / start > BIG_EXCLUSION:
            hints.append(
                f"Rule {step['step']!r} dropped {step['dropped'] / start:.0%} of the population -- confirm that's "
                "intended (ask_user if the plan didn't say)."
            )
    if not hints and _num(final):
        hints.append(f"Exclusions kept {final / start:.0%} of the population; no rule looks off.")
    return hints


def _quality_rules(outputs: dict[str, Any]) -> list[str]:
    results = _metric(outputs).get("results") or []
    errors = [r["name"] for r in results if not r.get("passed") and r.get("severity", "error") == "error"]
    warnings = [r["name"] for r in results if not r.get("passed") and r.get("severity") == "warning"]
    hints = []
    if errors:
        hints.append(f"Failed: {_names(errors)} -- fix the data (or ask_user) before modelling on it.")
    if warnings:
        hints.append(f"Warnings: {_names(warnings)} -- mention them in the stage summary.")
    if results and not hints:
        hints.append("Every data-quality rule passed.")
    return hints


def _split(outputs: dict[str, Any]) -> list[str]:
    samples = {}
    for port, packet in outputs.items():
        target = _target(packet)
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
    thin = [f"{p} has {ev}" for p, (_, _, ev) in samples.items() if ev is not None and ev < MIN_EVENTS]
    if thin:
        hints.append(f"Few events (< {MIN_EVENTS}): {_names(thin)} -- metrics on it will be noisy.")
    if len(samples) == 2:
        (a, (_, ra, _)), (b, (_, rb, _)) = samples.items()
        if ra and abs(rb - ra) / abs(ra) > SPLIT_RATE_GAP:
            hints.append(
                f"Target rate differs between {a} ({ra:.4f}) and {b} ({rb:.4f}) -- for a random split, set "
                "stratify_col to the target."
            )
    return hints


def _forward_default_flag(outputs: dict[str, Any]) -> list[str]:
    packet = outputs.get("out")
    target = _target(packet)
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
    if events < MIN_EVENTS:
        hints.append(f"Only {events} defaults (< {MIN_EVENTS}) -- a PD model on this will be unstable.")
    else:
        hints.append(f"{target}: {events} defaults in {flag.len() - nulls} rows ({events / (flag.len() - nulls):.2%}).")
    return hints


def _computed_target(column: str) -> Callable[[dict[str, Any]], list[str]]:
    def hints(outputs: dict[str, Any]) -> list[str]:
        packet = outputs.get("out")
        if not isinstance(packet, DataFramePacket) or column not in packet.data.columns or packet.data.height == 0:
            return []
        nulls = packet.data[column].null_count() / packet.data.height
        if nulls > MAX_NULL_SHARE:
            return [
                f"{nulls:.0%} of rows have a null {column} -- filter them out ({column} IS NOT NULL) before fitting."
            ]
        return []

    return hints


# ---- feature screens -------------------------------------------------------------------


def _iv_rows(rows: list[dict[str, Any]], keep_monotonic: bool) -> list[str]:
    if not rows:
        return []
    hints = []
    if "iv" in rows[0]:
        keep = [r["feature"] for r in rows if USELESS_IV <= (r.get("iv") or 0.0) < SUSPICIOUS_IV]
        drop = [r["feature"] for r in rows if (r.get("iv") or 0.0) < USELESS_IV]
        leak = [r["feature"] for r in rows if (r.get("iv") or 0.0) >= SUSPICIOUS_IV]
        if keep:
            hints.append(f"Candidate features (IV {USELESS_IV}-{SUSPICIOUS_IV}): {_names(keep)}.")
        if drop:
            hints.append(f"Leave out (IV < {USELESS_IV}, no signal): {_names(drop)}.")
        if leak:
            hints.append(
                f"IV >= {SUSPICIOUS_IV} usually means leakage: {_names(leak)}. ask_user before using any of them."
            )
    else:
        keep = [r["feature"] for r in rows if (r.get("r2_binned") or 0.0) >= WEAK_R2]
        drop = [r["feature"] for r in rows if (r.get("r2_binned") or 0.0) < WEAK_R2]
        if keep:
            hints.append(f"Candidate features (binned R^2 >= {WEAK_R2}): {_names(keep)}.")
        if drop:
            hints.append(f"Leave out (binned R^2 < {WEAK_R2}, no signal): {_names(drop)}.")
    if keep_monotonic:
        flat = [r["feature"] for r in rows if r.get("monotonic") is False and r["feature"] in keep]
        if flat:
            hints.append(f"Not monotonic in the target: {_names(flat)} -- prefer the monotonic ones, or check the bins.")
    return hints


def _binning(outputs: dict[str, Any]) -> list[str]:
    return _iv_rows(_rows(outputs.get("summary")), keep_monotonic=True)


def _iv_table(outputs: dict[str, Any]) -> list[str]:
    return _iv_rows(_metric(outputs).get("rows") or [], keep_monotonic=False)


def _correlation(outputs: dict[str, Any]) -> list[str]:
    metric = _metric(outputs)
    features, matrix = metric.get("features") or [], metric.get("correlation") or []
    pairs = [
        f"{features[i]}/{features[j]} ({matrix[i][j]:+.2f})"
        for i in range(len(features))
        for j in range(i + 1, len(features))
        if i < len(matrix) and j < len(matrix[i]) and _num(matrix[i][j]) and abs(matrix[i][j]) >= HIGH_CORRELATION
    ]
    vif = [v["feature"] for v in metric.get("vif") or [] if _num(v.get("vif")) and v["vif"] >= HIGH_VIF]
    hints = []
    if pairs:
        hints.append(f"Highly correlated (|r| >= {HIGH_CORRELATION}): {_names(pairs)} -- keep the stronger of each pair.")
    if vif:
        hints.append(f"VIF >= {HIGH_VIF:g}: {_names(vif)} -- largely redundant given the others; drop or merge.")
    if features and not hints:
        hints.append("No redundant features (no high correlations or VIFs).")
    return hints


def _characteristic_stability(outputs: dict[str, Any]) -> list[str]:
    rows = _rows(outputs.get("table"))
    unstable = [r["feature"] for r in rows if _num(r.get("psi")) and r["psi"] >= PSI_SHIFTED]
    monitor = [r["feature"] for r in rows if _num(r.get("psi")) and PSI_STABLE <= r["psi"] < PSI_SHIFTED]
    hints = []
    if unstable:
        hints.append(f"Unstable (PSI >= {PSI_SHIFTED}): {_names(unstable)} -- leave out, or ask_user before keeping.")
    if monitor:
        hints.append(f"Shifting (PSI {PSI_STABLE}-{PSI_SHIFTED}): {_names(monitor)} -- note them in the stage summary.")
    if rows and not hints:
        hints.append(f"Every feature is stable (PSI < {PSI_STABLE}).")
    return hints


def _target_trend(outputs: dict[str, Any]) -> list[str]:
    rows = [r for r in _rows(outputs.get("table")) if _num(r.get("n"))]
    if len(rows) < 2:
        return []
    hints = []
    counts = sorted(r["n"] for r in rows)
    median = counts[len(counts) // 2]
    if rows[-1]["n"] < IMMATURE_LAST_PERIOD * median:
        hints.append(
            f"The last period ({rows[-1]['period']}) has {rows[-1]['n']} rows against a typical {median} -- likely "
            "immature; consider leaving it out (time_split's oot_end)."
        )
    thin = [str(r["period"]) for r in rows if r["n"] < THIN_PERIOD]
    if thin:
        hints.append(f"Periods with < {THIN_PERIOD} rows: {_names(thin)} -- their target rates are noise.")
    return hints


# ---- fits ------------------------------------------------------------------------


def _coefficients(model: Any, woe_signs: bool) -> list[str]:
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
        [f for f, c in coefs.items() if f.endswith("_woe") and _num(c) and c > 0] if woe_signs else []
    )
    if wrong:
        hints.append(
            f"Positive coefficient on WoE feature(s) {_names(wrong)} -- the wrong sign, usually collinearity. "
            "Drop them (or the feature they overlap with) and refit."
        )
    weak = [f for f, s in stats.items() if (s.get("p_value") or 0.0) > MAX_P_VALUE and f not in wrong]
    if weak:
        hints.append(f"Not significant (p > {MAX_P_VALUE}): {_names(weak)} -- consider dropping them and refitting.")
    if coefs and not hints:
        woe = woe_signs and all(f.endswith("_woe") for f in coefs)
        hints.append(
            "Coefficients look sound: signs as expected and all significant." if woe else "All coefficients are significant."
        )
    return hints


def _logistic(outputs: dict[str, Any]) -> list[str]:
    return _coefficients(outputs.get("model"), woe_signs=True)


def _lgd_regression(outputs: dict[str, Any]) -> list[str]:
    return _coefficients(outputs.get("model"), woe_signs=False)


def _stepwise(outputs: dict[str, Any]) -> list[str]:
    metric = _metric(outputs)
    selected = metric.get("selected")
    if selected is None:
        return []
    if not selected:
        return ["Nothing passed p_enter -- revisit the candidate features (or loosen p_enter) before fitting."]
    hints = [f"Use these as the model's features: {_names(selected)}."]
    if metric.get("dropped"):
        hints.append(f"Dropped along the way: {_names(metric['dropped'])}.")
    if metric.get("target_type") == "binary":
        wrong = [
            c["feature"] for c in metric.get("coefficients") or [] if c["feature"].endswith("_woe") and c.get("sign") == "+"
        ]
        if wrong:
            hints.append(f"Wrong (positive) sign on {_names(wrong)} -- drop them before fitting the final model.")
    return hints


def _calibrate(outputs: dict[str, Any]) -> list[str]:
    cal = _metric(outputs, "model").get("calibration") or {}
    shift, before, after = cal.get("intercept_shift"), cal.get("mean_prediction_before"), cal.get("mean_prediction_after")
    if not (_num(shift) and _num(before) and _num(after)):
        return []
    if abs(shift) > BIG_INTERCEPT_SHIFT:
        return [
            f"A large shift (log-odds {shift:+.2f}, mean {before:.4f} -> {after:.4f}) -- check the central tendency "
            "is the right one, and mention it in the stage summary."
        ]
    return [f"Calibrated: mean prediction {before:.4f} -> {after:.4f}. Use this model for prediction from here on."]


def _master_scale(outputs: dict[str, Any]) -> list[str]:
    grades = _metric(outputs, "master_scale").get("grades") or []
    if not grades:
        return []
    hints = []
    rates = [g.get("observed_rate") for g in grades]
    if all(_num(r) for r in rates) and any(b < a for a, b in zip(rates, rates[1:])):
        hints.append("Observed rates aren't monotonic across grades -- refit with algorithm 'monotonic_default_rate'.")
    total = sum(g.get("n") or 0 for g in grades)
    thin = [str(g["grade"]) for g in grades if total and (g.get("n") or 0) / total < THIN_GRADE]
    if thin:
        hints.append(f"Thin grades ({_names(thin)}, < {THIN_GRADE:.0%} of the population) -- set min_grade_share.")
    if not hints:
        hints.append(f"{len(grades)} grades, monotonic and none too thin.")
    return hints


def _long_run_average(outputs: dict[str, Any]) -> list[str]:
    metric = _metric(outputs)
    hints = []
    periods = metric.get("n_periods")
    if _num(periods) and periods < SHORT_HISTORY:
        hints.append(
            f"Only {periods} period(s) -- likely short of a full cycle. Say so in the stage summary; a margin of "
            "conservatism may be needed."
        )
    dw, tw = metric.get("default_weighted"), metric.get("time_weighted")
    if _num(dw) and _num(tw) and dw and abs(tw - dw) / abs(dw) > LRA_GAP:
        hints.append(
            f"Default-weighted ({dw:.4f}) and time-weighted ({tw:.4f}) averages differ by more than {LRA_GAP:.0%} -- "
            "the choice matters; state which you use as the central tendency and why."
        )
    if not hints and _num(periods):
        hints.append(f"{periods} periods, and the default- and time-weighted averages agree.")
    return hints


# ---- discrimination --------------------------------------------------------------------


def _gini_value(gini: Any) -> list[str]:
    if not _num(gini):
        return []
    if gini < 0:
        return ["Gini is negative -- the score runs the wrong way (higher = safer). Check score_col is the predicted PD."]
    if gini >= SUSPICIOUS_GINI:
        return [f"Gini {gini:.2f} is unusually high for a credit model -- check the features for leakage before going on."]
    if gini < WEAK_GINI:
        return [f"Gini {gini:.2f} is weak -- revisit feature selection before validating further."]
    return [f"Gini {gini:.2f} is in the usual range."]


def _gini(outputs: dict[str, Any]) -> list[str]:
    return _gini_value(_metric(outputs).get("gini"))


def _roc(outputs: dict[str, Any]) -> list[str]:
    auc = _metric(outputs).get("auc")
    return _gini_value(2 * auc - 1) if _num(auc) else []


def _ks(outputs: dict[str, Any]) -> list[str]:
    ks = _metric(outputs).get("ks_statistic")
    if not _num(ks):
        return []
    if ks >= SUSPICIOUS_KS:
        return [f"KS {ks:.2f} is unusually high for a credit model -- check the features for leakage."]
    if ks < WEAK_KS:
        return [f"KS {ks:.2f} is weak -- the score separates goods and bads poorly."]
    return [f"KS {ks:.2f} is in the usual range."]


def _continuous_accuracy(outputs: dict[str, Any]) -> list[str]:
    m = _metric(outputs)
    hints = []
    if _num(m.get("r2")) and m["r2"] < 0:
        hints.append("R^2 is negative -- the model predicts worse than the mean. Revisit the features.")
    if _num(m.get("spearman")) and m["spearman"] < WEAK_SPEARMAN:
        hints.append(f"Spearman {m['spearman']:.2f} is weak -- the prediction barely ranks the outcome.")
    actual, bias = m.get("mean_actual"), m.get("bias")
    if _num(actual) and _num(bias) and actual and abs(bias) / abs(actual) > CALIBRATION_GAP:
        hints.append(
            f"Mean prediction is {abs(bias) / abs(actual):.0%} {'above' if bias > 0 else 'below'} the mean outcome -- "
            "calibrate the model."
        )
    if m and not hints and _num(m.get("spearman")):
        hints.append(f"Accuracy looks reasonable (Spearman {m['spearman']:.2f}, bias {m.get('bias', 0):+.4f}).")
    return hints


def _compare(outputs: dict[str, Any]) -> list[str]:
    rows = _rows(outputs.get("table"))
    if not rows:
        return []
    hints = []
    base = rows[0]
    for key, limit in (("gini", OVERFIT_GINI_DROP), ("spearman", OVERFIT_SPEARMAN_DROP)):
        for r in rows[1:]:
            if _num(base.get(key)) and _num(r.get(key)) and base[key] - r[key] > limit:
                hints.append(
                    f"{key.title()} falls {base[key] - r[key]:.2f} from {base['sample']} ({base[key]:.2f}) to "
                    f"{r['sample']} ({r[key]:.2f}) -- likely overfitting: try fewer features or stronger regularisation."
                )
    for r in rows:
        actual, predicted = r.get("mean_actual"), r.get("mean_predicted")
        if _num(actual) and _num(predicted) and actual:
            gap = (predicted - actual) / abs(actual)
            if abs(gap) > CALIBRATION_GAP:
                hints.append(
                    f"On {r['sample']} the mean prediction ({predicted:.4f}) is {abs(gap):.0%} "
                    f"{'above' if gap > 0 else 'below'} the mean outcome ({actual:.4f}) -- calibrate the model."
                )
    if not hints:
        hints.append("Performance holds across the samples and predictions match outcomes on average.")
    return hints


# ---- calibration and stability -----------------------------------------------------------


def _hosmer_lemeshow(outputs: dict[str, Any]) -> list[str]:
    p = _metric(outputs).get("p_value")
    if not _num(p):
        return []
    if p < CALIBRATION_P:
        return [
            f"Hosmer-Lemeshow p = {p:.3g} < {CALIBRATION_P}: predicted PDs don't match observed rates -- calibrate "
            "(calibrate_model) or explain why not in the stage summary."
        ]
    return [f"Hosmer-Lemeshow p = {p:.3g}: calibration is acceptable."]


def _bucketed_calibration(outputs: dict[str, Any]) -> list[str]:
    buckets = _metric(outputs).get("buckets") or []
    off = [
        b["bucket"]
        for b in buckets
        if _num(b.get("observed_mean"))
        and _num(b.get("predicted_mean"))
        and b["predicted_mean"]
        and abs(b["observed_mean"] - b["predicted_mean"]) / abs(b["predicted_mean"]) > CALIBRATION_GAP
    ]
    if off:
        return [
            f"{len(off)} of {len(buckets)} buckets miss their prediction by more than {CALIBRATION_GAP:.0%}: "
            f"{_names(off)} -- calibrate the model, or say why in the stage summary."
        ]
    return ["Every bucket's observed mean is close to its prediction."] if buckets else []


def _psi(outputs: dict[str, Any]) -> list[str]:
    psi = _metric(outputs).get("psi")
    if not _num(psi):
        return []
    if psi >= PSI_SHIFTED:
        return [f"PSI {psi:.3f} >= {PSI_SHIFTED}: a significant shift -- ask_user before relying on it."]
    if psi >= PSI_STABLE:
        return [f"PSI {psi:.3f}: a moderate shift -- worth noting in the stage summary."]
    return [f"PSI {psi:.3f} < {PSI_STABLE}: stable."]


def _rating(outputs: dict[str, Any]) -> list[str]:
    metric = _metric(outputs)
    if "monotonic" not in metric:
        return []
    if metric["monotonic"]:
        return ["Default rates rise grade by grade, as they should."]
    return [
        "Default rates aren't monotonic across grades -- refit the master scale with algorithm "
        "'monotonic_default_rate' (or fewer grades)."
    ]


def _grade_backtest(outputs: dict[str, Any]) -> list[str]:
    metric = _metric(outputs)
    grades = metric.get("grades") or []
    if not grades:
        return []
    hints = []
    red = [str(g["grade"]) for g in grades if g.get("traffic_light") == "red"]
    if red:
        hints.append(
            f"Red grades ({_names(red)}): their PD is too low for the observed default rate -- recalibrate, or add a "
            "margin of conservatism."
        )
    if (metric.get("portfolio") or {}).get("traffic_light") == "red":
        hints.append("The portfolio-level PD is too low -- recalibrate before anything else.")
    if metric.get("monotonic") is False:
        hints.append("Observed rates aren't monotonic across grades -- refit the master scale ('monotonic_default_rate').")
    hhi = metric.get("herfindahl")
    if _num(hhi) and hhi > 2 / len(grades):
        hints.append(f"The population is concentrated in few grades (Herfindahl {hhi:.2f}) -- consider more even grades.")
    if not hints:
        hints.append("The back-test passes: no red grades, monotonic, evenly spread.")
    return hints


# ---- stochastic ------------------------------------------------------------------


def _fit_distribution(outputs: dict[str, Any]) -> list[str]:
    dist = _metric(outputs, "distribution")
    stats = dist.get("fit_stats") or {}
    p = stats.get("ks_p")
    if not dist.get("family") or not _num(p):
        return []
    if p < GOOD_FIT_P:
        return [
            f"The best family ({dist['family']}) still fits poorly (KS p = {p:.3g}) -- try other families, or "
            "spliced_tail for a heavy tail."
        ]
    return [f"{dist['family']} fits best (lowest AIC) and fits well (KS p = {p:.3g})."]


def _validate_proxy(outputs: dict[str, Any]) -> list[str]:
    r2 = _metric(outputs, "diagnostics").get("out_of_sample_r2")
    if not _num(r2):
        return []
    if r2 < PROXY_MIN_R2:
        return [
            f"Out-of-sample R^2 {r2:.3f} < {PROXY_MIN_R2} -- the proxy isn't accurate enough: fit on more scenarios "
            "or a richer basis."
        ]
    return [f"Out-of-sample R^2 {r2:.3f}: the proxy is accurate enough to use."]


HINTS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    # data
    "data_profile": _profile,
    "apply_exclusions": _exclusions,
    "data_quality_rules": _quality_rules,
    "train_test_split": _split,
    "time_split": _split,
    "forward_default_flag": _forward_default_flag,
    "compute_lgd": _computed_target("lgd"),
    "compute_ccf": _computed_target("ccf"),
    # feature screens
    "fit_binning": _binning,
    "iv_table": _iv_table,
    "correlation_matrix": _correlation,
    "characteristic_stability": _characteristic_stability,
    "target_trend": _target_trend,
    # fits
    "logistic_regression": _logistic,
    "lgd_regression": _lgd_regression,
    "stepwise_selection": _stepwise,
    "calibrate_model": _calibrate,
    "fit_master_scale": _master_scale,
    "long_run_average": _long_run_average,
    # discrimination
    "auc_gini": _gini,
    "roc_curve": _roc,
    "ks_test": _ks,
    "continuous_accuracy": _continuous_accuracy,
    "compare_samples": _compare,
    # calibration and stability
    "calibration_test": _hosmer_lemeshow,
    "bucketed_calibration": _bucketed_calibration,
    "psi_test": _psi,
    "rating_summary": _rating,
    "grade_backtest": _grade_backtest,
    # stochastic
    "fit_distribution": _fit_distribution,
    "validate_proxy": _validate_proxy,
}


def decision_hints(category: str, outputs: dict[str, Any]) -> list[str]:
    """Hints for a block of `category` from its raw output values
    ({port: value}); empty when the category has none."""
    fn = HINTS.get(category)
    return fn(outputs) if fn else []
