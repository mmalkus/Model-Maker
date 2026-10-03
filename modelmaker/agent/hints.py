"""Decision hints (BuildOptions.decision_hints): short, rule-based "what
this result means and what to do next" notes added to run_to's report
when a block that the build has to judge ran green -- binning, a fit,
discrimination, stability, calibration checks.

They move judgement out of the model and into code, for small models
that call tools reliably but read a statistics table poorly. A hint is a
suggestion, not a guard: the model may depart from one with a reason
(note_deviation). The thresholds are the usual credit-risk rules of
thumb; they are the modelling policy a hinted build follows, so they're
kept together here, named, to be read and changed in one place."""

from __future__ import annotations

from typing import Any, Callable

from ..packet import DataFramePacket

# fit_binning's IV bands (see binning._iv_band): below this a feature
# carries no signal; at or above SUSPICIOUS_IV it usually leaks the target.
USELESS_IV = 0.02
SUSPICIOUS_IV = 0.5
# Continuous target: a binned feature explaining less than this share of
# the target's variance isn't worth a model slot.
WEAK_R2 = 0.005
# A non-intercept coefficient this unlikely to be non-zero.
MAX_P_VALUE = 0.05
# Gini (binary target) on any one sample.
WEAK_GINI = 0.2
SUSPICIOUS_GINI = 0.85
# Train -> test/OOT Gini drop (absolute) that suggests overfitting.
OVERFIT_GINI_DROP = 0.05
# Mean prediction vs. mean outcome on a sample, relative difference.
CALIBRATION_GAP = 0.2
# PSI: below STABLE is stable, from SHIFTED a significant shift.
PSI_STABLE = 0.1
PSI_SHIFTED = 0.25

MAX_NAMES = 12  # features named per hint, at most


def _names(items: list[str]) -> str:
    shown = ", ".join(items[:MAX_NAMES])
    return shown + (f" (+{len(items) - MAX_NAMES} more)" if len(items) > MAX_NAMES else "")


def _rows(value: Any) -> list[dict[str, Any]]:
    return value.data.to_dicts() if isinstance(value, DataFramePacket) else []


def _binning(outputs: dict[str, Any]) -> list[str]:
    rows = _rows(outputs.get("summary"))
    if not rows:
        return []
    if "iv" in rows[0]:
        keep = [r["feature"] for r in rows if USELESS_IV <= (r.get("iv") or 0.0) < SUSPICIOUS_IV]
        drop = [r["feature"] for r in rows if (r.get("iv") or 0.0) < USELESS_IV]
        leak = [r["feature"] for r in rows if (r.get("iv") or 0.0) >= SUSPICIOUS_IV]
        hints = []
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
        hints = []
        if keep:
            hints.append(f"Candidate features (binned R^2 >= {WEAK_R2}): {_names(keep)}.")
        if drop:
            hints.append(f"Leave out (binned R^2 < {WEAK_R2}, no signal): {_names(drop)}.")
    flat = [r["feature"] for r in rows if r.get("monotonic") is False and r["feature"] in keep]
    if flat:
        hints.append(f"Not monotonic in the target: {_names(flat)} -- prefer the monotonic ones, or check the bins.")
    return hints


def _coefficients(outputs: dict[str, Any]) -> list[str]:
    model = outputs.get("model")
    if not isinstance(model, dict):
        return []
    stats = {k: v for k, v in (model.get("statistics") or {}).items() if k != "intercept"}
    coefs = model.get("coefficients") or {k: v.get("estimate") for k, v in stats.items()}
    hints = []
    # WoE is higher where the outcome is rarer, so every WoE coefficient
    # of a PD model should be negative (logistic_regression's docstring).
    wrong = [f for f, c in coefs.items() if f.endswith("_woe") and isinstance(c, (int, float)) and c > 0]
    if wrong:
        hints.append(
            f"Positive coefficient on WoE feature(s) {_names(wrong)} -- the wrong sign, usually collinearity. "
            "Drop them (or the feature they overlap with) and refit."
        )
    weak = [f for f, s in stats.items() if (s.get("p_value") or 0.0) > MAX_P_VALUE and f not in wrong]
    if weak:
        hints.append(f"Not significant (p > {MAX_P_VALUE}): {_names(weak)} -- consider dropping them and refitting.")
    if coefs and not hints:
        woe = all(f.endswith("_woe") for f in coefs)
        hints.append(
            "Coefficients look sound: signs as expected and all significant." if woe else "All coefficients are significant."
        )
    return hints


def _gini(outputs: dict[str, Any]) -> list[str]:
    metric = outputs.get("metric")
    gini = metric.get("gini") if isinstance(metric, dict) else None
    if not isinstance(gini, (int, float)):
        return []
    if gini < 0:
        return ["Gini is negative -- the score runs the wrong way (higher = safer). Check score_col is the predicted PD."]
    if gini >= SUSPICIOUS_GINI:
        return [f"Gini {gini:.2f} is unusually high for a credit model -- check the features for leakage before going on."]
    if gini < WEAK_GINI:
        return [f"Gini {gini:.2f} is weak -- revisit feature selection before validating further."]
    return [f"Gini {gini:.2f} is in the usual range."]


def _compare(outputs: dict[str, Any]) -> list[str]:
    rows = _rows(outputs.get("table"))
    if not rows:
        return []
    hints = []
    base = rows[0]
    for r in rows[1:]:
        if isinstance(base.get("gini"), (int, float)) and isinstance(r.get("gini"), (int, float)):
            drop = base["gini"] - r["gini"]
            if drop > OVERFIT_GINI_DROP:
                hints.append(
                    f"Gini falls {drop:.2f} from {base['sample']} ({base['gini']:.2f}) to {r['sample']} "
                    f"({r['gini']:.2f}) -- likely overfitting: try fewer features or stronger regularisation."
                )
    for r in rows:
        actual, predicted = r.get("mean_actual"), r.get("mean_predicted")
        if isinstance(actual, (int, float)) and isinstance(predicted, (int, float)) and actual:
            gap = (predicted - actual) / abs(actual)
            if abs(gap) > CALIBRATION_GAP:
                hints.append(
                    f"On {r['sample']} the mean prediction ({predicted:.4f}) is {abs(gap):.0%} "
                    f"{'above' if gap > 0 else 'below'} the mean outcome ({actual:.4f}) -- calibrate the model."
                )
    if not hints:
        hints.append("Performance holds across the samples and predictions match outcomes on average.")
    return hints


def _psi(outputs: dict[str, Any]) -> list[str]:
    metric = outputs.get("metric")
    psi = metric.get("psi") if isinstance(metric, dict) else None
    if not isinstance(psi, (int, float)):
        return []
    if psi >= PSI_SHIFTED:
        return [f"PSI {psi:.3f} >= {PSI_SHIFTED}: a significant shift -- ask_user before relying on it."]
    if psi >= PSI_STABLE:
        return [f"PSI {psi:.3f}: a moderate shift -- worth noting in the stage summary."]
    return [f"PSI {psi:.3f} < {PSI_STABLE}: stable."]


def _rating(outputs: dict[str, Any]) -> list[str]:
    metric = outputs.get("metric")
    if not isinstance(metric, dict) or "monotonic" not in metric:
        return []
    if metric["monotonic"]:
        return ["Default rates rise grade by grade, as they should."]
    return ["Default rates aren't monotonic across grades -- refit the master scale with fewer grades."]


HINTS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "fit_binning": _binning,
    "logistic_regression": _coefficients,
    "auc_gini": _gini,
    "compare_samples": _compare,
    "psi_test": _psi,
    "rating_summary": _rating,
}


def decision_hints(category: str, outputs: dict[str, Any]) -> list[str]:
    """Hints for a block of `category` from its raw output values
    ({port: value}); empty when the category has none."""
    fn = HINTS.get(category)
    return fn(outputs) if fn else []
