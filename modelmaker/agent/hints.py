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

Each block's own hint function sits next to it in its block module
(BlockSpec.hints) and reads its thresholds and helpers from here, as
`h.MIN_EVENTS`, `h.names(...)` and so on.

Blocks with nothing to judge (reads, joins, transforms, plots, fitted
artifacts used only by wiring) get no hints; neither does glm_fit, whose
artifact carries no statistics to judge it by."""

from __future__ import annotations

from typing import Any

from ..blocks.base import BLOCK_REGISTRY
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


def names(items: list[str]) -> str:
    shown = ", ".join(str(i) for i in items[:MAX_NAMES])
    return shown + (f" (+{len(items) - MAX_NAMES} more)" if len(items) > MAX_NAMES else "")


def rows(value: Any) -> list[dict[str, Any]]:
    return value.data.to_dicts() if isinstance(value, DataFramePacket) else []


def num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def metric(outputs: dict[str, Any], port: str = "metric") -> dict[str, Any]:
    value = outputs.get(port)
    return value if isinstance(value, dict) else {}


def target_column(packet: Any) -> str | None:
    if not isinstance(packet, DataFramePacket):
        return None
    return next(
        (n for n, m in packet.schema_meta.items() if m.role == ColumnRole.TARGET and n in packet.data.columns), None
    )


def iv_rows(rows: list[dict[str, Any]], keep_monotonic: bool) -> list[str]:
    """fit_binning's summary / iv_table's rows: which features to keep,
    drop or question."""
    if not rows:
        return []
    hints = []
    if "iv" in rows[0]:
        keep = [r["feature"] for r in rows if USELESS_IV <= (r.get("iv") or 0.0) < SUSPICIOUS_IV]
        drop = [r["feature"] for r in rows if (r.get("iv") or 0.0) < USELESS_IV]
        leak = [r["feature"] for r in rows if (r.get("iv") or 0.0) >= SUSPICIOUS_IV]
        if keep:
            hints.append(f"Candidate features (IV {USELESS_IV}-{SUSPICIOUS_IV}): {names(keep)}.")
        if drop:
            hints.append(f"Leave out (IV < {USELESS_IV}, no signal): {names(drop)}.")
        if leak:
            hints.append(
                f"IV >= {SUSPICIOUS_IV} usually means leakage: {names(leak)}. ask_user before using any of them."
            )
    else:
        keep = [r["feature"] for r in rows if (r.get("r2_binned") or 0.0) >= WEAK_R2]
        drop = [r["feature"] for r in rows if (r.get("r2_binned") or 0.0) < WEAK_R2]
        if keep:
            hints.append(f"Candidate features (binned R^2 >= {WEAK_R2}): {names(keep)}.")
        if drop:
            hints.append(f"Leave out (binned R^2 < {WEAK_R2}, no signal): {names(drop)}.")
    if keep_monotonic:
        flat = [r["feature"] for r in rows if r.get("monotonic") is False and r["feature"] in keep]
        if flat:
            hints.append(f"Not monotonic in the target: {names(flat)} -- prefer the monotonic ones, or check the bins.")
    return hints


def decision_hints(category: str, outputs: dict[str, Any]) -> list[str]:
    """Hints for a block of `category` from its raw output values
    ({port: value}); empty when the block declares none (BlockSpec.hints)."""
    from .catalogue import ensure_blocks_registered

    ensure_blocks_registered()
    spec = BLOCK_REGISTRY.get(category)
    return spec.hints(outputs) if spec is not None and spec.hints is not None else []
