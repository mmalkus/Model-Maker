"""Column profile: a pure-code classification of a dataframe's columns, so a
model reads "reference_date 2022-07-01..2025-06-30" or "product_type (2
levels)" instead of working it out from raw statistics.

Only aggregates, like every other view the agent gets (see llm/redact.py):
no category labels, no text min/max, nothing of an id column. Dates are the
exception -- their range is shown, text dates read through parse_dates."""

from __future__ import annotations

from typing import Any

import polars as pl

from ..blocks.library import parse_dates
from ..packet import ColumnRole, DataFramePacket

ID_LIKE_SHARE = 0.95  # distinct / non-null at or above this: id-like
MAX_CATEGORIES = 30  # at most this many levels: categorical (one_hot_encode's cap)
SEGMENT_MAX_LEVELS, SEGMENT_MIN_SHARE = 10, 0.05  # few, sizeable levels: could segment
DISCRETE_MAX_VALUES = 12  # an integer column with at most this many values: discrete


def _date_range(values: pl.Series) -> dict[str, str]:
    # p75: where an out-of-time cutoff keeping ~75% of rows for development falls.
    s = values.sort()
    return {"min": str(s[0]), "max": str(s[-1]), "p75": str(s[int(0.75 * (s.len() - 1))])}


def _num(x: Any) -> Any:
    return round(x, 4 if abs(x) < 10 else 2) if isinstance(x, float) else x


def profile_column(name: str, s: pl.Series, role: str) -> dict[str, Any]:
    values = s.drop_nulls()
    p: dict[str, Any] = {"name": name, "role": role}
    if s.null_count():
        p["null_share"] = round(s.null_count() / s.len(), 3)
    if role in (ColumnRole.EXCLUDED.value, ColumnRole.ID.value):
        return {**p, "kind": role}
    distinct = values.n_unique() if values.len() else 0
    if distinct <= 1:
        return {**p, "kind": "empty" if distinct == 0 else "constant"}
    if s.dtype in (pl.Date, pl.Datetime) or isinstance(s.dtype, pl.Datetime):
        return {**p, "kind": "date", **_date_range(values)}
    if s.dtype == pl.String:
        parsed = parse_dates(pl.DataFrame({name: values}))[name]
        if parsed.dtype != pl.String:
            return {**p, "kind": "text date", **_date_range(parsed)}
        if distinct > MAX_CATEGORIES:
            return {**p, "kind": "id-like text" if distinct / values.len() >= ID_LIKE_SHARE else "text", "distinct": distinct}
        out = {**p, "kind": "binary" if distinct == 2 else "categorical", "levels": distinct}
        if distinct <= SEGMENT_MAX_LEVELS and values.value_counts()["count"].min() / values.len() >= SEGMENT_MIN_SHARE:
            out["segment_candidate"] = True
        return out
    if s.dtype == pl.Boolean:
        return {**p, "kind": "binary", "levels": 2}
    if not s.dtype.is_numeric():
        return {**p, "kind": "other", "distinct": distinct}
    lo, hi = values.min(), values.max()
    out = {**p, "min": _num(lo), "max": _num(hi), "distinct": distinct}
    if distinct == 2:
        out.update(kind="binary", levels=2, **({"rate_of_1": round(float(values.mean()), 4)} if {lo, hi} == {0, 1} else {}))
    elif s.dtype.is_integer() and values.len() > 50 and distinct / values.len() >= ID_LIKE_SHARE:
        out["kind"] = "id-like number"
    elif s.dtype.is_integer() and distinct <= DISCRETE_MAX_VALUES:
        out["kind"] = "discrete"
    else:
        out["kind"] = "numeric"
    return out


def profile(packet: DataFramePacket) -> list[dict[str, Any]]:
    df = packet.data
    return [
        profile_column(name, df[name], getattr(meta.role, "value", str(meta.role)))
        for name, meta in packet.schema_meta.items()
        if name in df.columns
    ]


def has_text_dates(cols: list[dict[str, Any]]) -> bool:
    return any(c["kind"] == "text date" for c in cols)


_GROUPS = {
    "date": "dates", "text date": "dates stored as text (parse_dates converts them)", "numeric": "numeric",
    "discrete": "discrete (few integer values)", "binary": "binary", "categorical": "categorical",
    "excluded": "excluded (never a feature)", "id": "id", "empty": "empty (no values)", "constant": "constant (one value)",
}


def describe(cols: list[dict[str, Any]], rows: int) -> str:
    """The profile as a few prompt lines, grouped by kind of column."""
    groups: dict[str, list[str]] = {}
    for c in cols:
        kind, text = c["kind"], c["name"]
        if "min" in c:
            text += f" {c['min']}..{c['max']}"
        if "p75" in c:
            text += f" (75% of rows before {c['p75']})"
        notes = [f"{c['levels']} levels" if kind in ("binary", "categorical") and "levels" in c else "",
                 f"{c['distinct']} values" if kind == "discrete" else "",
                 f"rate of 1 = {c['rate_of_1']:.2%}" if "rate_of_1" in c else "",
                 "could segment" if c.get("segment_candidate") else "",
                 f"{c['null_share']:.0%} missing" if c.get("null_share") else ""]
        notes = ", ".join(n for n in notes if n)
        if c["role"] == ColumnRole.TARGET.value:
            group, text = "target", f"{c['name']} ({kind}, {notes})"
        elif c["role"] == ColumnRole.DATE.value:
            group = "observation date (date_col params fill from it)"
        else:
            group = _GROUPS.get(kind, kind)
            text = c["name"] if kind in ("excluded", "id", "empty", "constant") else text + (f" ({notes})" if notes else "")
        groups.setdefault(group, []).append(text)
    order = ["target", "observation date (date_col params fill from it)", "id"] + list(_GROUPS.values())
    keys = sorted(groups, key=lambda k: order.index(k) if k in order else len(order))
    return f"{rows:,} rows.\n" + "\n".join(f"- {k}: " + "; ".join(groups[k]) for k in keys)
