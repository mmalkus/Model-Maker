"""The data the plan is being made for, and the facts read off it. Facts
are what decision states are built from: counts, shares and dtypes -- never
rows, and no min/max of text or id columns (the AI guardrail)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .. import hints as h


@dataclass
class Workspace:
    df: pl.DataFrame
    roles: dict[str, str]  # column -> "id" | "date" | "target" | "feature" | "excluded"
    built: list[str] = field(default_factory=list)  # menu option keys run, in order
    resolved: set[str] = field(default_factory=set)  # options run or declined
    artifacts: dict[str, Any] = field(default_factory=dict)  # block outputs kept aside
    model_type: str | None = None

    def col(self, role: str) -> str | None:
        return next(
            (c for c, r in self.roles.items() if r == role and c in self.df.columns),
            None,
        )

    def cols(self, *roles: str) -> list[str]:
        return [c for c in self.df.columns if self.roles.get(c, "unassigned") in roles]

    # ---- dates ----

    def dates(self, col: str | None = None) -> pl.Series | None:
        col = col or self.col("date")
        if col is None:
            return None
        s = self.df[col]
        if s.dtype == pl.Utf8:
            return s.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
        if s.dtype.is_integer():  # YYYYMM periods
            return pl.Series([None if v is None else f"{v // 100:04d}-{v % 100:02d}-01" for v in s]).str.to_date()
        return s.cast(pl.Date)


def _looks_like_date(s: pl.Series) -> bool:
    if s.dtype in (pl.Date, pl.Datetime):
        return True
    if s.dtype != pl.Utf8:
        return False
    sample = s.drop_nulls().head(200)
    return sample.len() > 0 and sample.str.contains(r"^\d{4}-\d{2}-\d{2}").all()


def _binary(s: pl.Series) -> bool:
    return s.dtype.is_numeric() and set(s.drop_nulls().unique().to_list()) <= {0, 1}


def history_months(ws: Workspace) -> int | None:
    d = ws.dates()
    if d is None or d.drop_nulls().len() == 0:
        return None
    lo, hi = d.min(), d.max()
    return (hi.year - lo.year) * 12 + hi.month - lo.month + 1


def dataset_facts(ws: Workspace) -> dict[str, Any]:
    df = ws.df
    target, id_col, date_col = ws.col("target"), ws.col("id"), ws.col("date")
    facts: dict[str, Any] = {"rows": df.height}
    facts["panel"] = bool(id_col and df[id_col].n_unique() < df.height)
    facts["has_date"] = date_col is not None
    if date_col:
        facts["history_months"] = history_months(ws)
    if target:
        t = df[target]
        facts["target"] = "binary" if _binary(t) else "continuous"
        if facts["target"] == "binary":
            facts["events"] = int(t.sum() or 0)
            facts["event_rate"] = float(t.mean() or 0.0)
        facts["target_nulls"] = int(t.null_count())
    else:
        facts["target"] = "missing"
        facts["status_candidates"] = status_candidates(ws)
    return facts


def status_candidates(ws: Workspace) -> list[str]:
    """0/1 columns that could be a monthly default status (panel data with
    no target yet). Excluded ones count too: a screen may rightly have
    excluded the status as the outcome itself."""
    return [c for c in ws.cols("unassigned", "feature", "excluded") if _binary(ws.df[c])]


def column_facts(ws: Workspace, col: str) -> dict[str, Any]:
    s = ws.df[col]
    n = max(s.len(), 1)
    non_null = s.drop_nulls()
    facts: dict[str, Any] = {
        "column": col,
        "dtype": str(s.dtype),
        "fill_rate": non_null.len() / n,
        "distinct_share": non_null.n_unique() / max(non_null.len(), 1),
    }
    if non_null.len():
        facts["top_value_share"] = int(non_null.value_counts(sort=True)["count"][0]) / non_null.len()
    if _looks_like_date(s):
        facts["date_like"] = True
    target = ws.col("target")
    if target and ws.df[target].null_count() < ws.df.height:
        # Filled only for defaulters (or only for non-defaulters) means the
        # column is recorded after the outcome.
        t = ws.df[target].cast(pl.Float64)
        filled = s.is_not_null()
        events = t == 1
        if events.sum():
            facts["fill_events"] = float((filled & events).sum() / events.sum())
        if (~events).sum():
            facts["fill_non_events"] = float((filled & ~events).sum() / (~events).sum())
    id_like = s.dtype.is_integer() and facts["distinct_share"] >= h.ID_LIKE_DISTINCT
    if s.dtype.is_numeric() and non_null.len() and not id_like:
        facts["min"], facts["max"] = float(non_null.min()), float(non_null.max())
        if facts["min"] < 0:
            facts["negative_share"] = float((non_null < 0).sum() / non_null.len())
    return facts
