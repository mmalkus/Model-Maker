"""Data quality & profiling blocks -- the checks a model validator asks for
before looking at a single coefficient (model-developer-review.md §1.1),
and which the tool had none of until now. Same contract as the rest of the
block library: each `fn` is a plain function over pl.DataFrame / literal
params, no DataFramePacket/ColumnMeta, so a --with-metadata=off compile
emits it verbatim; any helper a block needs is defined *inside* that
block's own fn, since the compiler inlines only inspect.getsource(fn)
itself (see compiler.py), not sibling module-level helpers.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from ..packet import ColumnMeta, ColumnRole
from .base import BlockSpec, PortSpec, register_block


def data_profile(df: pl.DataFrame, features: list[str] | None = None) -> dict:
    """Per-column profile -- fill rate, distinct count, dominant-value
    share, and (numeric columns) percentiles. `DataFramePacket.compute_summary`
    already computes a lighter version of this on demand for the UI's
    preview panel; this promotes it to a block whose output is a table you
    can wire onward, e.g. straight into a model documentation pack."""
    cols = features or df.columns
    n = df.height
    rows = []
    for name in cols:
        s = df[name]
        null_count = int(s.null_count())
        non_null = n - null_count
        n_unique = int(s.n_unique())
        row: dict = {
            "column": name,
            "dtype": str(s.dtype),
            "count": n,
            "null_count": null_count,
            "fill_rate": (non_null / n) if n else None,
            "n_unique": n_unique,
            "distinct_ratio": (n_unique / non_null) if non_null else None,
        }
        if non_null:
            top_value, top_count = s.drop_nulls().value_counts(sort=True).row(0)
            row["dominant_value"] = top_value
            row["dominant_value_share"] = top_count / non_null
        else:
            row["dominant_value"] = None
            row["dominant_value_share"] = None
        if s.dtype.is_numeric() and non_null:
            row["min"] = float(s.min())
            row["max"] = float(s.max())
            row["mean"] = float(s.mean())
            row["std"] = float(s.std()) if non_null > 1 else None
            row["percentiles"] = {
                f"p{int(q * 100)}": float(s.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            }
        else:
            row["min"] = s.min()
            row["max"] = s.max()
            row["mean"] = None
            row["std"] = None
            row["percentiles"] = None
        rows.append(row)
    return {"kind": "data_profile", "row_count": n, "columns": rows}


register_block(
    BlockSpec(
        category="data_profile",
        block_type="output",
        group="quality",
        display_name="Data profile",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=data_profile,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def apply_exclusions(df: pl.DataFrame, rules: list[dict]) -> tuple[pl.DataFrame, dict]:
    """Applies a named, ordered list of exclusion rules and keeps every row
    that survives all of them, while recording the population waterfall --
    starting population, each rule's own drop, and the final modelling
    population -- the single most-requested table in a model document.
    Today's plain Filter block drops rows with no record of how many or
    why; this is the dedicated alternative for when that trail matters.

    Each rule is `{"name": str, "expr": str}`, `expr` a SQL boolean
    expression identifying rows to KEEP (same convention as Filter).
    Rules apply in order against whatever survived the rules before it, so
    each step's count is conditional on every earlier exclusion, not
    independent of it."""
    starting = df.height
    steps = [{"step": "starting population", "rule": None, "dropped": 0, "remaining": starting}]
    kept = df
    for rule in rules:
        before = kept.height
        kept = kept.filter(pl.sql_expr(rule["expr"]))
        after = kept.height
        steps.append({"step": rule["name"], "rule": rule["expr"], "dropped": before - after, "remaining": after})
    summary = {
        "kind": "exclusion_waterfall",
        "starting_population": starting,
        "final_population": kept.height,
        "total_dropped": starting - kept.height,
        "steps": steps,
    }
    return kept, summary


def _apply_exclusions_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()  # "summary" is a dict, not a DataFrame -- filtered out before this runs
    return {"out": {name: in_meta[name] for name in df.columns if name in in_meta}}


register_block(
    BlockSpec(
        category="apply_exclusions",
        block_type="standard",
        group="quality",
        display_name="Apply exclusions",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out"), PortSpec("summary", type="scalar_metric")],
        fn=apply_exclusions,
        metadata_transform=_apply_exclusions_meta,
    )
)


def data_quality_rules(df: pl.DataFrame, rules: list[dict]) -> dict:
    """Runs a declarative list of assertions and reports a pass/fail table
    -- the kind of check a validator expects to see run every time, not
    eyeballed once during development. Each rule is a dict:

        {"name": str, "rule": <kind>, "severity": "error" | "warning" = "error", ...}

    `<kind>` is one of:
      - "not_null" (needs "column") -- flags null values.
      - "unique" (needs "column" or "columns") -- flags duplicate values
        (or duplicate combinations, for "columns") -- the key-integrity
        check for e.g. duplicate account-months.
      - "in_set" (needs "column", "values") -- flags non-null values not
        in the given set.
      - "between" (needs "column", optional "min"/"max") -- flags non-null
        values outside [min, max]; the hook for a domain rule like "LGD
        must be in [0, 1]" or "exposure >= 0".
      - "row_count_between" (optional "min"/"max", no "column") -- checks
        df.height instead of a column.

    Never raises on a breach -- a failing rule is a modelling fact to
    review, not a crash. Read the top-level "passed" (true only if every
    error-severity rule passed) to decide whether to halt a pipeline on
    it, e.g. by wiring a downstream block that checks the result."""
    results = []
    for rule in rules:
        kind = rule["rule"]
        severity = rule.get("severity", "error")
        if kind == "row_count_between":
            n = df.height
            lo, hi = rule.get("min"), rule.get("max")
            violations = 0 if (lo is None or n >= lo) and (hi is None or n <= hi) else 1
            detail = f"row_count={n}"
        elif kind == "unique":
            cols = rule.get("columns") or [rule["column"]]
            violations = int(df.height - df.select(cols).n_unique())
            detail = f"columns={cols}, duplicate_rows={violations}"
        else:
            col = rule["column"]
            s = df[col]
            non_null = s.drop_nulls()
            if kind == "not_null":
                violations = int(s.null_count())
            elif kind == "in_set":
                allowed = list(rule["values"])
                violations = int((~non_null.is_in(allowed)).sum()) if non_null.len() else 0
            elif kind == "between":
                lo, hi = rule.get("min"), rule.get("max")
                below = int((non_null < lo).sum()) if lo is not None else 0
                above = int((non_null > hi).sum()) if hi is not None else 0
                violations = below + above
            else:
                raise ValueError(f"unknown rule kind: {kind!r}")
            detail = f"column={col}"
        results.append(
            {
                "name": rule.get("name", kind),
                "rule": kind,
                "severity": severity,
                "violations": violations,
                "passed": violations == 0,
                "detail": detail,
            }
        )
    return {
        "kind": "data_quality_rules",
        "row_count": df.height,
        "passed": all(r["passed"] for r in results if r["severity"] == "error"),
        "results": results,
    }


register_block(
    BlockSpec(
        category="data_quality_rules",
        block_type="output",
        group="quality",
        display_name="Data quality rules",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=data_quality_rules,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def missing_value_treatment(df: pl.DataFrame, strategies: dict[str, dict]) -> pl.DataFrame:
    """Fills or drops missing values per column under an explicit, named
    strategy -- absent from the tool until now, even though how missing
    values were treated is itself a documented modelling assumption.
    `strategies` maps a column name to
    `{"method": "constant" | "mean" | "median" | "mode" | "flag" | "drop_rows",
    "value": ..., "add_indicator": bool}`:

      - constant: fills nulls with "value".
      - mean / median: fills nulls with the column's own mean/median
        (numeric columns only).
      - mode: fills nulls with the column's most frequent non-null value.
      - flag: leaves values untouched -- a deliberate no-fill decision
        (e.g. a downstream model that handles missingness natively), still
        recorded as such in the output metadata (see
        _missing_value_treatment_meta below).
      - drop_rows: drops every row where this column is null, applied
        after every fill-type strategy above so a drop_rows strategy on
        one column never discards rows a constant/mean/... strategy on
        another column just filled.

    `add_indicator: true` on any strategy also adds a `{column}_was_missing`
    0/1 column recording which rows were originally null, computed before
    any fill runs -- the standard way to keep "this was imputed" itself as
    a feature rather than silently hiding it."""
    indicator_exprs = [
        pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_was_missing")
        for col, spec in strategies.items()
        if spec.get("add_indicator")
    ]
    out = df.with_columns(indicator_exprs) if indicator_exprs else df
    for col, spec in strategies.items():
        method = spec["method"]
        if method == "constant":
            out = out.with_columns(pl.col(col).fill_null(spec["value"]))
        elif method == "mean":
            out = out.with_columns(pl.col(col).fill_null(pl.col(col).mean()))
        elif method == "median":
            out = out.with_columns(pl.col(col).fill_null(pl.col(col).median()))
        elif method == "mode":
            out = out.with_columns(pl.col(col).fill_null(pl.col(col).mode().first()))
        elif method in ("flag", "drop_rows"):
            pass  # no fill: flag is a deliberate no-op, drop_rows is handled below
        else:
            raise ValueError(f"unknown missing-value strategy: {method!r}")
    drop_cols = [col for col, spec in strategies.items() if spec["method"] == "drop_rows"]
    if drop_cols:
        out = out.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in drop_cols]))
    return out


def _missing_value_treatment_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    strategies = params.get("strategies", {})
    result = dict(in_meta)
    for col, spec in strategies.items():
        if col in result:
            existing = result[col]
            result[col] = replace(existing, tags=[*existing.tags, f"missing_treatment:{spec['method']}"])
        indicator_col = f"{col}_was_missing"
        if spec.get("add_indicator") and indicator_col in df.columns:
            result[indicator_col] = ColumnMeta(dtype=str(df.schema[indicator_col]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="missing_value_treatment",
        block_type="standard",
        group="quality",
        display_name="Missing value treatment",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=missing_value_treatment,
        metadata_transform=_missing_value_treatment_meta,
    )
)
