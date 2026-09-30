"""Binning / univariate-analysis blocks: fit a binning once (on the
development/train sample), then apply it everywhere else. This is the
fit/apply split woe_transform lacks -- woe_transform re-derives bins and WoE
from whatever data it's given, so running it on a test or out-of-time
sample quietly uses *that* sample's target (leakage that inflates test
Gini). The fitted binning is a plain JSON-shaped dict on a "binning" port,
same convention as a "model" or "master_scale" artifact.

fit_binning doubles as the univariate analysis: alongside the artifact it
emits the per-bin table (population, event rate / mean target, WoE, IV
contribution) and a per-feature summary (IV, direction-adjusted univariate
Gini/KS, or rank correlation and explained variance for a continuous
target, fill rate, monotonicity).

Same self-containment rule as the rest of the library: every helper a
block needs lives *inside* its fn, since the compiled script inlines only
inspect.getsource(fn) (see compiler.py).
"""

from __future__ import annotations

import polars as pl

from ..metadata_transforms import infer_dtypes
from ..packet import ColumnMeta, ColumnRole
from .base import BlockSpec, PortSpec, register_block


def fit_binning(
    df: pl.DataFrame,
    target: str,
    features: list[str] | None = None,
    max_bins: int = 6,
    min_bin_share: float = 0.05,
    monotonic: bool = True,
    fine_bins: int = 20,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Univariate analysis and binning (coarse classing) of every candidate
    feature against `target` -- fit it on the development/train sample and
    reuse the `binning` artifact with apply_binning on test/OOT data, so
    bins and WoE are never re-derived from a sample's own target.

    `target` is either binary (0/1, 1 = event/bad -- a PD target) or
    continuous (e.g. LGD or CCF in [0, 1]); it auto-fills from the input's
    role=target column. `features` defaults to every other column that
    looks like a feature: numeric columns, plus text columns with at most 50
    distinct values (so id-like strings and dates are skipped) -- pass the
    list explicitly to be sure ids and excluded columns stay out.

    Numeric features start from `fine_bins` quantile bins, then: bins below
    `min_bin_share` of the population are merged into their closest
    neighbour; with `monotonic` (default), adjacent bins are merged until
    the event rate / mean target moves in one direction only (the direction
    of the feature's rank correlation with the target); finally the closest
    adjacent pair is merged until at most `max_bins` remain. Categorical
    features get one bin per category, with categories below
    `min_bin_share` pooled into an "__other__" bin (which also catches
    categories unseen at fit time). Nulls always get their own "__missing__"
    bin, never mixed into a value bin.

    Outputs:
      - `binning`: the fitted artifact (kind "binning") for apply_binning /
        scorecard_table.
      - `bins`: one row per feature x bin -- label, bounds, count, share of
        population, events (binary), event rate or mean target, WoE and IV
        contribution (binary). The standard univariate exhibit.
      - `summary`: one row per feature, sorted by strength -- IV and its
        band, univariate Gini/AUC/KS of the binned feature (direction-
        adjusted, so a higher-is-safer feature isn't reported as a negative
        Gini), Spearman correlation of the raw feature (numeric) and
        explained variance of the binned feature (continuous target), fill
        rate, bin count, and whether the binned target rate is monotonic."""
    import math

    import numpy as np
    from scipy.stats import rankdata

    y_all = df[target].cast(pl.Float64).to_numpy()
    if np.isnan(y_all).any():
        raise ValueError(f"target '{target}' has nulls -- drop or treat them before binning")
    uniq = np.unique(y_all)
    is_binary = bool(np.isin(uniq, [0.0, 1.0]).all())
    if is_binary:
        total_bads = float(y_all.sum())
        total_goods = float(len(y_all) - total_bads)
        if total_bads == 0 or total_goods == 0:
            raise ValueError(f"binary target '{target}' needs both 0 and 1 values")
    overall_mean = float(y_all.mean())
    n_total = len(y_all)

    if features is None:
        features = [
            c
            for c in df.columns
            if c != target
            and (
                (df.schema[c].is_numeric() and df.schema[c] != pl.Boolean)
                or (df.schema[c] in (pl.Utf8, pl.Categorical, pl.Boolean) and df[c].n_unique() <= 50)
            )
        ]

    def _woe(bads: float, n: float) -> float:
        goods = n - bads
        return math.log(((goods + 0.5) / (total_goods + 0.5)) / ((bads + 0.5) / (total_bads + 0.5)))

    def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
        if len(x) < 3:
            return None
        rx, ry = rankdata(x), rankdata(y)
        if rx.std() == 0 or ry.std() == 0:
            return None
        return float(np.corrcoef(rx, ry)[0, 1])

    def _auc(score: np.ndarray, y: np.ndarray) -> float | None:
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = rankdata(score)  # average ranks, so ties count half
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def _ks(score: np.ndarray, y: np.ndarray) -> float | None:
        pos, neg = np.sort(score[y == 1]), np.sort(score[y == 0])
        if len(pos) == 0 or len(neg) == 0:
            return None
        grid = np.unique(score)
        return float(np.max(np.abs(np.searchsorted(pos, grid, "right") / len(pos) - np.searchsorted(neg, grid, "right") / len(neg))))

    def _iv_band(iv: float) -> str:
        if iv < 0.02:
            return "useless"
        if iv < 0.1:
            return "weak"
        if iv < 0.3:
            return "medium"
        if iv < 0.5:
            return "strong"
        return "suspicious"

    def _merge_numeric(xs: np.ndarray, ys: np.ndarray) -> list[float]:
        """Inner edges (bins are (lo, hi], right-closed) after merging."""
        edges = list(np.unique(np.quantile(xs, np.linspace(0, 1, fine_bins + 1))[1:-1]))
        # pools: [sum_y, count]; pool i spans (edges[i-1], edges[i]].
        idx = np.searchsorted(np.array(edges), xs, side="left")
        pools = [[float(ys[idx == k].sum()), int((idx == k).sum())] for k in range(len(edges) + 1)]
        # Drop empty pools together with the edge that bounds them above.
        keep_edges, keep_pools = [], []
        for k, p in enumerate(pools):
            if p[1] == 0:
                continue
            keep_pools.append(p)
            if k < len(edges):
                keep_edges.append(edges[k])
        pools = keep_pools
        edges = keep_edges[: len(pools) - 1]

        def merge(i: int) -> None:  # merge pool i with pool i+1
            pools[i] = [pools[i][0] + pools[i + 1][0], pools[i][1] + pools[i + 1][1]]
            del pools[i + 1]
            del edges[i]

        def rate(p) -> float:
            return p[0] / p[1]

        # 1. minimum bin size
        min_n = max(1, math.ceil(min_bin_share * len(xs)))
        while len(pools) > 1:
            small = [i for i, p in enumerate(pools) if p[1] < min_n]
            if not small:
                break
            i = min(small, key=lambda k: pools[k][1])
            if i == 0:
                merge(0)
            elif i == len(pools) - 1:
                merge(i - 1)
            else:
                left_gap = abs(rate(pools[i]) - rate(pools[i - 1]))
                right_gap = abs(rate(pools[i]) - rate(pools[i + 1]))
                merge(i - 1 if left_gap <= right_gap else i)
        # 2. monotonic event rate / mean target
        if monotonic and len(pools) > 1:
            rho = _spearman(xs, ys) or 0.0
            sign = 1.0 if rho >= 0 else -1.0
            changed = True
            while changed and len(pools) > 1:
                changed = False
                for i in range(len(pools) - 1):
                    if sign * (rate(pools[i + 1]) - rate(pools[i])) < -1e-12:
                        merge(i)
                        changed = True
                        break
        # 3. at most max_bins
        while len(pools) > max(1, max_bins):
            gaps = [abs(rate(pools[i + 1]) - rate(pools[i])) for i in range(len(pools) - 1)]
            merge(int(np.argmin(gaps)))
        return [float(e) for e in edges]

    def _fmt(v: float) -> str:
        return f"{v:.6g}"

    artifact_features: dict[str, dict] = {}
    bin_rows: list[dict] = []
    summary_rows: list[dict] = []

    for feat in features:
        s = df[feat]
        null_mask = s.is_null().to_numpy()
        fill_rate = float(1.0 - null_mask.mean()) if n_total else None
        is_numeric = s.dtype.is_numeric() and s.dtype != pl.Boolean
        bins: list[dict] = []
        assign = np.full(n_total, -1)  # bin index per row, for the summary stats

        if is_numeric:
            xs_all = s.cast(pl.Float64).to_numpy()
            xs, ys = xs_all[~null_mask], y_all[~null_mask]
            edges = _merge_numeric(xs, ys) if len(xs) else []
            bounds = [-math.inf] + edges + [math.inf]
            idx = np.searchsorted(np.array(edges), xs_all, side="left") if edges else np.zeros(n_total, dtype=int)
            for k in range(len(bounds) - 1):
                lo, hi = bounds[k], bounds[k + 1]
                label = f"({_fmt(lo) if lo != -math.inf else '-inf'}, {_fmt(hi) if hi != math.inf else 'inf'}]"
                bins.append({"label": label, "lower": None if lo == -math.inf else lo, "upper": None if hi == math.inf else hi})
                assign[(idx == k) & ~null_mask] = k
            spec = {"type": "numeric", "edges": edges}
        else:
            cats = s.cast(pl.Utf8).to_numpy()
            values, counts = np.unique(cats[~null_mask].astype(str), return_counts=True)
            min_n = max(1, math.ceil(min_bin_share * n_total))
            big = [v for v, c in zip(values, counts) if c >= min_n]
            small = [v for v, c in zip(values, counts) if c < min_n]
            # Order category bins by event rate / mean target (readability,
            # and what a scorecard table is normally sorted by).
            def _cat_rate(v: str) -> float:
                return float(y_all[(~null_mask) & (cats == v)].mean())

            big.sort(key=_cat_rate)
            mapping = {}
            for k, v in enumerate(big):
                bins.append({"label": str(v), "lower": None, "upper": None, "categories": [str(v)]})
                mapping[str(v)] = k
                assign[(~null_mask) & (cats == v)] = k
            other_idx = len(bins)
            bins.append({"label": "__other__", "lower": None, "upper": None, "categories": [str(v) for v in small]})
            for v in small:
                mapping[str(v)] = other_idx
                assign[(~null_mask) & (cats == v)] = other_idx
            spec = {"type": "categorical", "mapping": mapping, "other_bin": other_idx}

        missing_idx = len(bins)
        bins.append({"label": "__missing__", "lower": None, "upper": None})
        assign[null_mask] = missing_idx

        feat_iv = 0.0
        row_value = np.zeros(n_total)  # WoE (binary) or bin mean (continuous) per row
        for k, b in enumerate(bins):
            mask = assign == k
            n = int(mask.sum())
            ys_k = y_all[mask]
            mean_k = float(ys_k.mean()) if n else None
            b.update({"bin": k, "count": n, "share": n / n_total if n_total else 0.0, "target_mean": mean_k})
            if is_binary:
                bads = float(ys_k.sum())
                # An empty bin (e.g. no missings at fit time) is neutral: WoE 0.
                woe = _woe(bads, n) if n else 0.0
                good_pct = (n - bads + 0.5) / (total_goods + 0.5)
                bad_pct = (bads + 0.5) / (total_bads + 0.5)
                iv_k = (good_pct - bad_pct) * woe if n else 0.0
                feat_iv += iv_k
                b.update({"events": int(bads), "woe": woe, "iv_contribution": iv_k})
                row_value[mask] = woe
            else:
                b["value"] = mean_k if n else overall_mean
                row_value[mask] = b["value"]
        # Drop empty value bins from the table (keep the missing bin: its
        # WoE/value is what unseen nulls map to at apply time).
        for b in bins:
            bin_rows.append({"feature": feat, **{k: b.get(k) for k in ("bin", "label", "lower", "upper", "count", "share", "events", "target_mean", "woe", "iv_contribution")}})

        value_rates = [b["target_mean"] for b in bins if b["label"] not in ("__missing__", "__other__") and b["count"]]
        if is_numeric and len(value_rates) > 1:
            diffs = np.diff(value_rates)
            monotone = bool((diffs >= -1e-12).all() or (diffs <= 1e-12).all())
        else:
            monotone = None

        summ: dict = {
            "feature": feat,
            "type": "numeric" if is_numeric else "categorical",
            "fill_rate": fill_rate,
            "n_bins": sum(1 for b in bins if b["count"]),
            "monotonic": monotone,
        }
        if is_binary:
            # Higher WoE = safer, so the binned score for risk is -WoE.
            auc = _auc(-row_value, y_all)
            summ.update(
                {
                    "iv": feat_iv,
                    "iv_band": _iv_band(feat_iv),
                    "auc": auc,
                    "gini": (2 * auc - 1) if auc is not None else None,
                    "ks": _ks(-row_value, y_all),
                }
            )
            if is_numeric:
                raw_auc = _auc(xs_all[~null_mask], y_all[~null_mask])
                summ["raw_direction"] = None if raw_auc is None else ("higher = riskier" if raw_auc >= 0.5 else "higher = safer")
        else:
            ss_tot = float(((y_all - overall_mean) ** 2).sum())
            ss_res = float(((y_all - row_value) ** 2).sum())
            summ.update(
                {
                    "r2_binned": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
                    "spearman_binned": _spearman(row_value, y_all),
                }
            )
        if is_numeric:
            summ["spearman_raw"] = _spearman(xs_all[~null_mask], y_all[~null_mask])
        summary_rows.append(summ)
        artifact_features[feat] = {**spec, "bins": [{k: b.get(k) for k in ("bin", "label", "lower", "upper", "count", "target_mean", "woe", "value")} for b in bins], "missing_bin": missing_idx}

    sort_key = "iv" if is_binary else "r2_binned"
    summary_rows.sort(key=lambda r: -(r.get(sort_key) or 0.0))
    artifact = {
        "kind": "binning",
        "target": target,
        "target_type": "binary" if is_binary else "continuous",
        "overall_target_mean": overall_mean,
        "features": artifact_features,
    }
    bins_df = pl.DataFrame(
        bin_rows,
        schema={
            "feature": pl.Utf8, "bin": pl.Int64, "label": pl.Utf8, "lower": pl.Float64, "upper": pl.Float64,
            "count": pl.Int64, "share": pl.Float64, "events": pl.Int64, "target_mean": pl.Float64,
            "woe": pl.Float64, "iv_contribution": pl.Float64,
        },
    )
    if not is_binary:
        bins_df = bins_df.drop("events", "woe", "iv_contribution")
    summary_df = pl.DataFrame(summary_rows, infer_schema_length=None)
    return artifact, bins_df, summary_df


def _fit_binning_meta(input_metas, outputs, params):
    return infer_dtypes(input_metas, {k: v for k, v in outputs.items() if isinstance(v, pl.DataFrame)}, params)


register_block(
    BlockSpec(
        category="fit_binning",
        block_type="standard",
        group="modelling",
        display_name="Univariate analysis & binning",
        inputs=[PortSpec("df")],
        outputs=[
            PortSpec("binning", type="binning"),
            PortSpec("bins", required=False),
            PortSpec("summary", required=False),
        ],
        fn=fit_binning,
        metadata_transform=_fit_binning_meta,
    )
)


def apply_binning(df: pl.DataFrame, binning: dict, output: str = "woe", features: list[str] | None = None) -> pl.DataFrame:
    """Applies a binning fitted by fit_binning to any sample (train, test,
    out-of-time, or new applications) without refitting -- every sample
    gets exactly the bins and WoE values derived on the fit sample.

    For each binned feature (or just `features`, when given) adds:
      - `output` = "woe" (default): `<feature>_woe`, the bin's weight of
        evidence (binary target only) -- the usual input for a logistic
        PD model / scorecard.
      - "target_mean": `<feature>_tm`, the bin's mean target on the fit
        sample (target encoding; the natural choice for an LGD/CCF model).
      - "bin": `<feature>_bin`, the bin label only.
      - "both": the WoE (or target-mean) column plus the bin label.
    Nulls map to the feature's missing bin; a category unseen at fit time
    maps to the "__other__" bin. Original columns are kept."""
    import numpy as np

    if binning.get("kind") != "binning":
        raise ValueError(f"apply_binning needs a binning artifact from fit_binning, got {binning.get('kind')!r}")
    if output not in ("woe", "target_mean", "bin", "both"):
        raise ValueError("output must be 'woe', 'target_mean', 'bin' or 'both'")
    binary = binning["target_type"] == "binary"
    if output == "woe" and not binary:
        raise ValueError("WoE needs a binary target -- use output='target_mean' for a continuous-target binning")
    value_key = "woe" if binary and output in ("woe", "both") else "value"
    if value_key == "value" and binary:
        value_key = "target_mean"
    suffix = "_woe" if value_key == "woe" else "_tm"

    specs = binning["features"]
    names = features or list(specs)
    new_cols = []
    for feat in names:
        spec = specs.get(feat)
        if spec is None:
            raise ValueError(f"'{feat}' isn't in this binning; binned features: {sorted(specs)}")
        if feat not in df.columns:
            raise ValueError(f"column '{feat}' is missing from this data")
        s = df[feat]
        null_mask = s.is_null().to_numpy()
        if spec["type"] == "numeric":
            xs = s.cast(pl.Float64).fill_null(0.0).to_numpy()
            idx = np.searchsorted(np.array(spec["edges"], dtype=float), xs, side="left") if spec["edges"] else np.zeros(len(xs), dtype=int)
        else:
            cats = s.cast(pl.Utf8).fill_null("").to_list()
            idx = np.array([spec["mapping"].get(c, spec["other_bin"]) for c in cats], dtype=int)
        idx = np.where(null_mask, spec["missing_bin"], idx)
        bins = spec["bins"]
        if output != "bin":
            lookup = np.array([
                (b.get(value_key) if b.get(value_key) is not None else (0.0 if value_key == "woe" else binning["overall_target_mean"]))
                for b in bins
            ], dtype=float)
            new_cols.append(pl.Series(f"{feat}{suffix}", lookup[idx]))
        if output in ("bin", "both"):
            labels = np.array([b["label"] for b in bins], dtype=object)
            new_cols.append(pl.Series(f"{feat}_bin", labels[idx].tolist(), dtype=pl.Utf8))
    return df.with_columns(new_cols)


def _apply_binning_meta(input_metas, outputs, params):
    in_meta = input_metas.get("df", {})
    (out_df,) = outputs.values()
    result = {}
    for name in out_df.columns:
        if name in in_meta:
            result[name] = in_meta[name]
        else:
            role = ColumnRole.SEGMENT if name.endswith("_bin") else ColumnRole.FEATURE
            result[name] = ColumnMeta(dtype=str(out_df.schema[name]), role=role)
    return {"out": result}


register_block(
    BlockSpec(
        category="apply_binning",
        block_type="standard",
        group="modelling",
        display_name="Apply binning (WoE)",
        inputs=[PortSpec("df"), PortSpec("binning", type="binning")],
        outputs=[PortSpec("out")],
        fn=apply_binning,
        metadata_transform=_apply_binning_meta,
    )
)


def scorecard_table(model: dict, binning: dict, base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0) -> pl.DataFrame:
    """The points-based scorecard itself: points per feature per bin, from
    a logistic_regression fitted on apply_binning's `<feature>_woe` columns
    and the binning it came from. Standard points-to-double-odds scaling --
    `base_odds` (good:bad) scores `base_score`, every `pdo` points doubles
    the odds -- with the intercept spread evenly over the features, so an
    applicant's score is just the sum of their bins' points.
    points = -(coef * WoE + intercept / n) * pdo/ln(2) + offset / n,
    offset = base_score - pdo/ln(2) * ln(base_odds)."""
    import math

    if model.get("kind") != "logistic_regression":
        raise ValueError(f"scorecard_table needs a logistic_regression model, got {model.get('kind')!r}")
    if binning.get("kind") != "binning" or binning.get("target_type") != "binary":
        raise ValueError("scorecard_table needs a binary-target binning from fit_binning")
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    coefs = model["coefficients"]
    n = len(coefs)
    rows = []
    for col, coef in coefs.items():
        feat = col[: -len("_woe")] if col.endswith("_woe") else col
        spec = binning["features"].get(feat)
        if spec is None:
            raise ValueError(f"model feature '{col}' isn't a WoE column of a binned feature (expected '<feature>_woe')")
        for b in spec["bins"]:
            if not b.get("count") and b["label"] not in ("__missing__", "__other__"):
                continue
            woe = b.get("woe") or 0.0
            points = -(coef * woe + model["intercept"] / n) * factor + offset / n
            rows.append({"feature": feat, "bin": b["bin"], "label": b["label"], "woe": woe, "coefficient": coef, "points": round(points, 2)})
    return pl.DataFrame(rows)


register_block(
    BlockSpec(
        category="scorecard_table",
        block_type="output",
        group="modelling",
        display_name="Scorecard points table",
        inputs=[PortSpec("model", type="model"), PortSpec("binning", type="binning")],
        outputs=[PortSpec("table")],
        fn=scorecard_table,
        metadata_transform=infer_dtypes,
    )
)
