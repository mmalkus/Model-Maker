from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Python port of frontend/src/ParamsForm.tsx's PARAM_SPECS + deriveColumnFieldSpecs
# -- the same declarative field lists, so the TUI's param form matches the
# web UI's for every block category it covers. Anything not listed here
# (custom AI-authored blocks, or a standard block like groupby_agg whose
# params don't map cleanly to flat fields) falls back to raw JSON editing in
# the inspector, same as the web UI's textarea fallback.

FieldKind = Literal["text", "number", "select", "column", "columns"]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind
    placeholder: str = ""
    step: float | None = None
    options: tuple[str, ...] = ()
    # 'target' picks up the role=target column; 'predicted' picks up
    # whichever column a modelling block upstream tagged role=predicted --
    # see packet.resolve_role_column / blocks/modelling.py. Only meaningful
    # for kind='column'; leaving the param out of `params` entirely puts it
    # back in this dynamically-resolved "Auto" state.
    auto_role: Literal["target", "predicted"] | None = None


PARAM_SPECS: dict[str, list[FieldSpec]] = {
    "filter": [
        FieldSpec("expr", "Filter expression (SQL)", "text", placeholder="age > 30 and region = 'West'"),
    ],
    "select": [
        FieldSpec("cols", "Columns to keep", "columns"),
    ],
    "join": [
        FieldSpec("on", "Join on", "columns"),
        FieldSpec("how", "How", "select", options=("inner", "left", "right", "outer", "semi", "anti", "cross")),
    ],
    "train_test_split": [
        FieldSpec("test_size", "Test size (fraction)", "number", step=0.05),
        FieldSpec("seed", "Random seed", "number"),
        FieldSpec("stratify_col", "Stratify by (optional, e.g. the default flag)", "column"),
    ],
    "write_csv": [
        FieldSpec("filename", "Output filename", "text", placeholder="output.csv"),
    ],
    "generate_image": [
        FieldSpec("kind", "Chart type", "select", options=("hist", "bar", "scatter", "line")),
        FieldSpec("x", "X column", "column"),
        FieldSpec("y", "Y column (bar / scatter / line)", "column"),
        FieldSpec("bins", "Bins (hist)", "number"),
        FieldSpec("title", "Title", "text"),
    ],
    "glm_fit": [
        FieldSpec("target", "Target column", "column", auto_role="target"),
        FieldSpec("features", "Feature columns", "columns"),
        FieldSpec("family", "Family", "select", options=("gaussian", "poisson", "gamma", "inverse_gaussian")),
        FieldSpec("alpha", "Regularization (alpha)", "number", step=0.01),
    ],
    "logistic_regression": [
        FieldSpec("target", "Target column (binary)", "column", auto_role="target"),
        FieldSpec("features", "Feature columns", "columns"),
        FieldSpec("C", "Inverse regularization (C)", "number", step=0.1),
        FieldSpec("max_iter", "Max iterations", "number"),
    ],
    "woe_transform": [
        FieldSpec("col", "Column to transform", "column"),
        FieldSpec("target", "Target column (binary)", "column", auto_role="target"),
        FieldSpec("bins", "Bins (numeric columns)", "number"),
    ],
    "ks_test": [
        FieldSpec("score_col", "Score column", "column", auto_role="predicted"),
        FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
    ],
    "auc_gini": [
        FieldSpec("score_col", "Score column", "column", auto_role="predicted"),
        FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
    ],
    "psi_test": [
        FieldSpec("col", "Column to compare", "column"),
        FieldSpec("bins", "Bins", "number"),
    ],
    "fit_master_scale": [
        FieldSpec("score_col", "Score column", "column", auto_role="predicted"),
        FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
        FieldSpec("n_grades", "Number of grades", "number"),
        FieldSpec("algorithm", "Algorithm", "select", options=("quantile", "equal_width", "monotonic_default_rate")),
        FieldSpec("min_grade_share", "Minimum grade share (e.g. 0.03)", "number", step=0.01),
    ],
    "assign_rating_grade": [
        FieldSpec("score_col", "Score column (defaults to the scale's own)", "column", auto_role="predicted"),
    ],
    "rating_summary": [
        FieldSpec("grade_col", "Grade column", "column"),
        FieldSpec("target_col", "Target column (binary)", "column", auto_role="target"),
    ],
    "lgd_regression": [
        FieldSpec("target", "Target column (LGD or CCF, in [0, 1])", "column", auto_role="target"),
        FieldSpec("features", "Feature columns", "columns"),
        FieldSpec("max_iter", "Max iterations", "number"),
        FieldSpec("tol", "Convergence tolerance", "number", step=1e-8),
    ],
    "compute_lgd": [
        FieldSpec("ead_col", "EAD column", "column"),
        FieldSpec("recovered_col", "Recovered amount column", "column"),
        FieldSpec("cost_col", "Workout cost column (optional)", "column"),
        FieldSpec("floor", "Floor", "number", step=0.05),
        FieldSpec("cap", "Cap", "number", step=0.05),
    ],
    "forward_default_flag": [
        FieldSpec("id_col", "Account id column", "column"),
        FieldSpec("date_col", "Month column (date or YYYYMM)", "column"),
        FieldSpec("default_col", "Current default status column", "column"),
        FieldSpec("horizon", "Horizon (months)", "number"),
        FieldSpec("flag_col", "Flag column name (optional)", "text", placeholder="default_12m"),
        FieldSpec("incomplete", "Incomplete outcome windows", "select", options=("drop", "null",)),
    ],
    "compute_ccf": [
        FieldSpec("limit_col", "Limit column", "column"),
        FieldSpec("balance_ref_col", "Balance at reference date column", "column"),
        FieldSpec("balance_default_col", "Balance at default column", "column"),
        FieldSpec("floor", "Floor", "number", step=0.05),
        FieldSpec("cap", "Cap", "number", step=0.05),
        FieldSpec("no_headroom", "No undrawn headroom at reference", "select", options=("zero", "null",)),
    ],
    "continuous_accuracy": [
        FieldSpec("actual_col", "Actual column", "column", auto_role="target"),
        FieldSpec("predicted_col", "Predicted column", "column", auto_role="predicted"),
    ],
    "bucketed_calibration": [
        FieldSpec("actual_col", "Actual column", "column", auto_role="target"),
        FieldSpec("predicted_col", "Predicted column", "column", auto_role="predicted"),
        FieldSpec("bins", "Bins", "number"),
    ],
    # Binning / univariate, calibration, LGD/CCF/EAD and back-testing blocks.
    "time_split": [
        FieldSpec("date_col", "Date column", "column"),
        FieldSpec("cutoff", "Cut-off date (first out-of-time day)", "text", placeholder="2025-01-01"),
        FieldSpec("oot_end", "Out-of-time end date (optional, exclusive)", "text", placeholder="2026-01-01"),
    ],
    "apply_binning": [
        FieldSpec("output", "Output", "select", options=("woe", "target_mean", "bin", "both",)),
        FieldSpec("features", "Features (optional -- defaults to every binned feature)", "columns"),
    ],
    "scorecard_table": [
        FieldSpec("base_score", "Base score", "number"),
        FieldSpec("base_odds", "Base odds (good:bad)", "number"),
        FieldSpec("pdo", "Points to double the odds", "number"),
    ],
    "characteristic_stability": [
        FieldSpec("features", "Features (optional -- defaults to all shared columns)", "columns"),
        FieldSpec("bins", "Bins (numeric features)", "number"),
    ],
    "target_trend": [
        FieldSpec("date_col", "Date column", "column"),
        FieldSpec("target_col", "Target column", "column", auto_role="target"),
        FieldSpec("period", "Period", "select", options=("month", "quarter", "year",)),
        FieldSpec("features", "Features to trend (optional)", "columns"),
    ],
    "bin_chart": [
        FieldSpec("feature", "Feature", "text", placeholder="credit_score"),
        FieldSpec("title", "Title (optional)", "text"),
    ],
    "stepwise_selection": [
        FieldSpec("target", "Target column (binary, or LGD/CCF in [0, 1])", "column", auto_role="target"),
        FieldSpec("features", "Candidate features", "columns"),
        FieldSpec("direction", "Direction", "select", options=("both", "forward", "backward",)),
        FieldSpec("p_enter", "p-value to enter", "number", step=0.01),
        FieldSpec("p_remove", "p-value to remove", "number", step=0.01),
        FieldSpec("max_features", "Max features (optional)", "number"),
    ],
    "calibrate_model": [
        FieldSpec("central_tendency", "Central tendency (target mean, e.g. long-run default rate)", "number", step=0.001),
    ],
    "margin_of_conservatism": [
        FieldSpec("predicted_col", "Estimate column", "column", auto_role="predicted"),
        FieldSpec("add_on", "Add-on", "number", step=0.01),
        FieldSpec("multiplier", "Multiplier", "number", step=0.05),
        FieldSpec("floor", "Floor (optional)", "number", step=0.0001),
        FieldSpec("cap", "Cap (optional)", "number", step=0.05),
        FieldSpec("output_col", "Output column", "text", placeholder="predicted_moc"),
    ],
    "discount_recoveries": [
        FieldSpec("id_col", "Facility id column (both inputs)", "column"),
        FieldSpec("default_date_col", "Default date column (facilities)", "column"),
        FieldSpec("cf_date_col", "Cash-flow date column (cash flows)", "column"),
        FieldSpec("recovery_col", "Recovery amount column", "column"),
        FieldSpec("cost_col", "Workout cost column (optional)", "column"),
        FieldSpec("annual_rate", "Annual discount rate", "number", step=0.005),
        FieldSpec("rate_col", "Per-facility rate column (optional, overrides the rate)", "column"),
    ],
    "compute_ead": [
        FieldSpec("balance_col", "Drawn balance column", "column"),
        FieldSpec("limit_col", "Limit column", "column"),
        FieldSpec("predicted_col", "CCF column", "column", auto_role="predicted"),
        FieldSpec("output_col", "Output column", "text", placeholder="ead_predicted"),
    ],
    "long_run_average": [
        FieldSpec("target_col", "Realised value column (LGD, CCF or default flag)", "column", auto_role="target"),
        FieldSpec("date_col", "Default date column", "column"),
        FieldSpec("weight_col", "Exposure weight column (optional, e.g. EAD)", "column"),
        FieldSpec("period", "Period", "select", options=("year", "quarter", "month",)),
    ],
    "grade_backtest": [
        FieldSpec("grade_col", "Grade column", "column"),
        FieldSpec("target_col", "Default flag column", "column", auto_role="target"),
        FieldSpec("pd_col", "Grade PD column (e.g. grade_pd)", "column"),
        FieldSpec("confidence", "Confidence level", "number", step=0.005),
    ],
}


def _prettify_column_param_label(key: str) -> str:
    base = key[:-4] if key.endswith("_col") else key
    base = base.replace("_", " ")
    return f"{base[:1].upper()}{base[1:]} column"


def derive_column_field_specs(params: dict[str, object]) -> list[FieldSpec]:
    """AI-drafted (custom) blocks name every existing input column they read
    as a `<name>_col` string parameter (see llm/prompts.CONTRACT) instead of
    hardcoding it into the function body. Turn each such param into a
    'column' field (a picker over the block's real input columns) instead
    of leaving it in raw JSON -- a stale value (e.g. after an upstream
    rename) then shows up as an extra, clearly-off option rather than
    silently failing at run time."""
    keys = sorted(k for k, v in params.items() if k.endswith("_col") and (v is None or isinstance(v, str)))
    return [FieldSpec(key, _prettify_column_param_label(key), "column") for key in keys]


def field_specs_for(category: str, params: dict[str, object], is_custom: bool) -> list[FieldSpec]:
    """The field list to render for a block: its declarative PARAM_SPECS
    entry if it has one, else (for custom blocks) the _col-derived column
    pickers, else nothing -- the caller falls back to raw JSON."""
    if category in PARAM_SPECS:
        return list(PARAM_SPECS[category])
    if is_custom:
        return derive_column_field_specs(params)
    return []
