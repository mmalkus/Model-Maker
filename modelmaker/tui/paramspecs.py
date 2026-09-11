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
