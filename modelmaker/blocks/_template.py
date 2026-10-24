"""What the blocks in this file are about -- one line, shown as the file's
heading in the block library. Group related blocks in one file (e.g. all
PD backtests in pd_backtests.py).

This is the template "New block" starts from, and the reference for every
key a block may use. See docs/block-authoring.md for the full guide.
"""

# Only polars at module level: import anything else inside the function
# that uses it, so each block stays a self-contained unit in a compiled
# script.
import polars as pl

# A plain literal -- read without running this file. One entry per block;
# the key is the block's id and must be the name of a function below.
BLOCKS = {
    "flag_outliers": {
        # Shown on the canvas and in the palette.
        "display_name": "Flag outliers",
        # Bump when you change the block: placed copies offer "Update to vN".
        "version": 1,
        # "input" (no upstream; refresh / change detection), "standard", or
        # "output" (may take the engine's output_dir).
        "block_type": "standard",
        # Palette group (optional) and catalogue tags (from the fixed list).
        "group": "preprocessing",
        "tags": ["data_prep"],
        # pip requirements beyond Model-Maker's own (optional).
        "requires": [],
        # Ports. Each input port is a parameter of the function, by name.
        # "lazy": True on an input means the function also works on a
        # pl.LazyFrame there (a streaming run may fuse it); on an output it
        # means the function returns a pl.LazyFrame (the engine collects it).
        # "aggregate": True marks an output as a statistics table (one row
        # per feature/bin/grade/..., never per record).
        "inputs": [{"name": "df", "type": "dataframe", "lazy": True}],
        "outputs": [{"name": "out", "type": "dataframe"}],
        # Params form fields worth a label, placeholder or options; every
        # other param gets a field inferred from the signature. Kinds: text,
        # number, select, column, columns, checkbox, path (a file the block
        # reads -- its changes are detected), sql, env (an env var name).
        "form": [
            {"key": "column", "label": "Column", "kind": "column"},
            {"key": "z", "label": "Z-score threshold", "kind": "number", "step": 0.5},
        ],
        # How output columns get their metadata (optional). Columns from an
        # input keep its role; new ones get "new_role"; "roles" tags named
        # columns. {"inherit": "none"} for a block that builds a new table.
        "meta": {"new_role": "feature"},
        # The decision-hint rule that judges this block's result for an AI
        # build (optional; see agent/hint_rules.py for the names).
        # "hints": "auc_gini",
        # Param renamed since an earlier version: Update carries the value.
        # "renamed": {"threshold": "z"},
    },
}


def flag_outliers(df: pl.DataFrame, column: str, z: float = 3.0) -> pl.DataFrame:
    """Adds `<column>_outlier`: true where `column` is more than `z`
    standard deviations from its mean.

    The first line is the block's summary; the whole docstring is its
    description in the library and the AI catalogue."""
    score = (pl.col(column) - pl.col(column).mean()) / pl.col(column).std()
    return df.with_columns((score.abs() > z).alias(f"{column}_outlier"))


def _example_flag_outliers():
    # Optional self-check, run by "Try it" and `modelmaker check-blocks`.
    df = pl.DataFrame({"x": [1.0, 1.1, 0.9, 1.0, 9.0]})
    out = flag_outliers(df, "x", z=1.5)
    assert out["x_outlier"].to_list() == [False, False, False, False, True]
