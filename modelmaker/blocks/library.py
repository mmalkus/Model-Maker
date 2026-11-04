"""Data: read sources, filter, join, split and write.

Each block is a plain function over pl.DataFrame and literal params -- no
DataFramePacket, no ColumnMeta. A compiled script emits these verbatim; the
engine wraps packets around calls at run time and applies each block's
declared metadata rules separately.
"""

from __future__ import annotations

import polars as pl

BLOCKS = {
    "read_csv": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read CSV",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [{"key": "path", "label": "File path", "kind": "path", "options": [".csv"]}],
        "meta": {"inherit": "none"},
    },
    "demo_credit_data": {
        "display_name": "Demo credit data (PD/LGD/CCF)",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "meta": {"inherit": "none"},
    },
    "read_parquet": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read Parquet",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [{"key": "path", "label": "File path", "kind": "path", "options": [".parquet"]}],
        "meta": {"inherit": "none"},
    },
    "read_json": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read JSON",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "form": [{"key": "path", "label": "File path", "kind": "path", "options": [".json"]}],
        "meta": {"inherit": "none"},
    },
    "read_excel": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read Excel",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "path", "label": "File path", "kind": "path", "options": [".xlsx"]},
            {
                "key": "sheet",
                "label": "Sheet name (optional -- defaults to the first sheet)",
                "kind": "text",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "read_sql": {
        "capabilities": ["filesystem"],
        "use_instead": "read_source",
        "display_name": "Read SQL",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "form": [
            {
                "key": "connection_env",
                "label": "Connection variable",
                "kind": "env",
                "placeholder": "WAREHOUSE_URI",
            },
            {
                "key": "query",
                "label": "SQL query",
                "kind": "sql",
                "placeholder": "SELECT * FROM loans",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "read_dataset": {
        "display_name": "Read dataset",
        "version": 1,
        "block_type": "input",
        "capabilities": ["datasets"],
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [
            {"key": "dataset", "label": "Dataset", "kind": "dataset"},
            {"key": "sheet", "label": "Sheet name (Excel only; optional)", "kind": "text"},
        ],
        "meta": {"inherit": "none"},
    },
    "read_source": {
        "display_name": "Read from source",
        "version": 1,
        "block_type": "input",
        "capabilities": ["sources"],
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [
            {"key": "source", "label": "Source", "kind": "source"},
            {"key": "query", "label": "SQL query", "kind": "sql", "placeholder": "SELECT * FROM loans"},
        ],
        "meta": {"inherit": "none"},
    },
    "filter": {
        "display_name": "Filter",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [
            {
                "key": "expr",
                "label": "Filter expression (SQL)",
                "kind": "text",
                "placeholder": "age > 30 and region = 'West'",
            },
        ],
    },
    "select": {
        "display_name": "Select columns",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [{"key": "cols", "label": "Columns to keep", "kind": "columns"}],
    },
    "groupby_agg": {
        "display_name": "Group by / aggregate",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "meta": {"inherit": {"columns_param": "by"}},
    },
    "join": {
        "display_name": "Join",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "left", "lazy": True}, {"name": "right", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "on", "label": "Join on", "kind": "columns"},
            {
                "key": "how",
                "label": "How",
                "kind": "select",
                "options": ["inner", "left", "right", "full", "semi", "anti"],
            },
        ],
    },
    "train_test_split": {
        "display_name": "Train/test split",
        "version": 1,
        "block_type": "standard",
        "tags": ["sampling"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "train"}, {"name": "test"}],
        "form": [
            {"key": "test_size", "label": "Test size (fraction)", "kind": "number", "step": 0.05},
            {"key": "seed", "label": "Random seed", "kind": "number"},
            {
                "key": "stratify_col",
                "label": "Stratify by (optional, e.g. the default flag)",
                "kind": "column",
            },
        ],
        "hints": "split",
    },
    "time_split": {
        "display_name": "Out-of-time split",
        "version": 1,
        "block_type": "standard",
        "tags": ["sampling"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "development"}, {"name": "out_of_time"}],
        "form": [
            {"key": "date_col", "label": "Date column", "kind": "column"},
            {
                "key": "cutoff",
                "label": "Cut-off date (first out-of-time day)",
                "kind": "text",
                "placeholder": "2025-01-01",
            },
            {
                "key": "oot_end",
                "label": "Out-of-time end date (optional, exclusive)",
                "kind": "text",
                "placeholder": "2026-01-01",
            },
        ],
        "hints": "split",
    },
    "derive_columns": {
        "display_name": "Derive columns",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "meta": {"new_role": "feature"},
    },
    "one_hot_encode": {
        "display_name": "One-hot encode",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "meta": {"new_role": "feature"},
    },
    "write_csv": {
        "capabilities": ["filesystem"],
        "use_instead": "export",
        "display_name": "Write CSV",
        "version": 1,
        "block_type": "output",
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [],
        "form": [
            {
                "key": "filename",
                "label": "Output filename",
                "kind": "text",
                "placeholder": "output.csv",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "export": {
        "display_name": "Export",
        "version": 1,
        "block_type": "output",
        "tags": ["output"],
        "capabilities": ["exports"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "format", "label": "Format", "kind": "select", "options": ["csv", "parquet"]},
            {"key": "name", "label": "Export name", "kind": "text", "placeholder": "<block name>"},
        ],
        "meta": {"inherit": "all"},
    },
    "display_table": {
        "display_name": "Display table",
        "version": 2,
        "block_type": "output",
        "tags": ["output"],
        "display": "table",
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "save_csv", "label": "Also save as CSV", "kind": "checkbox"},
            {"key": "filename", "label": "CSV file name", "kind": "text", "placeholder": "<block id>.csv"},
        ],
    },
    "display_value": {
        "display_name": "View value",
        "version": 2,
        "block_type": "output",
        "tags": ["output"],
        "display": "value",
        "inputs": [{"name": "value", "type": "any"}],
        "outputs": [{"name": "value", "type": "any"}],
        "form": [
            {"key": "save_csv", "label": "Also save as CSV", "kind": "checkbox"},
            {"key": "filename", "label": "CSV file name", "kind": "text", "placeholder": "<block id>.csv"},
        ],
    },
    "display_kpi": {
        "display_name": "Headline number",
        "version": 1,
        "block_type": "output",
        "tags": ["output"],
        "display": "kpi",
        "inputs": [{"name": "value", "type": "any"}],
        "outputs": [{"name": "kpi", "type": "scalar_metric"}],
        "form": [
            {"key": "column", "label": "Column (or metric key)", "kind": "column"},
            {
                "key": "agg",
                "label": "Aggregate rows",
                "kind": "select",
                "options": ["first", "last", "sum", "mean", "min", "max", "count"],
            },
            {"key": "label", "label": "Label", "kind": "text", "placeholder": "e.g. Total RWA"},
            {
                "key": "number_format",
                "label": "Format",
                "kind": "select",
                "options": ["number", "compact", "percent"],
            },
            {"key": "decimals", "label": "Decimals", "kind": "number", "step": 1},
            {"key": "unit", "label": "Unit", "kind": "text", "placeholder": "e.g. EUR"},
        ],
    },
    "generate_image": {
        "display_name": "Generate image",
        "version": 1,
        "block_type": "output",
        "tags": ["output"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "image", "type": "image"}],
        "form": [
            {
                "key": "kind",
                "label": "Chart type",
                "kind": "select",
                "options": ["hist", "bar", "scatter", "line"],
            },
            {"key": "x", "label": "X column", "kind": "column"},
            {"key": "y", "label": "Y column (bar / scatter / line)", "kind": "column"},
            {"key": "bins", "label": "Bins (hist)", "kind": "number"},
            {"key": "title", "label": "Title", "kind": "text"},
        ],
        "meta": {"inherit": "none"},
    },
}


def read_csv(path: str) -> pl.LazyFrame:
    """Reads a CSV file into a dataframe on the `out` port; no inputs.
    `path` is the file path (relative to the working directory the engine
    runs in). Column dtypes are inferred by polars; column roles start
    untagged."""
    # A lazy scan: the engine collects it, pushing sample mode's row cap
    # down into the scan first, and a streaming run fuses it with whatever
    # follows (see the output port's "lazy" flag).
    return pl.scan_csv(path)


def demo_credit_data(n_rows: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generates the synthetic retail credit-risk demo dataset (term loans
    and credit cards, one row per facility) on the `out` port; no inputs,
    no file. Supports PD (`default_flag`, a 12-month default), LGD
    (`balance_at_default`, `recovery_amount`, `workout_cost` on defaults --
    see compute_lgd) and CCF (`credit_limit`, `balance_at_reference`,
    `balance_at_default` on defaulted credit cards -- see compute_ccf).
    `n_rows` is the number of records; `seed` fixes the data, so the same
    seed always gives the same rows, and a smaller `n_rows` gives the first
    rows of the larger dataset."""
    from modelmaker import demo_data

    if n_rows < 1:
        raise ValueError(f"n_rows must be at least 1, got {n_rows}")
    return demo_data.to_frame(n_rows, seed)


def read_parquet(path: str) -> pl.LazyFrame:
    """Reads a Parquet file into a dataframe on the `out` port; no inputs.
    `path` is the file path. Dtypes come from the Parquet schema."""
    return pl.scan_parquet(path)


def read_json(path: str) -> pl.DataFrame:
    """Reads a JSON file into a dataframe on the `out` port; no inputs.
    `path` ending in .jsonl/.ndjson (case-insensitive) is read as
    newline-delimited JSON (one object per line); anything else must be a
    plain JSON array of records."""
    # A plain JSON array has no lazy reader (polars must load it whole to
    # find its structure), so this block returns an eager frame and never
    # joins a streaming run's fusion.
    if path.lower().endswith((".jsonl", ".ndjson")):
        return pl.read_ndjson(path)
    return pl.read_json(path)


def read_excel(path: str, sheet: str | None = None) -> pl.DataFrame:
    """Reads one worksheet of an Excel workbook into a dataframe on the
    `out` port; no inputs. `path` is the .xlsx file path; `sheet` is the
    worksheet name -- unset/empty reads the first sheet. The first row is
    taken as the header."""
    return pl.read_excel(path, sheet_name=sheet) if sheet else pl.read_excel(path)


def read_sql(connection_env: str, query: str) -> pl.DataFrame:
    """Runs a SQL query against a database and returns the result set as a
    dataframe on the `out` port; no inputs. `connection_env` is the *name*
    of an environment variable (e.g. "WAREHOUSE_URI") holding the
    connection URI -- never the URI itself; raises KeyError if that
    variable isn't set. `query` is the SQL text, in the target database's
    own dialect. Set the block's optional `probe_query` (a cheap query such
    as SELECT MAX(updated_at) ...) to let "check for changes" detect new
    data."""
    import os

    # The connection string itself never lives in params -- it's baked as a
    # literal into saved project files and compiled scripts, so a raw DB
    # password there would leak into both. The secret is read fresh from
    # the environment at run time, live or compiled.
    uri = os.environ[connection_env]
    return pl.read_database_uri(query, uri)


def read_dataset(dataset: str, sheet: str | None = None) -> pl.LazyFrame:
    """Reads one of the workspace's datasets into a dataframe on the `out`
    port; no inputs. `dataset` is the dataset's name, picked from the
    workspace's datasets; before the block runs, the engine resolves it to
    the dataset's current file (read-only) and passes that path in its
    place -- so replacing the dataset re-runs the block. The file's format
    follows its extension: CSV (dtypes inferred), Parquet, Arrow IPC, JSON
    (a plain array of records), NDJSON or Excel (`sheet` names the
    worksheet; unset reads the first)."""
    lower = dataset.lower()
    if lower.endswith(".csv"):
        return pl.scan_csv(dataset)
    if lower.endswith(".parquet"):
        return pl.scan_parquet(dataset)
    if lower.endswith((".arrow", ".ipc", ".feather")):
        return pl.scan_ipc(dataset)
    if lower.endswith((".ndjson", ".jsonl")):
        return pl.scan_ndjson(dataset)
    if lower.endswith(".json"):
        return pl.read_json(dataset).lazy()
    if lower.endswith(".xlsx"):
        return (pl.read_excel(dataset, sheet_name=sheet) if sheet else pl.read_excel(dataset)).lazy()
    raise ValueError(f"can't read {dataset!r}: unknown dataset format")


def read_source(source: str, query: str) -> pl.LazyFrame:
    """Runs a SQL query against one of the data sources the administrator
    set up and returns the result set as a dataframe on the `out` port; no
    inputs. `source` is the source's name; `query` is the SQL text, in that
    database's own dialect. The query runs in the application, with the
    administrator's credentials, never in the block's own process: the
    engine stores its result as a dataset and passes that file in place of
    `source`. Refresh the block to run the query again."""
    # `query` has already run (see Runner._resolve_data); it's a parameter
    # so that it's part of the cache key and shown in the form.
    del query
    return pl.scan_parquet(source)


def filter(df: pl.DataFrame, expr: str) -> pl.DataFrame:
    """Keeps only the rows of `df` for which `expr` is true, emitted on
    `out` with the same columns and roles. `expr` is a Polars SQL boolean
    expression over column names, e.g. "age >= 18 AND region = 'EU'" or
    "balance IS NOT NULL" (string literals in single quotes)."""
    return df.filter(pl.sql_expr(expr))


def select(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Keeps only the listed columns of `df`, in the order given by `cols`,
    emitted on `out`; kept columns retain their roles. Every name in `cols`
    must exist in the input."""
    return df.select(cols)


def groupby_agg(df: pl.DataFrame, by: list[str], aggs: dict[str, str | list[str]]) -> pl.DataFrame:
    """Groups `df` by the `by` key columns and aggregates, one row per
    group on `out`. `aggs` maps a column name to the name of a polars
    expression method applied to it, e.g. {"balance": "sum", "pd":
    "mean"}; valid names include sum, mean, median, min, max, std, var,
    count, n_unique, first, last. A single aggregation keeps the column's
    own name; a list of them (e.g. {"lgd": ["mean", "count", "std"]})
    gives one column per aggregation, named `<column>_<agg>`. Only `by`
    plus the aggregated columns appear in the output. Key columns keep
    their roles; aggregated columns lose theirs. Output row order is not
    guaranteed."""
    agg_exprs = []
    for c, fn in aggs.items():
        if isinstance(fn, (list, tuple)):
            agg_exprs += [getattr(pl.col(c), f)().alias(f"{c}_{f}") for f in fn]
        else:
            agg_exprs.append(getattr(pl.col(c), fn)().alias(c))
    return df.group_by(by).agg(agg_exprs)


def join(left: pl.DataFrame, right: pl.DataFrame, on: list[str], how: str = "inner") -> pl.DataFrame:
    """Joins the `left` and `right` input dataframes on the key columns in
    `on` (same names on both sides), emitted on `out`. `how` is a polars
    join strategy: "inner" (default), "left", "right", "full", "semi", or
    "anti" ("cross" is not usable: polars rejects join keys with it).
    Non-key columns present on both sides get a "_right" suffix on the
    right-hand copy; a "full" join also keeps the right-hand keys as
    "<key>_right". Where a column exists on both sides, the left side's
    role wins. "outer" (polars' old name for "full") still works, for
    projects saved before the rename."""
    if how == "outer":
        how = "full"
    if how == "cross":
        raise ValueError('how="cross" is not supported: polars rejects join keys with it')
    return left.join(right, on=on, how=how)


def train_test_split(
    df: pl.DataFrame, test_size: float = 0.2, seed: int = 0, stratify_col: str | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Randomly splits `df`'s rows into two disjoint dataframes on the
    `train` and `test` ports; both keep every column and role. `test_size`
    is the fraction of rows sent to `test` (0-1, default 0.2; the test row
    count is rounded down), the rest go to `train`. `seed` fixes the
    shuffle, so the same seed gives the same split.

    `stratify_col` (e.g. the default flag, or a segment) splits each of its
    values separately, so `train` and `test` get the same share of each --
    the usual choice for a low default rate, where a plain random split can
    leave the test sample with a noticeably different event rate. Left
    unset, it's a simple random split."""
    if stratify_col is None:
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
        n_test = int(len(shuffled) * test_size)
        test = shuffled.head(n_test)
        train = shuffled.tail(len(shuffled) - n_test)
        return train, test
    trains, tests = [], []
    for _, part in sorted(df.group_by(stratify_col, maintain_order=True), key=lambda kv: str(kv[0])):
        shuffled = part.sample(fraction=1.0, shuffle=True, seed=seed)
        n_test = int(len(shuffled) * test_size)
        tests.append(shuffled.head(n_test))
        trains.append(shuffled.tail(len(shuffled) - n_test))
    return pl.concat(trains), pl.concat(tests)


def time_split(
    df: pl.DataFrame, date_col: str, cutoff: str, oot_end: str | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Out-of-time split on a date column: rows dated before `cutoff`
    (an ISO date, "YYYY-MM-DD") go to `development`, rows on or after it to
    `out_of_time` -- the standard way to hold back the most recent period
    for validation, instead of two hand-written filters. `oot_end`, when
    set, also drops rows on/after that date from `out_of_time` (e.g. a
    period whose outcome window isn't complete yet). `date_col` may be a
    Date/Datetime column or ISO date strings. Rows with a null date go to
    neither output. Both outputs keep every column and role."""
    from datetime import date

    dtype = df.schema[date_col]
    if dtype == pl.Utf8:
        d = pl.col(date_col).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    elif isinstance(dtype, pl.Datetime):
        d = pl.col(date_col).dt.date()
    else:
        d = pl.col(date_col).cast(pl.Date)
    cut = date.fromisoformat(cutoff)
    development = df.filter(d < cut)
    oot_mask = d >= cut
    if oot_end is not None:
        oot_mask = oot_mask & (d < date.fromisoformat(oot_end))
    out_of_time = df.filter(oot_mask)
    return development, out_of_time


def derive_columns(df: pl.DataFrame, expressions: dict[str, str]) -> pl.DataFrame:
    """Adds (or replaces) columns computed from SQL expressions over the
    existing columns -- ratios, flags, differences, bands -- the everyday
    feature engineering that otherwise needs a custom code block.
    `expressions` maps each new column name to a Polars SQL expression,
    e.g. {"utilisation": "balance / NULLIF(credit_limit, 0)",
    "headroom": "credit_limit - balance", "is_secured": "collateral_value > 0",
    "age_band": "CASE WHEN age < 30 THEN 'young' ELSE 'other' END"}.
    Expressions are evaluated against the input columns (not each other),
    in one step. New columns are tagged role=feature; a replaced column
    keeps its role."""
    return df.with_columns([pl.sql_expr(expr).alias(name) for name, expr in expressions.items()])


def one_hot_encode(
    df: pl.DataFrame, columns: list[str], drop_first: bool = True, max_categories: int = 30, keep_original: bool = False
) -> pl.DataFrame:
    """Dummy-encodes categorical `columns` into 0/1 indicator columns named
    `<column>_<value>` -- what a regression (logistic, GLM, LGD/CCF
    fractional logit) needs for a categorical driver like product type or
    seniority. `drop_first` (default) leaves out the alphabetically first
    category of each column as the reference level (always the same one,
    whatever the row order), so the dummies aren't collinear with the
    intercept. A null gets its own `<column>_null` indicator. Refuses a
    column with more than `max_categories` distinct values (almost
    certainly an id -- use fit_binning for a high-cardinality driver).
    `keep_original` keeps the source columns; by default they're dropped.

    The category set comes from the data it's run on, so encode before any
    train/test split to get the same dummies on both sides."""
    for c in columns:
        n = df[c].n_unique()
        if n > max_categories:
            raise ValueError(f"'{c}' has {n} distinct values (> max_categories={max_categories}) -- not a categorical to one-hot encode")
    dummies = []
    for c in columns:
        values = df[c].cast(pl.Utf8)
        levels = sorted(values.drop_nulls().unique().to_list())
        if drop_first and levels:
            levels = levels[1:]
        dummies += [(values == level).fill_null(False).cast(pl.Int8).alias(f"{c}_{level}") for level in levels]
        if values.null_count():
            dummies.append(values.is_null().cast(pl.Int8).alias(f"{c}_null"))
    base = df if keep_original else df.drop(columns)
    return base.hstack(dummies)


def write_csv(df: pl.DataFrame, filename: str, output_dir: str = ".") -> None:
    """Writes the `df` input to a CSV file; no output ports. `filename` is
    the file name (e.g. "scored.csv"), written inside `output_dir`, which
    the engine injects as the run's output directory -- leave it unset.
    Overwrites any existing file of the same name."""
    import os

    # .lazy() is a no-op on a LazyFrame: in a streaming run this block is
    # handed its upstream's uncollected plan and writes it straight to disk.
    df.lazy().sink_csv(os.path.join(output_dir, filename))


EXPORT_FORMATS = ("csv", "parquet")


def export(df: pl.DataFrame, format: str = "csv", name: str = "") -> pl.DataFrame:
    """Exports the `df` input as a file the user downloads -- `format` is
    "csv" or "parquet", `name` the export's name (default: the block's
    name) -- and passes `df` on unchanged on `out`. The export is made by
    the application from the block's result after each successful run (see
    datasets.ExportSink); in sample mode it holds only the sampled rows."""
    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}, got {format!r}")
    return df


def display_table(
    df: pl.DataFrame, save_csv: bool = True, filename: str = "", output_dir: str = ".", block_id: str = "table"
) -> pl.DataFrame:
    """Shows a results table right on its canvas node -- the first rows and
    the row count, updated whenever its upstream runs -- and passes `df` on
    unchanged on `out`. With `save_csv` on (the default) it also writes the
    whole table to `filename` (default `<block id>.csv`) in `output_dir`,
    the run's output directory, which the engine injects -- leave it unset.
    Use it for the tables worth reporting, e.g. a one-row RWA total."""
    if save_csv:
        _save_display_csv(df, filename, output_dir, block_id)
    return df


def display_value(value, save_csv: bool = True, filename: str = "", output_dir: str = ".", block_id: str = "value"):
    """Shows whatever comes in right on its canvas node and passes it on
    unchanged: a dataframe (first rows and row count), a metric or model
    dict (its keys and values) or an image. Its ports are typed "any" so it
    wires up to whatever you point it at. A dataframe is also saved as CSV
    like display_table does (`save_csv`, `filename`, `output_dir`)."""
    if save_csv and isinstance(value, pl.DataFrame):
        _save_display_csv(value, filename, output_dir, block_id)
    return value


def _save_display_csv(df: pl.DataFrame, filename: str, output_dir: str, block_id: str) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    df.write_csv(os.path.join(output_dir, filename or f"{block_id}.csv"))


_KPI_AGGS = ("first", "last", "sum", "mean", "min", "max", "count")


def display_kpi(
    value,
    column: str = "",
    agg: str = "first",
    label: str = "",
    number_format: str = "number",
    decimals: int = 2,
    unit: str = "",
) -> dict:
    """Shows one headline number, with a label, right on the canvas -- e.g.
    the total RWA at the end of a capital pipeline, or a model's Gini.
    `value` is a dataframe (its `column`, reduced over the rows by `agg`:
    first, last, sum, mean, min, max or count), a metric dict such as
    auc_gini's (its `column` key) or a plain number. Leave `column` empty
    when there is only one numeric column or key. `label` defaults to the
    column name; `number_format` (number, compact like 1.2M, percent),
    `decimals` and `unit` only change how the number is shown. Returns
    {"label", "value", "column", "agg", "number_format", "decimals", "unit"}
    on `kpi`."""
    import math
    import numbers

    if agg not in _KPI_AGGS:
        raise ValueError(f"agg must be one of {', '.join(_KPI_AGGS)}, got {agg!r}")
    if isinstance(value, pl.DataFrame):
        if not column:
            numeric = [c for c, t in value.schema.items() if t.is_numeric()]
            if len(numeric) != 1:
                raise ValueError(f"set `column`: the table has {len(numeric)} numeric columns ({', '.join(numeric)})")
            column = numeric[0]
        if column not in value.columns:
            raise ValueError(f"no column {column!r}; columns are {', '.join(value.columns)}")
        s = value[column]
        if agg == "count":
            number = s.drop_nulls().len()
        elif s.is_empty():
            number = None
        else:
            number = s[0] if agg == "first" else s[-1] if agg == "last" else getattr(s, agg)()
    elif isinstance(value, dict):
        if not column:
            numeric = [k for k, v in value.items() if isinstance(v, numbers.Real) and not isinstance(v, bool)]
            if len(numeric) != 1:
                raise ValueError(f"set `column` to one of the metric's keys: {', '.join(map(str, value))}")
            column = numeric[0]
        if column not in value:
            raise ValueError(f"no key {column!r}; keys are {', '.join(map(str, value))}")
        number = value[column]
    elif isinstance(value, numbers.Real):
        number = value
    else:
        raise TypeError(f"display_kpi needs a dataframe, a metric dict or a number, got {type(value).__name__}")
    if number is not None and not isinstance(number, numbers.Real):
        raise TypeError(f"{column!r} isn't a number: {number!r}")
    if isinstance(number, float) and not math.isfinite(number):
        number = None
    return {
        "label": label or column or "value",
        "value": number if number is None or isinstance(number, int) else float(number),
        "column": column,
        "agg": agg,
        "number_format": number_format,
        "decimals": int(decimals),
        "unit": unit,
    }


def _example_display_kpi():
    df = pl.DataFrame({"segment": ["retail", "sme"], "rwa": [120.0, 80.0]})
    assert display_kpi(df, column="rwa", agg="sum", label="Total RWA")["value"] == 200.0
    assert display_kpi({"gini": 0.42, "n": 1000}, column="gini")["label"] == "gini"


def generate_image(
    df: pl.DataFrame,
    kind: str = "hist",
    x: str = "",
    y: str = "",
    bins: int = 30,
    title: str = "",
    output_dir: str = ".",
    block_id: str = "",
) -> bytes:
    """Draws a matplotlib chart from the `df` input and emits it as PNG
    bytes on the `image` port, also saving it as
    generate_image_<block_id>.png in `output_dir`. `kind` is one of
    "hist" (default; histogram of column `x` with `bins` bins, `y`
    unused), "bar" (x = category labels, y = bar heights -- one bar per
    row, so aggregate first), "scatter", or "line" (both plot `y` against
    `x` in row order). `x` is always required; `y` is required for every
    kind but hist. `title` is optional. `output_dir` and `block_id` are
    engine-injected; leave them unset."""
    # Self-contained imports (rather than relying on this module's own
    # top-level imports) so this function stays a valid, independent unit
    # both when run live by the engine and when inlined verbatim into a
    # compiled script.
    import io
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "hist":
        ax.hist(df[x].to_list(), bins=bins)
    elif kind == "bar":
        ax.bar([str(v) for v in df[x].to_list()], df[y].to_list())
    elif kind == "scatter":
        ax.scatter(df[x].to_list(), df[y].to_list(), s=10, alpha=0.6)
    elif kind == "line":
        ax.plot(df[x].to_list(), df[y].to_list())
    else:
        raise ValueError(f"unknown chart kind: {kind!r}")
    ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"generate_image_{block_id}.png" if block_id else "generate_image.png"
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(buf.getvalue())

    return buf.getvalue()
