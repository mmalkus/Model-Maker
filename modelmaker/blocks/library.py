"""Standard block library. Each `fn` is a plain function over pl.DataFrame /
literal params only — no DataFramePacket, no ColumnMeta. This is what a
--with-metadata=off compile emits verbatim (see plan section 7); the engine
wraps packets around calls to these at run time, and applies the paired
metadata_transform separately.
"""

from __future__ import annotations

import os
from dataclasses import replace

import polars as pl

from ..metadata_transforms import broadcast_to_all_outputs, infer_dtypes, narrow_from_single_input, passthrough
from ..packet import ColumnMeta, ColumnRole
from .base import BlockSpec, PortSpec, register_block


def read_csv(path: str, sample_rows: int | None = None) -> pl.DataFrame:
    """Reads a CSV file into a dataframe on the `out` port; no inputs.
    `path` is the file path (relative to the working directory the engine
    runs in). Column dtypes are inferred by polars; column roles start
    untagged. `sample_rows` is not a user param -- the engine injects it in
    sample mode to read only the first N rows; leave it unset."""
    # `sample_rows`, when the engine's sample mode is on, is injected by
    # Runner.run_block the same way output_dir/block_id are (see
    # util.accepts_param) -- an input block opts in just by naming the
    # parameter. Scanning with `n_rows` instead of reading the whole file
    # and truncating afterward (the fallback for input blocks that don't
    # accept this param -- see runner.py) is the one place lazy evaluation
    # earns its keep for sample mode: the rest of the pipeline already runs
    # on the truncated, small-by-construction sample, so nothing downstream
    # needs lazy frames to iterate cheaply. Full runs (sample_rows=None)
    # read exactly as before.
    if sample_rows is not None:
        return pl.scan_csv(path, n_rows=sample_rows).collect()
    return pl.read_csv(path)


def _probe_path_mtime(params: dict) -> float | None:
    # Shared "check for changes" probe for every file-based input block
    # (read_csv/read_parquet/read_json/read_excel): cheap, read-only, and
    # good enough to flag "source changed" without touching the cached
    # packet (see plan section 6).
    try:
        return os.path.getmtime(params["path"])
    except OSError:
        return None


def _read_csv_lazy(path: str, sample_rows: int | None = None) -> pl.LazyFrame:
    # The lazy twin of read_csv (see BlockSpec.lazy_fn) -- used only by a
    # streaming run's fusion path (see Runner._dispatch_fused_group), which
    # collects once at the end of a whole fused chain rather than here.
    # Ordinary calls to `read_csv` above are untouched and still always
    # return a materialized pl.DataFrame.
    if sample_rows is not None:
        return pl.scan_csv(path, n_rows=sample_rows)
    return pl.scan_csv(path)


register_block(
    BlockSpec(
        category="read_csv",
        block_type="input",
        display_name="Read CSV",
        inputs=[],
        outputs=[PortSpec("out")],
        fn=read_csv,
        lazy_fn=_read_csv_lazy,
        metadata_transform=infer_dtypes,
        probe=_probe_path_mtime,
    )
)


def read_parquet(path: str, sample_rows: int | None = None) -> pl.DataFrame:
    """Reads a Parquet file into a dataframe on the `out` port; no inputs.
    `path` is the file path. Dtypes come from the Parquet schema.
    `sample_rows` is engine-injected in sample mode (first N rows only);
    leave it unset."""
    if sample_rows is not None:
        return pl.scan_parquet(path).head(sample_rows).collect()
    return pl.read_parquet(path)


def _read_parquet_lazy(path: str, sample_rows: int | None = None) -> pl.LazyFrame:
    # Lazy twin of read_parquet, same reasoning as read_csv's (see
    # BlockSpec.lazy_fn) -- polars has no `n_rows` kwarg on scan_parquet, so
    # sampling goes through a `.head()` the query optimizer pushes down.
    lf = pl.scan_parquet(path)
    return lf.head(sample_rows) if sample_rows is not None else lf


register_block(
    BlockSpec(
        category="read_parquet",
        block_type="input",
        display_name="Read Parquet",
        inputs=[],
        outputs=[PortSpec("out")],
        fn=read_parquet,
        lazy_fn=_read_parquet_lazy,
        metadata_transform=infer_dtypes,
        probe=_probe_path_mtime,
    )
)


def read_json(path: str, sample_rows: int | None = None) -> pl.DataFrame:
    """Reads a JSON file into a dataframe on the `out` port; no inputs.
    `path` ending in .jsonl/.ndjson (case-insensitive) is read as
    newline-delimited JSON (one object per line); anything else must be a
    plain JSON array of records. `sample_rows` is engine-injected in sample
    mode (first N rows only); leave it unset."""
    # Newline-delimited JSON has a real lazy/streaming reader; a plain JSON
    # array does not (polars must load it whole to find its structure), so
    # this block -- unlike read_csv/read_parquet -- has no lazy_fn twin and
    # simply never joins a streaming run's fusion group.
    if path.lower().endswith((".jsonl", ".ndjson")):
        if sample_rows is not None:
            return pl.scan_ndjson(path).head(sample_rows).collect()
        return pl.read_ndjson(path)
    df = pl.read_json(path)
    return df.head(sample_rows) if sample_rows is not None else df


register_block(
    BlockSpec(
        category="read_json",
        block_type="input",
        display_name="Read JSON",
        inputs=[],
        outputs=[PortSpec("out")],
        fn=read_json,
        metadata_transform=infer_dtypes,
        probe=_probe_path_mtime,
    )
)


def read_excel(path: str, sheet: str | None = None, sample_rows: int | None = None) -> pl.DataFrame:
    """Reads one worksheet of an Excel workbook into a dataframe on the
    `out` port; no inputs. `path` is the .xlsx file path; `sheet` is the
    worksheet name -- unset/empty reads the first sheet. The first row is
    taken as the header. `sample_rows` is engine-injected in sample mode
    (the whole sheet is still read, then truncated); leave it unset."""
    df = pl.read_excel(path, sheet_name=sheet) if sheet else pl.read_excel(path)
    return df.head(sample_rows) if sample_rows is not None else df


register_block(
    BlockSpec(
        category="read_excel",
        block_type="input",
        display_name="Read Excel",
        inputs=[],
        outputs=[PortSpec("out")],
        fn=read_excel,
        metadata_transform=infer_dtypes,
        probe=_probe_path_mtime,
    )
)


def read_sql(connection_env: str, query: str, sample_rows: int | None = None) -> pl.DataFrame:
    """Runs a SQL query against a database and returns the result set as a
    dataframe on the `out` port; no inputs. `connection_env` is the *name*
    of an environment variable (e.g. "WAREHOUSE_URI") holding the
    connection URI -- never the URI itself; raises KeyError if that
    variable isn't set. `query` is the SQL text, in the target database's
    own dialect. Optional `probe_query` param (a cheap query such as
    SELECT MAX(updated_at) ...) lets "check for changes" detect new data;
    it isn't passed to this function. `sample_rows` is engine-injected in
    sample mode (the full query still runs, then is truncated client-side);
    leave it unset."""
    # The connection string itself never lives in params -- it's baked as a
    # literal into saved project files and (per compiler.py section 7)
    # compiled scripts, so a raw DB password there would leak into both.
    # `connection_env` is only the *name* of an env var the user sets
    # locally (the project already loads .env via api.py); the secret is
    # read fresh from the environment at run time, live or compiled.
    uri = os.environ[connection_env]
    df = pl.read_database_uri(query, uri)
    # No LIMIT/TOP/FETCH FIRST pushdown for sample_rows -- that syntax
    # differs per SQL dialect (Postgres/MySQL LIMIT vs. SQL Server TOP vs.
    # Oracle FETCH FIRST), so this truncates client-side instead. Costs a
    # full round trip in sample mode; keeps this block dialect-agnostic.
    return df.head(sample_rows) if sample_rows is not None else df


def _probe_sql(params: dict) -> str | None:
    # Optional: only runs if the user gave a cheap probe_query (e.g. a
    # COUNT(*) or a MAX(updated_at)) -- there's no generic mtime/checksum
    # equivalent for a database table. No probe_query means no probe;
    # Runner.check_for_changes already treats spec.probe returning None as
    # "never flags changed", which is the right default here.
    query = params.get("probe_query")
    if not query:
        return None
    try:
        uri = os.environ[params["connection_env"]]
        result = pl.read_database_uri(query, uri)
    except Exception:
        return None
    return str(result.row(0)) if result.height else ""


register_block(
    BlockSpec(
        category="read_sql",
        block_type="input",
        display_name="Read SQL",
        inputs=[],
        outputs=[PortSpec("out")],
        fn=read_sql,
        metadata_transform=infer_dtypes,
        probe=_probe_sql,
    )
)


def filter_rows(df: pl.DataFrame, expr: str) -> pl.DataFrame:
    """Keeps only the rows of `df` for which `expr` is true, emitted on
    `out` with the same columns and roles. `expr` is a Polars SQL boolean
    expression over column names, e.g. "age >= 18 AND region = 'EU'" or
    "balance IS NOT NULL" (string literals in single quotes)."""
    return df.filter(pl.sql_expr(expr))


register_block(
    BlockSpec(
        category="filter",
        block_type="standard",
        display_name="Filter",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=filter_rows,
        # filter_rows is pure expression-based code (`.filter(pl.sql_expr(...))`),
        # identical over pl.DataFrame/pl.LazyFrame -- reused verbatim as the
        # lazy_fn a streaming run's fusion path calls (see BlockSpec.lazy_fn).
        lazy_fn=filter_rows,
        metadata_transform=passthrough,
    )
)


def select_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Keeps only the listed columns of `df`, in the order given by `cols`,
    emitted on `out`; kept columns retain their roles. Every name in `cols`
    must exist in the input."""
    return df.select(cols)


register_block(
    BlockSpec(
        category="select",
        block_type="standard",
        display_name="Select columns",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=select_cols,
        lazy_fn=select_cols,  # same reasoning as filter's lazy_fn above
        metadata_transform=narrow_from_single_input,
    )
)


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


def _groupby_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    by = set(params.get("by", []))
    result = {}
    for name in df.columns:
        if name in by and name in in_meta:
            result[name] = in_meta[name]
        else:
            result[name] = ColumnMeta(dtype=str(df.schema[name]))
    return {"out": result}


register_block(
    BlockSpec(
        category="groupby_agg",
        block_type="standard",
        display_name="Group by / aggregate",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=groupby_agg,
        lazy_fn=groupby_agg,  # same reasoning as filter's lazy_fn above
        metadata_transform=_groupby_meta,
    )
)


def join(left: pl.DataFrame, right: pl.DataFrame, on: list[str], how: str = "inner") -> pl.DataFrame:
    """Joins the `left` and `right` input dataframes on the key columns in
    `on` (same names on both sides), emitted on `out`. `how` is a polars
    join strategy: "inner" (default), "left", "right", "full", "semi", or
    "anti" ("cross" is not usable: polars rejects join keys with it).
    Non-key columns present on both sides get a "_right" suffix on the
    right-hand copy; a "full" join also keeps the right-hand keys as
    "<key>_right". Where a column exists on both sides, the left side's
    role wins."""
    return left.join(right, on=on, how=how)


def _join_meta(input_metas, outputs, params):
    left_meta = input_metas["left"]
    right_meta = input_metas["right"]
    (df,) = outputs.values()
    merged = {**right_meta, **left_meta}
    return {"out": {name: merged[name] for name in df.columns if name in merged}}


register_block(
    BlockSpec(
        category="join",
        block_type="standard",
        display_name="Join",
        inputs=[PortSpec("left"), PortSpec("right")],
        outputs=[PortSpec("out")],
        fn=join,
        lazy_fn=join,  # same reasoning as filter's lazy_fn above
        metadata_transform=_join_meta,
    )
)


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


register_block(
    BlockSpec(
        category="train_test_split",
        block_type="standard",
        display_name="Train/test split",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("train"), PortSpec("test")],
        fn=train_test_split,
        metadata_transform=broadcast_to_all_outputs,
    )
)


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


register_block(
    BlockSpec(
        category="time_split",
        block_type="standard",
        display_name="Out-of-time split",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("development"), PortSpec("out_of_time")],
        fn=time_split,
        metadata_transform=broadcast_to_all_outputs,
    )
)


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


def _derive_columns_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (out,) = outputs.values()
    result = {}
    for name in out.columns:
        if name in in_meta:
            result[name] = replace(in_meta[name], dtype=str(out.schema[name]))
        else:
            result[name] = ColumnMeta(dtype=str(out.schema[name]), role=ColumnRole.FEATURE)
    return {"out": result}


register_block(
    BlockSpec(
        category="derive_columns",
        block_type="standard",
        display_name="Derive columns",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=derive_columns,
        lazy_fn=derive_columns,  # pure expressions, same over a LazyFrame
        metadata_transform=_derive_columns_meta,
    )
)


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


def _one_hot_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (out,) = outputs.values()
    result = {}
    for name in out.columns:
        result[name] = in_meta[name] if name in in_meta else ColumnMeta(dtype=str(out.schema[name]), role=ColumnRole.FEATURE)
    return {"out": result}


register_block(
    BlockSpec(
        category="one_hot_encode",
        block_type="standard",
        display_name="One-hot encode",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=one_hot_encode,
        metadata_transform=_one_hot_meta,
    )
)


def write_csv(df: pl.DataFrame, filename: str, output_dir: str = ".") -> None:
    """Writes the `df` input to a CSV file; no output ports. `filename` is
    the file name (e.g. "scored.csv"), written inside `output_dir`, which
    the engine injects as the run's output directory -- leave it unset.
    Overwrites any existing file of the same name."""
    df.write_csv(os.path.join(output_dir, filename))


def _write_csv_sink(df: pl.LazyFrame, filename: str, output_dir: str = ".") -> None:
    # The streaming-sink twin of write_csv (see BlockSpec.lazy_sink_fn):
    # called only when this block is folded onto the end of a streaming
    # run's fusion group as its terminal write (see
    # Runner._build_fusion_groups), so the group's whole upstream plan --
    # source scan included -- is written straight to disk via the
    # streaming engine without ever materializing a cached DataFrame for
    # the block that feeds it.
    df.sink_csv(os.path.join(output_dir, filename))


register_block(
    BlockSpec(
        category="write_csv",
        block_type="output",
        display_name="Write CSV",
        inputs=[PortSpec("df")],
        outputs=[],
        fn=write_csv,
        lazy_sink_fn=_write_csv_sink,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def display_table(df: pl.DataFrame) -> pl.DataFrame:
    """A no-op pass-through: exists to mark a point in the pipeline as a
    reportable table. The engine/UI can preview it like any dataframe
    output; the compiled script carries it through unchanged."""
    return df


register_block(
    BlockSpec(
        category="display_table",
        block_type="output",
        display_name="Display table",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=display_table,
        metadata_transform=passthrough,
    )
)


def display_value(value):
    """A no-op pass-through, like display_table, but for any port type --
    a dataframe, a model/scalar_metric artifact (a plain dict), or an image.
    Its ports are typed "any" so it wires up to whatever you point it at;
    the UI figures out how to render whatever actually comes through."""
    return value


def _display_value_meta(input_metas, outputs, params):
    # Only reached when the value passed through actually was a DataFrame
    # (see runner.py: the transform only runs when an output is one) --
    # reuse the real upstream column metadata, same as display_table.
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    return {"value": {name: in_meta[name] if name in in_meta else ColumnMeta(dtype=str(df.schema[name])) for name in df.columns}}


register_block(
    BlockSpec(
        category="display_value",
        block_type="output",
        display_name="View value",
        inputs=[PortSpec("value", type="any")],
        outputs=[PortSpec("value", type="any")],
        fn=display_value,
        metadata_transform=_display_value_meta,
    )
)


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


register_block(
    BlockSpec(
        category="generate_image",
        block_type="output",
        display_name="Generate image",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("image", type="image")],
        fn=generate_image,
        metadata_transform=lambda *_a, **_k: {},
    )
)
