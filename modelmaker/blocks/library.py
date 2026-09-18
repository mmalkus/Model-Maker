"""Standard block library. Each `fn` is a plain function over pl.DataFrame /
literal params only — no DataFramePacket, no ColumnMeta. This is what a
--with-metadata=off compile emits verbatim (see plan section 7); the engine
wraps packets around calls to these at run time, and applies the paired
metadata_transform separately.
"""

from __future__ import annotations

import os

import polars as pl

from ..metadata_transforms import broadcast_to_all_outputs, infer_dtypes, narrow_from_single_input, passthrough
from ..packet import ColumnMeta
from .base import BlockSpec, PortSpec, register_block


def read_csv(path: str, sample_rows: int | None = None) -> pl.DataFrame:
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


def filter_rows(df: pl.DataFrame, expr: str) -> pl.DataFrame:
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


def groupby_agg(df: pl.DataFrame, by: list[str], aggs: dict[str, str]) -> pl.DataFrame:
    agg_exprs = [getattr(pl.col(c), fn)().alias(c) for c, fn in aggs.items()]
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


def train_test_split(df: pl.DataFrame, test_size: float = 0.2, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n_test = int(len(shuffled) * test_size)
    test = shuffled.head(n_test)
    train = shuffled.tail(len(shuffled) - n_test)
    return train, test


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


def write_csv(df: pl.DataFrame, filename: str, output_dir: str = ".") -> None:
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
