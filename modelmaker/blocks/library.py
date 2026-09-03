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


def read_csv(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def _probe_read_csv(params: dict) -> float | None:
    try:
        return os.path.getmtime(params["path"])
    except OSError:
        return None


register_block(
    BlockSpec(
        category="read_csv",
        block_type="input",
        display_name="Read CSV",
        inputs=[],
        outputs=[PortSpec("out")],
        fn=read_csv,
        metadata_transform=infer_dtypes,
        probe=_probe_read_csv,
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


register_block(
    BlockSpec(
        category="write_csv",
        block_type="output",
        display_name="Write CSV",
        inputs=[PortSpec("df")],
        outputs=[],
        fn=write_csv,
        metadata_transform=lambda *_a, **_k: {},
    )
)
