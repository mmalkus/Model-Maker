import polars as pl

from modelmaker.packet import ColumnMeta, ColumnRole, DataFramePacket


def test_with_lineage_appends_and_resets_summary():
    df = pl.DataFrame({"a": [1, 2, 3]})
    p = DataFramePacket(data=df, schema_meta={"a": ColumnMeta(dtype="Int64")})
    p2 = p.with_lineage("b_001").compute_summary()
    p3 = p2.with_lineage("b_002")

    assert p3.lineage == ["b_001", "b_002"]
    assert p2.lineage == ["b_001"]
    assert p.lineage == []
    assert p3.summary is None
    assert p2.summary is not None


def test_compute_summary_numeric_and_non_numeric():
    df = pl.DataFrame({"amount": [1.0, 2.0, None], "name": ["a", "b", "b"]})
    p = DataFramePacket(
        data=df,
        schema_meta={
            "amount": ColumnMeta(dtype="Float64", role=ColumnRole.FEATURE),
            "name": ColumnMeta(dtype="String"),
        },
    )
    p = p.compute_summary()

    assert p.summary["amount"].count == 3
    assert p.summary["amount"].null_count == 1
    assert p.summary["amount"].mean == 1.5
    assert p.summary["name"].mean is None
    assert p.summary["name"].n_unique == 2


def test_compute_summary_is_cached_until_forced():
    df = pl.DataFrame({"a": [1, 2, 3]})
    p = DataFramePacket(data=df, schema_meta={}).compute_summary()
    first = p.summary
    p2 = p.compute_summary()
    assert p2.summary is first
    p3 = p.compute_summary(force=True)
    assert p3.summary is not first
