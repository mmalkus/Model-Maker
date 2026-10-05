"""The column profile (agent/profile.py): a pure-code classification of a
dataframe's columns for prompts -- and what it must never reveal."""

from __future__ import annotations

from datetime import date

import polars as pl

from modelmaker.agent.profile import describe, has_text_dates, profile
from modelmaker.packet import ColumnMeta, ColumnRole, DataFramePacket


def _packet(df: pl.DataFrame, roles: dict[str, ColumnRole]) -> DataFramePacket:
    meta = {c: ColumnMeta(str(df.schema[c]), roles.get(c, ColumnRole.FEATURE)) for c in df.columns}
    return DataFramePacket(df, meta)


def _frame(n: int = 200) -> pl.DataFrame:
    return pl.DataFrame({
        "cust": [f"C{i:05d}" for i in range(n)],
        "bad": [1 if i % 10 == 0 else 0 for i in range(n)],
        "opened": [f"{1 + i % 28:02d}/0{1 + i % 9}/2023" for i in range(n)],
        "seen": [date(2022, 1, 1 + i % 28) for i in range(n)],
        "income": [1000.0 + i for i in range(n)],
        "late": [i % 4 for i in range(n)],
        "segment": ["retail" if i % 2 else "sme" for i in range(n)],
        "city": ["Amsterdam", "Utrecht", "Delft", "Leiden", "Breda", "Gouda", "Ede"][:7] * (n // 7) + ["Ede"] * (n % 7),
        "note": [f"free text {i}" for i in range(n)],
        "flat": [5] * n,
        "gone": [None] * n,
        "leaky": [0.5] * (n - 1) + [0.7],
    })


def test_profile_classifies_each_kind_of_column():
    roles = {"cust": ColumnRole.ID, "bad": ColumnRole.TARGET, "leaky": ColumnRole.EXCLUDED}
    cols = {c["name"]: c for c in profile(_packet(_frame(), roles))}
    assert cols["cust"]["kind"] == "id" and cols["leaky"]["kind"] == "excluded"
    assert cols["bad"]["kind"] == "binary" and cols["bad"]["rate_of_1"] == 0.1
    assert cols["opened"]["kind"] == "text date" and cols["opened"]["min"] == "2023-01-01"
    assert cols["seen"]["kind"] == "date" and cols["seen"]["min"] == "2022-01-01"
    assert cols["income"]["kind"] == "numeric" and cols["income"]["min"] == 1000.0
    assert cols["late"]["kind"] == "discrete" and cols["late"]["distinct"] == 4
    assert cols["segment"]["kind"] == "binary" and cols["segment"]["segment_candidate"]
    assert cols["city"]["kind"] == "categorical" and cols["city"]["levels"] == 7
    assert cols["note"]["kind"] == "id-like text"
    assert cols["flat"]["kind"] == "constant" and cols["gone"]["kind"] == "empty"
    assert has_text_dates(list(cols.values()))


def test_the_description_never_shows_values_out_of_the_data():
    roles = {"cust": ColumnRole.ID, "bad": ColumnRole.TARGET, "leaky": ColumnRole.EXCLUDED}
    text = describe(profile(_packet(_frame(), roles)), 200)
    assert "200 rows" in text and "bad (binary, 2 levels, rate of 1 = 10.00%)" in text
    for value in ("C00000", "C00199", "Amsterdam", "retail", "sme", "free text", "/2023"):  # text dates show as ISO ranges
        assert value not in text, value
    assert "seen 2022-01-01..2022-01-28 (75% of rows before 2022-01-" in text  # real dates show their range
