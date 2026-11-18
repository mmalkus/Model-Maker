"""Data plumbing for sources bigger than memory: Read CSV's parsing options,
Write Parquet as a streaming sink, dataset reads joining a streaming run,
sample mode keeping whole IDs, and the cache's in-memory budget."""

from __future__ import annotations

import polars as pl
import pytest

from modelmaker.blocks.library import read_csv
from modelmaker.cache import CacheStore, entry_size, memory_budget_bytes
from modelmaker.datasets import DataAccess
from modelmaker.graph import Graph, Wire
from modelmaker.packet import ColumnRole, DataFramePacket
from modelmaker.run_worker import _collect
from modelmaker.runner import Runner

from .helpers import make_block
from .test_datasets import FakeDatasets

# ---- Read CSV ---------------------------------------------------------------


def test_read_csv_pipe_separated_without_header(tmp_path):
    path = tmp_path / "perf.txt"
    path.write_text("1|201901|0\n1|201902|1\n2|201901|0\n")
    df = read_csv(str(path), separator="|", has_header=False, column_names="loan_id, period, dlq").collect()
    assert df.columns == ["loan_id", "period", "dlq"]
    assert df["period"].to_list() == [201901, 201902, 201901]


def test_read_csv_tab_and_all_text(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("a\tb\n1\t2\n")
    df = read_csv(str(path), separator="\\t", infer_schema_rows=0).collect()
    assert df.schema == {"a": pl.String, "b": pl.String}


def test_read_csv_defaults_unchanged(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,x\n")
    assert read_csv(str(path)).collect().to_dicts() == [{"a": 1, "b": "x"}]


def test_read_csv_rejects_a_long_separator(tmp_path):
    with pytest.raises(ValueError, match="single character"):
        read_csv(str(tmp_path / "x.csv"), separator="||")


# ---- Write Parquet ----------------------------------------------------------


def test_csv_to_parquet_streams_straight_to_disk(tmp_path):
    src = tmp_path / "big.csv"
    src.write_text("a|b\n" + "".join(f"{i}|{i % 3}\n" for i in range(1000)))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    graph = Graph(
        blocks={
            "r": make_block("r", "read_csv", params={"path": str(src), "separator": "|"}),
            "w": make_block("w", "write_parquet", params={"filename": "big.parquet"}, x=1),
        },
        wires={"w1": Wire("w1", "r", "out", "w", "df")},
    )
    runner = Runner(graph, CacheStore(), output_dir=str(out_dir))
    report = runner.run_all_streaming()
    # The read is fused and the write folded on as a sink: nothing between
    # them is collected or cached.
    assert report == {"r": "fused", "w": "green"}, (report, runner.state["w"].last_error)
    written = pl.read_parquet(out_dir / "big.parquet")
    assert written.height == 1000 and written.columns == ["a", "b"]


def test_write_parquet_checks_compression(tmp_path):
    from modelmaker.blocks.library import write_parquet

    with pytest.raises(ValueError, match="compression"):
        write_parquet(pl.DataFrame({"a": [1]}), "x.parquet", compression="brotli9", output_dir=str(tmp_path))


# ---- dataset reads in a streaming run -----------------------------------------


def test_read_dataset_fuses_into_a_streaming_run(tmp_path):
    (tmp_path / "ds").mkdir()
    datasets = FakeDatasets(tmp_path / "ds")
    datasets.put("loans", b"id,amount\n1,10\n2,20\n3,30\n")
    graph = Graph(
        blocks={
            "r": make_block("r", "read_dataset", params={"dataset": "loans"}),
            "f": make_block("f", "filter", params={"expr": "amount > 10"}, x=1),
            "s": make_block("s", "select", params={"cols": ["id"]}, x=2),
        },
        wires={"w1": Wire("w1", "r", "out", "f", "df"), "w2": Wire("w2", "f", "out", "s", "df")},
    )
    runner = Runner(graph, CacheStore(), data=DataAccess.make(datasets=datasets))
    report = runner.run_all_streaming()
    assert report == {"r": "fused", "f": "fused", "s": "green"}, (report, runner.state["s"].last_error)
    out = runner.cache.get(runner.state["s"].last_successful_key).outputs["out"].data
    assert out["id"].to_list() == [2, 3]


def test_a_missing_dataset_fails_the_fused_group(tmp_path):
    (tmp_path / "ds").mkdir()
    datasets = FakeDatasets(tmp_path / "ds")
    graph = Graph(
        blocks={
            "r": make_block("r", "read_dataset", params={"dataset": "nope"}),
            "f": make_block("f", "filter", params={"expr": "amount > 10"}, x=1),
        },
        wires={"w1": Wire("w1", "r", "out", "f", "df")},
    )
    runner = Runner(graph, CacheStore(), data=DataAccess.make(datasets=datasets))
    report = runner.run_all_streaming()
    assert report["f"] == "red"
    assert "no dataset named 'nope'" in runner.state["f"].last_error


# ---- sample mode by ID ----------------------------------------------------------


def _panel(tmp_path, loans=200, months=24):
    """A monthly panel sorted by loan then month, as loan performance
    files are."""
    path = tmp_path / "panel.csv"
    rows = "".join(f"{loan},{m}\n" for loan in range(loans) for m in range(months))
    path.write_text("loan_id,month\n" + rows)
    return path


def test_sample_by_key_keeps_whole_histories(tmp_path):
    lf = pl.scan_csv(_panel(tmp_path))
    df = _collect(lf, {"rows": 960, "key": "loan_id"})
    per_loan = df.group_by("loan_id").len()
    assert per_loan["len"].unique().to_list() == [24]  # every kept loan complete
    # Roughly the asked-for size, and spread over the file, not its head.
    assert 400 <= df.height <= 2000
    assert df["loan_id"].max() > 100
    # Deterministic.
    assert df.equals(_collect(pl.scan_csv(_panel(tmp_path)), {"rows": 960, "key": "loan_id"}))


def test_sample_by_key_small_source_and_missing_key(tmp_path):
    lf = pl.scan_csv(_panel(tmp_path, loans=2, months=3))
    assert _collect(lf, {"rows": 100, "key": "loan_id"}).height == 6
    # A key the output doesn't have: the first N rows, as before.
    assert _collect(lf, {"rows": 4, "key": "nope"}).height == 4
    assert _collect(lf, 4).height == 4


def test_sample_mode_uses_the_tagged_id(tmp_path):
    path = _panel(tmp_path)
    read = make_block("r", "read_csv", params={"path": str(path)})
    runner = Runner(Graph(blocks={"r": read}, wires={}), CacheStore(), sample_rows=960)
    assert runner.refresh("r") == "green"
    first_n = runner.cache.get(runner.state["r"].last_successful_key).outputs["out"].data
    assert first_n["loan_id"].n_unique() == 40  # the first 960 rows: 40 loans

    read.column_role_overrides = {"loan_id": ColumnRole.ID.value}
    # Sampling by ID is part of the key, so it never reuses a first-N cut.
    assert runner.state["r"].last_successful_basis.get("sample_by") is None
    assert runner._basis("r")["sample_by"] == "loan_id"
    assert runner.refresh("r") == "green"
    by_id = runner.cache.get(runner.state["r"].last_successful_key).outputs["out"].data
    assert by_id.group_by("loan_id").len()["len"].unique().to_list() == [24]
    assert by_id["loan_id"].max() > 100


# ---- the cache's memory budget ---------------------------------------------------


def _packet(n):
    return {"out": DataFramePacket(data=pl.DataFrame({"x": list(range(n))}), schema_meta={})}


def test_cache_evicts_least_recently_used_to_disk(tmp_path):
    one = entry_size(type("E", (), {"outputs": _packet(1000)})())
    store = CacheStore(tmp_path, memory_budget=int(one * 2.5))
    keys = [f"{i:016x}" for i in range(4)]
    for k in keys[:3]:
        store.set(k, _packet(1000))
    store.get(keys[0])  # now the most recently used
    store.set(keys[3], _packet(1000))
    assert store.memory_bytes <= store.memory_budget
    held = set(store._mem)
    assert keys[3] in held and keys[0] in held
    assert keys[1] not in held  # least recently used went first
    # An evicted entry comes back from disk, intact.
    assert store.get(keys[1]).outputs["out"].data["x"].to_list() == list(range(1000))


def test_cache_never_evicts_what_only_memory_holds():
    store = CacheStore(None, memory_budget=1)
    for i in range(3):
        store.set(f"{i:016x}", _packet(100))
    assert len(store._mem) == 3
    assert all(store.get(f"{i:016x}") is not None for i in range(3))


def test_cache_memory_budget_from_the_environment(monkeypatch):
    monkeypatch.setenv("MODELMAKER_CACHE_MEMORY_MB", "512")
    assert memory_budget_bytes() == 512 * 1024**2
    monkeypatch.setenv("MODELMAKER_CACHE_MEMORY_MB", "0")
    assert memory_budget_bytes() is None
    monkeypatch.delenv("MODELMAKER_CACHE_MEMORY_MB")
    default = memory_budget_bytes()
    assert default is None or default >= 256 * 1024**2
