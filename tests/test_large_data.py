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


# ---- Stage CSV as Parquet ----------------------------------------------------


def _zip(path, members):
    import zipfile

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)


def _part_mtimes(folder):
    return {p.name: p.stat().st_mtime_ns for p in folder.glob("*.parquet")}


def test_stage_zip_members_into_parquet_parts(tmp_path):
    from modelmaker.blocks.library import stage_csv_parquet

    src = tmp_path / "perf.zip"
    _zip(src, {"2020Q1.csv": "1|012020|0\n2|012020|1\n", "2020Q2.csv": "1|042020|0\n", "readme.pdf": "x"})
    df = stage_csv_parquet(str(src), separator="|", has_header=False, column_names="id,period,dlq", infer_schema_rows=0).collect()
    assert df.to_dicts()[0] == {"id": "1", "period": "012020", "dlq": "0"}  # leading zeros kept as text
    assert df.height == 3
    staged = tmp_path / "perf_parquet"
    assert set(_part_mtimes(staged)) == {"2020Q1.parquet", "2020Q2.parquet"}  # text members only
    # Read straight from the zip: nothing but the parts and the manifest.
    assert {p.name for p in staged.iterdir()} == {"2020Q1.parquet", "2020Q2.parquet", "_manifest.json"}


def test_stage_reads_a_zip_member_in_blocks_without_unpacking(tmp_path, monkeypatch):
    import builtins

    from modelmaker.blocks import library

    # Tiny blocks, so a member spans many of them; one quoted field holds
    # a line break and must not be split.
    monkeypatch.setattr(library, "_STAGE_CHUNK_BYTES", 40)
    rows = [f'{i},"note {i}"' for i in range(200)]
    rows[57] = '57,"two\nlines"'
    src = tmp_path / "big.zip"
    _zip(src, {"big.csv": "id,note\n" + "\n".join(rows) + "\n"})

    written = []
    real_open = builtins.open

    def _spy(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+"):
            written.append(str(file))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", _spy)
    df = library.stage_csv_parquet(str(src)).collect()
    assert df.height == 200 and df["id"].to_list() == list(range(200))
    assert df["note"][57] == "two\nlines" and df["note"][199] == "note 199"
    # The only files Python opened for writing are the manifest's.
    assert all(w.endswith((".json", ".json.tmp")) for w in written), written


def test_stage_skips_unchanged_parts_and_rebuilds_changed_ones(tmp_path):
    from modelmaker.blocks.library import stage_csv_parquet

    src = tmp_path / "perf.zip"
    _zip(src, {"q1.csv": "a,b\n1,x\n", "q2.csv": "a,b\n2,y\n", "q3.csv": "a,b\n3,z\n"})
    stage_csv_parquet(str(src))
    before = _part_mtimes(tmp_path / "perf_parquet")

    # Same content: nothing rebuilt.
    stage_csv_parquet(str(src))
    assert _part_mtimes(tmp_path / "perf_parquet") == before

    # One member changed, one dropped: only that part is rebuilt, the
    # dropped one's part removed.
    _zip(src, {"q1.csv": "a,b\n1,x\n", "q2.csv": "a,b\n2,changed\n"})
    out = stage_csv_parquet(str(src)).collect()
    after = _part_mtimes(tmp_path / "perf_parquet")
    assert set(after) == {"q1.parquet", "q2.parquet"}
    assert after["q1.parquet"] == before["q1.parquet"] and after["q2.parquet"] != before["q2.parquet"]
    assert out["b"].to_list() == ["x", "changed"]

    # Different read options: everything is rebuilt.
    stage_csv_parquet(str(src), compression="snappy")
    assert all(_part_mtimes(tmp_path / "perf_parquet")[p] != after[p] for p in after)


def test_stage_plain_and_gzipped_files_by_hash(tmp_path):
    import gzip
    import os

    from modelmaker.blocks.library import stage_csv_parquet

    csv = tmp_path / "loans.csv"
    csv.write_text("id|amt\n1|10\n2|20\n")
    out_dir = tmp_path / "staged"
    assert stage_csv_parquet(str(csv), separator="|", parquet_dir=str(out_dir)).collect()["amt"].to_list() == [10, 20]
    before = _part_mtimes(out_dir)
    # Touched but identical: the hash matches, no rebuild.
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    stage_csv_parquet(str(csv), separator="|", parquet_dir=str(out_dir))
    assert _part_mtimes(out_dir) == before
    # Real change: rebuilt.
    csv.write_text("id|amt\n1|10\n2|20\n3|30\n")
    assert stage_csv_parquet(str(csv), separator="|", parquet_dir=str(out_dir)).collect().height == 3

    gz = tmp_path / "perf.txt.gz"
    with gzip.open(gz, "wt") as f:
        f.write("a\tb\n1\t2\n")
    assert stage_csv_parquet(str(gz), separator="tab").collect().to_dicts() == [{"a": 1, "b": 2}]
    assert (tmp_path / "perf_parquet" / "perf.parquet").exists()


def test_stage_parts_share_the_first_parts_schema(tmp_path):
    from modelmaker.blocks.library import stage_csv_parquet

    src = tmp_path / "p.zip"
    # The second member alone would infer `b` as text.
    _zip(src, {"1.csv": "a,b\n1,2\n", "2.csv": "a,b\n3,oops\n"})
    with pytest.raises(ValueError, match="infer_schema_rows to 0"):
        stage_csv_parquet(str(src))
    df = stage_csv_parquet(str(src), infer_schema_rows=0).collect()
    assert df.schema == {"a": pl.String, "b": pl.String} and df.height == 2


def test_stage_block_runs_in_the_engine_and_streams(tmp_path):
    src = tmp_path / "perf.zip"
    _zip(src, {"q1.csv": "id|v\n1|5\n2|15\n", "q2.csv": "id|v\n3|25\n"})
    graph = Graph(
        blocks={
            "s": make_block("s", "stage_csv_parquet", params={"path": str(src), "separator": "|"}),
            "f": make_block("f", "filter", params={"expr": "v > 10"}, x=1),
        },
        wires={"w1": Wire("w1", "s", "out", "f", "df")},
    )
    runner = Runner(graph, CacheStore())
    report = runner.run_all_streaming()
    assert report == {"s": "fused", "f": "green"}, (report, runner.state["f"].last_error)
    out = runner.cache.get(runner.state["f"].last_successful_key).outputs["out"].data
    assert out["id"].to_list() == [2, 3]


def test_stage_block_compiles(tmp_path):
    from modelmaker.compiler import compile_graph

    src = tmp_path / "loans.csv"
    src.write_text("a,b\n1,x\n")
    graph = Graph(blocks={"s": make_block("s", "stage_csv_parquet", params={"path": str(src)})}, wires={})
    runner = Runner(graph, CacheStore())
    assert runner.refresh("s") == "green"
    source = compile_graph(graph, runner=runner)
    ns: dict = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert ns["s"].to_dicts() == [{"a": 1, "b": "x"}]


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
