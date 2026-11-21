"""Data plumbing for sources bigger than memory: Read CSV's parsing options,
Write Parquet as a streaming sink, dataset reads joining a streaming run,
sample mode keeping whole IDs, and the cache's in-memory budget."""

from __future__ import annotations

import sys

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


# ---- staging a folder or glob, a slim stage ------------------------------------


def test_stage_a_folder_of_zips_with_only_some_columns(tmp_path):
    from modelmaker.blocks.library import stage_csv_parquet

    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "2016Q1.zip", {"2016Q1.csv": "1|012016|0|x\n2|012016|1|y\n"})
    _zip(raw / "2016Q2.zip", {"2016Q2.csv": "3|042016|0|z\n"})
    (raw / "notes.pdf").write_text("not data")
    df = stage_csv_parquet(
        str(raw), separator="|", has_header=False, column_names="id,period,dlq,junk", infer_schema_rows=0, columns="id, dlq"
    ).collect()
    assert df.columns == ["id", "dlq"] and df["id"].to_list() == ["1", "2", "3"]
    staged = tmp_path / "raw_parquet"
    assert set(_part_mtimes(staged)) == {"2016Q1.parquet", "2016Q2.parquet"}  # named like the zips, no prefix

    # A glob, and a zip whose member is named differently: prefixed with the zip.
    _zip(raw / "2016Q3.zip", {"perf.csv": "4|072016|0|w\n"})
    out = stage_csv_parquet(str(raw / "2016Q*.zip"), separator="|", has_header=False, column_names="id,period,dlq,junk", infer_schema_rows=0)
    assert out.collect().height == 4
    assert "2016Q3__perf.parquet" in _part_mtimes(tmp_path / "raw" / "staged_parquet")


def test_stage_picks_header_less_columns_by_position(tmp_path):
    # A wide file without a header: name only the fields kept (#97).
    from modelmaker.blocks.library import stage_csv_parquet

    raw = tmp_path / "raw"
    raw.mkdir()
    _zip(raw / "2016Q1.zip", {"2016Q1.csv": "a|1|012016|0|x|y\nb|2|012016|1|x|y\n"})
    (raw / "2016Q2.csv").write_text("c|3|042016|0|x|y\n")
    df = stage_csv_parquet(str(raw), separator="|", has_header=False, infer_schema_rows=0, columns="4:dlq, 2:loan_id").collect()
    assert df.columns == ["loan_id", "dlq"] and df.rows() == [("1", "0"), ("2", "1"), ("3", "0")]
    for bad, message in [
        ("2:loan_id, period", "position:name"),
        ("2:a, 2:b", "picked twice"),
        ("0:a", "position:name"),
    ]:
        with pytest.raises(ValueError, match=message):
            stage_csv_parquet(str(raw), separator="|", has_header=False, columns=bad)
    with pytest.raises(ValueError, match="leave column_names empty"):
        stage_csv_parquet(str(raw), separator="|", has_header=False, column_names="x,y", columns="1:a")

    # Types inferred by the first part; later parts are cast to them.
    typed = tmp_path / "typed"
    typed.mkdir()
    (typed / "a.csv").write_text("a|1|2.5|x\nb|2|3.5|y\n")
    _zip(typed / "b.zip", {"b.csv": "c|3|4.5|z\n"})
    df = stage_csv_parquet(str(typed), separator="|", has_header=False, columns="3:rate, 2:loan_id").collect()
    assert df.schema == {"loan_id": pl.Int64, "rate": pl.Float64}
    assert sorted(df.rows()) == [(1, 2.5), (2, 3.5), (3, 4.5)]


def test_stage_columns_are_header_names_when_the_file_has_a_header(tmp_path):
    # A header name may contain a colon -- even one that looks like a
    # position: with a header, columns are always names (#97).
    from modelmaker.blocks.library import stage_csv_parquet

    src = tmp_path / "wide.csv"
    src.write_text("id,time:stamp,1:x,other\n1,t1,a,o\n2,t2,b,o\n")
    df = stage_csv_parquet(str(src), columns="time:stamp, 1:x").collect()
    assert df.columns == ["time:stamp", "1:x"] and df.rows() == [("t1", "a"), ("t2", "b")]


def test_stage_builds_parts_side_by_side(tmp_path, monkeypatch):
    # Several files stage at once (#97), with the same result as one by one.
    import threading

    from modelmaker.blocks import library

    raw = tmp_path / "raw"
    raw.mkdir()
    for q in range(1, 7):
        _zip(raw / f"2016Q{q}.zip", {f"2016Q{q}.csv": "".join(f"{q}{i}|0{q}2016|{i % 2}\n" for i in range(50))})
    real_build = library._stage_build
    # The first part builds alone (it sets the schema); the other five then
    # go three at a time. Each of the first three waits until all three are
    # running, so the test fails (barrier timeout) unless they overlap.
    barrier = threading.Barrier(3, timeout=10)
    calls = 0
    lock = threading.Lock()

    def _counting_build(*a, **k):
        nonlocal calls
        with lock:
            calls += 1
            n = calls
        if 2 <= n <= 4:
            barrier.wait()
        return real_build(*a, **k)

    monkeypatch.setattr(library, "_stage_build", _counting_build)
    opts = dict(separator="|", has_header=False, column_names="id,period,dlq", infer_schema_rows=0)
    parallel = library.stage_csv_parquet(str(raw), workers=3, **opts).collect()
    assert calls == 6
    serial = library.stage_csv_parquet(str(raw), workers=1, parquet_dir=str(tmp_path / "serial"), **opts).collect()
    assert parallel.equals(serial) and parallel.height == 300
    manifest = (raw.parent / "raw_parquet" / "_manifest.json").read_text()
    assert all(f"2016Q{q}.parquet" in manifest for q in range(1, 7))
    assert library._stage_worker_count(0, 1) == 1 and 1 <= library._stage_worker_count(0, 40) <= 4
    assert library._stage_worker_count(8, 3) == 3

    # A re-run with workers: only the changed file is rebuilt, against the
    # schema of a part that is already current.
    before = _part_mtimes(raw.parent / "raw_parquet")
    _zip(raw / "2016Q4.zip", {"2016Q4.csv": "".join(f"9{i}|042016|1\n" for i in range(50))})
    again = library.stage_csv_parquet(str(raw), workers=3, **opts).collect()
    after = _part_mtimes(raw.parent / "raw_parquet")
    assert [p for p in after if after[p] != before[p]] == ["2016Q4.parquet"]
    assert again.height == 300 and again.filter(pl.col("dlq") == "1").height == 25 * 5 + 50


def test_stage_worker_count_follows_free_memory(monkeypatch):
    from modelmaker import run_worker
    from modelmaker.blocks import library

    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr(run_worker, "available_memory_bytes", lambda: 3 * 1024**3)  # room for one at 2 GB
    assert library._stage_worker_count(0, 10) == 1
    monkeypatch.setattr(run_worker, "available_memory_bytes", lambda: 5 * 1024**3)
    assert library._stage_worker_count(0, 10) == 2
    monkeypatch.setattr(run_worker, "available_memory_bytes", lambda: None)  # unknown: cores and the cap only
    assert library._stage_worker_count(0, 10) == 4
    assert library._stage_worker_count(2, 10) == 2  # asked for: no heuristic


def test_stage_failure_with_workers_keeps_the_parts_that_finished(tmp_path, monkeypatch):
    # One part fails while others stage beside it: the error is raised, and
    # every part that did finish is in the manifest, so a re-run builds only
    # the failed one and leaves no file the manifest doesn't know (#97).
    import json

    from modelmaker.blocks import library

    raw = tmp_path / "raw"
    raw.mkdir()
    for q in range(1, 6):
        _zip(raw / f"2016Q{q}.zip", {f"2016Q{q}.csv": f"{q}|0{q}2016|0\n"})
    real_build = library._stage_build

    def _failing_build(path, unit, out, options, schema):
        if "2016Q3" in unit["source"]:
            raise ValueError("staging 2016Q3 failed: bad row")
        return real_build(path, unit, out, options, schema)

    monkeypatch.setattr(library, "_stage_build", _failing_build)
    opts = dict(separator="|", has_header=False, column_names="id,period,dlq", infer_schema_rows=0)
    with pytest.raises(ValueError, match="2016Q3 failed"):
        library.stage_csv_parquet(str(raw), workers=2, **opts)
    target = raw.parent / "raw_parquet"
    recorded = set(json.loads((target / "_manifest.json").read_text())["parts"])
    assert "2016Q3.parquet" not in recorded and recorded == set(_part_mtimes(target))

    monkeypatch.setattr(library, "_stage_build", real_build)
    before = _part_mtimes(target)
    assert library.stage_csv_parquet(str(raw), workers=2, **opts).collect().height == 5
    after = _part_mtimes(target)
    assert set(after) == {f"2016Q{q}.parquet" for q in range(1, 6)}
    # Only what the failed run didn't finish is built: 2016Q3, and any part
    # still waiting for a worker when it failed.
    rebuilt = {p for p in after if before.get(p) != after[p]}
    assert "2016Q3.parquet" in rebuilt and rebuilt == set(after) - recorded


def test_stage_later_parts_never_fail_on_columns_not_staged(tmp_path):
    from modelmaker.blocks.library import stage_csv_parquet

    raw = tmp_path / "raw"
    raw.mkdir()
    # `b` would infer as an integer from the first file and break on the
    # second -- but it isn't staged, so it's never parsed as one.
    (raw / "1.csv").write_text("a,b\n1,2\n")
    (raw / "2.csv").write_text("a,b\n3,oops\n")
    df = stage_csv_parquet(str(raw), columns="a").collect()
    assert df.schema == {"a": pl.Int64} and df["a"].to_list() == [1, 3]


# ---- the collect guard ------------------------------------------------------------


def test_guarded_collect_matches_a_plain_collect(monkeypatch):
    from modelmaker import run_worker

    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "512")
    lf = pl.LazyFrame({"a": range(2_500_000)}).with_columns(b=pl.col("a") % 7)
    assert run_worker._collect_lazy(lf).equals(lf.collect())
    empty = run_worker._collect_lazy(lf.filter(pl.col("a") < 0))
    assert empty.height == 0 and empty.schema == lf.collect_schema()


def test_guarded_collect_stops_an_output_too_big_for_memory(monkeypatch):
    from modelmaker import run_worker

    lf = pl.LazyFrame({"a": range(3_000_000)}).with_columns(b=pl.col("a").cast(pl.String))
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "5")
    with pytest.raises(run_worker.OutputTooBig, match="too big to hold in memory") as stopped:
        run_worker._collect_lazy(lf)
    assert "its 5 MB limit (set by MODELMAKER_COLLECT_LIMIT_MB=5)" in str(stopped.value)
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "0")  # turned off
    assert run_worker.collect_limit_bytes() is None
    assert run_worker._collect_lazy(lf).height == 3_000_000


def test_frame_bytes_counts_the_view_of_every_string_value():
    # polars' estimated_size() counts a string's characters but not its
    # 16-byte view, so short codes (MMYYYY dates, status codes, read as text)
    # held ~4x what it said and the guard let a loan tape run the machine out
    # of memory instead of stopping it.
    from modelmaker.run_worker import STRING_VIEW_BYTES, frame_bytes

    n = 100_000
    df = pl.DataFrame({"period": ["012014"] * n, "dlq": ["00"] * n, "upb": [1.0] * n})
    assert frame_bytes(df) == df.estimated_size() + STRING_VIEW_BYTES * 2 * n
    assert frame_bytes(df) >= 3 * df.estimated_size()
    nested = pl.DataFrame({"l": [["a", "b"], ["c"]], "s": [{"x": "a", "y": 1}, {"x": None, "y": 2}], "b": [b"x", b"y"]})
    assert frame_bytes(nested) == nested.estimated_size() + STRING_VIEW_BYTES * (3 + 2 + 2)
    deeper = pl.DataFrame(
        {"a": pl.Series([["x", "y"], ["z", "w"]], dtype=pl.Array(pl.String, 2)), "ll": [[["a"], ["b", "c"]], []]}
    )
    # Array: 2 rows x 2 values; the list of lists: a, b and c.
    assert frame_bytes(deeper) == deeper.estimated_size() + STRING_VIEW_BYTES * (4 + 3)
    numeric = pl.DataFrame({"a": range(10), "l": [[1]] * 10})
    assert frame_bytes(numeric) == numeric.estimated_size()


def test_guarded_collect_stops_short_codes_that_polars_underestimates(monkeypatch):
    from modelmaker import run_worker

    # 1.5M rows of three short text codes: polars says ~16 MB, they hold ~85 MB.
    lf = pl.LazyFrame({"a": range(1_500_000)}).select(
        period=pl.col("a").mod(120).cast(pl.String).str.zfill(6),
        dlq=pl.col("a").mod(7).cast(pl.String).str.zfill(2),
        state=pl.col("a").mod(50).cast(pl.String),
    )
    collected = lf.collect()
    assert collected.height == 1_500_000 and collected.estimated_size() < 30 * 1024**2
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "40")
    with pytest.raises(run_worker.OutputTooBig, match="too big to hold in memory"):
        run_worker._collect_lazy(lf)


def test_cache_entry_size_counts_string_views():
    from modelmaker.run_worker import STRING_VIEW_BYTES

    df = pl.DataFrame({"code": ["00"] * 1000})
    entry = type("E", (), {"outputs": {"out": DataFramePacket(data=df, schema_meta={})}})()
    assert entry_size(entry) == df.estimated_size() + STRING_VIEW_BYTES * 1000


def test_the_default_limit_says_what_it_is_based_on(monkeypatch):
    # 2.1 GB on a 16 GB machine read as "this machine can spare 2.1 GB" (#94).
    from modelmaker import run_worker

    free, total = run_worker.memory_bytes()  # this machine's, for real
    if sys.platform in ("linux", "darwin", "win32"):
        assert free is not None and total and 0 < free <= total
    monkeypatch.delenv(run_worker.COLLECT_LIMIT_ENV, raising=False)
    monkeypatch.setattr(run_worker, "memory_bytes", lambda: (int(5.3 * 1024**3), 16 * 1024**3))
    limit, basis = run_worker.collect_limit()
    assert limit == int(5.3 * 1024**3 * run_worker.COLLECT_FREE_SHARE)
    assert "40% of the 5.3 GB of memory free when the block started, of 16.0 GB in total" in basis
    lf = pl.LazyFrame({"a": range(10)})
    monkeypatch.setattr(run_worker, "memory_bytes", lambda: (100, None))
    with pytest.raises(run_worker.OutputTooBig, match=r"limit \(40% of the 0 MB of memory free when the block started; other"):
        run_worker._collect_lazy(lf)
    monkeypatch.setattr(run_worker, "memory_bytes", lambda: (None, None))
    assert run_worker.collect_limit() is None and run_worker._collect_lazy(lf).height == 10


def test_macos_free_memory_counts_free_and_inactive_pages():
    # macOS has no /proc/meminfo or SC_AVPHYS_PAGES; free memory read as
    # unknown there, and that turned the collect limit off altogether.
    from modelmaker import run_worker

    assert run_worker._darwin_free_from_counts(1000, 3000, 16384) == 4000 * 16384


@pytest.mark.skipif(sys.platform != "darwin", reason="reads the Mach VM statistics")
def test_macos_free_memory_is_read_and_sets_a_limit(monkeypatch):
    from modelmaker import run_worker

    free = run_worker._darwin_available_bytes()
    total = run_worker._sysconf_bytes("SC_PHYS_PAGES")
    assert free is not None and total and 0 < free <= total
    monkeypatch.delenv(run_worker.COLLECT_LIMIT_ENV, raising=False)
    assert run_worker.collect_limit() is not None


def test_a_too_big_input_block_ends_red_not_in_a_crash(tmp_path, monkeypatch):
    from modelmaker import run_worker

    path = tmp_path / "big.parquet"
    pl.DataFrame({"a": range(3_000_000)}).with_columns(b=pl.col("a").cast(pl.String)).write_parquet(path)
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "5")
    graph = Graph(blocks={"r": make_block("r", "read_parquet", params={"path": str(path)})}, wires={})
    runner = Runner(graph, CacheStore())
    assert runner.refresh("r") == "red"
    assert "too big to hold in memory" in runner.state["r"].last_error


def test_run_outcome_points_an_out_of_memory_block_at_reducing():
    from modelmaker.assistant_mcp import OUT_OF_MEMORY_NEXT, run_outcome

    for error in (
        "OutputTooBig: this block's output is too big to hold in memory: ...",
        "RuntimeError: worker process exited unexpectedly (code 3221226505) -- likely out of memory or a crash",
    ):
        graph = {"blocks": [{"id": "b", "name": "B", "status": "red", "error": error}]}
        assert run_outcome(graph, None)["next"] == OUT_OF_MEMORY_NEXT
