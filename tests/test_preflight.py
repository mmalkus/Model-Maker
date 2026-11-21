"""The size check before a run (modelmaker/preflight.py): estimates of an
input block's output without running it, which blocks a run would collect
whole past the memory limit, and the advice shown on a block stopped for
running out of memory."""

from __future__ import annotations

import gzip
import zipfile

import polars as pl
import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker import preflight, run_worker
from modelmaker.blocks.library import stage_csv_parquet
from modelmaker.cache import CacheStore
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner
from modelmaker.session import ProjectSession

from .helpers import make_block

N = 20_000


def _loan_tape(path, n=N, header=False):
    """A pipe-separated, header-less file shaped like a loan performance
    file: id, MMYYYY period, a rate and a two-character status."""
    lines = [f"{i}|{(i % 12) + 1:02d}2019|{4 + i % 3}.25|0{i % 4}\n" for i in range(n)]
    text = ("loan_id|period|rate|dlq\n" if header else "") + "".join(lines)
    path.write_text(text)
    return path


def _stage_params(path, tmp_path, **extra):
    return {
        "path": str(path),
        "separator": "|",
        "has_header": False,
        "infer_schema_rows": 0,
        "parquet_dir": str(tmp_path / "staged"),
        **extra,
    }


# ---- estimates ---------------------------------------------------------------


def test_parquet_estimate_counts_rows_from_metadata(tmp_path):
    path = tmp_path / "t.parquet"
    df = pl.DataFrame({"id": range(N), "code": ["00"] * N})
    df.write_parquet(path)
    est = preflight.estimate_block(make_block("r", "read_parquet", params={"path": str(path)}))
    assert est.exact and est.rows == N
    assert est.bytes == pytest.approx(run_worker.frame_bytes(df), rel=0.01)


def test_csv_estimate_scales_a_sample_of_the_text(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "SAMPLE_TEXT_BYTES", 64 * 1024)  # sample a fraction of the file
    path = _loan_tape(tmp_path / "perf.csv", header=True)
    est = preflight.estimate_block(make_block("r", "read_csv", params={"path": str(path), "separator": "|"}))
    assert not est.exact and est.rows == pytest.approx(N, rel=0.05)
    whole = pl.read_csv(path, separator="|", infer_schema_length=100)
    assert est.bytes == pytest.approx(run_worker.frame_bytes(whole), rel=0.1)


def test_stage_estimate_reads_zip_sizes_then_the_staged_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "SAMPLE_TEXT_BYTES", 64 * 1024)
    text = _loan_tape(tmp_path / "2019Q1.csv")
    for q in ("2019Q1", "2019Q2"):
        with zipfile.ZipFile(tmp_path / f"{q}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(text, f"{q}.csv")
    text.unlink()
    params = _stage_params(tmp_path, tmp_path, columns="1:loan_id, 4:dlq")
    block = make_block("s", "stage_csv_parquet", params=params)
    before = preflight.estimate_block(block)
    assert not before.exact and before.rows == pytest.approx(2 * N, rel=0.05)
    staged = stage_csv_parquet(**params).collect()
    after = preflight.estimate_block(block)
    assert after.exact and after.rows == 2 * N
    assert after.bytes == pytest.approx(run_worker.frame_bytes(staged), rel=0.01)
    assert before.bytes == pytest.approx(after.bytes, rel=0.1)
    # Changed options make the staged Parquet stale: back to the text.
    block.params["columns"] = "1:loan_id"
    assert not preflight.estimate_block(block).exact


def test_no_estimate_for_gzip_other_blocks_or_a_missing_file(tmp_path):
    gz = tmp_path / "perf.csv.gz"
    gz.write_bytes(gzip.compress(b"a|b\n1|2\n"))
    assert preflight.estimate_block(make_block("s", "stage_csv_parquet", params=_stage_params(gz, tmp_path))) is None
    assert preflight.estimate_block(make_block("r", "read_csv", params={"path": str(tmp_path / "gone.csv")})) is None
    assert preflight.estimate_block(make_block("d", "demo_credit_data")) is None


# ---- which blocks a run would collect too big ---------------------------------


def _graph(tmp_path, consumer):
    path = tmp_path / "t.parquet"
    pl.DataFrame({"id": range(N), "code": ["00"] * N}).write_parquet(path)
    blocks = {"r": make_block("r", "read_parquet", params={"path": str(path)})}
    wires = {}
    if consumer:
        params = {"expr": "id > 0"} if consumer == "filter" else {}
        blocks["c"] = make_block("c", consumer, params=params, x=1)
        port = blocks["c"].inputs[0].name
        wires = {"w": Wire("w", "r", "out", "c", port)}
    graph = Graph(blocks=blocks, wires=wires)
    return graph, Runner(graph, CacheStore())


def test_flags_a_source_over_the_limit(tmp_path, monkeypatch):
    graph, runner = _graph(tmp_path, "filter")
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "1")  # 1 MB; the source is ~0.6 MB of ids + 0.3 MB of views
    monkeypatch.setattr(preflight, "estimate_block", lambda b: preflight.SizeEstimate(10**7, 5 * 1024**2, True))
    out = preflight.preflight(graph, runner, "run_all")
    [flag] = out["blocks"]
    assert flag["id"] == "r" and flag["rows"] == 10**7 and flag["size"] == "5 MB"
    assert flag["can_stream"] and flag["whole_consumers"] == []
    assert out["limit"] == "1 MB" and "MODELMAKER_COLLECT_LIMIT_MB=1" in out["limit_basis"]
    # A streaming run fuses it into the lazy filter: nothing to flag.
    assert preflight.preflight(graph, runner, "run_all_streaming")["blocks"] == []
    # Run block on the filter runs only the filter, and Run to here never
    # reads the sources upstream (the filter is left blocked instead).
    assert preflight.preflight(graph, runner, "run_block", "c")["blocks"] == []
    assert preflight.preflight(graph, runner, "run_to_here", "c")["blocks"] == []
    assert [b["id"] for b in preflight.preflight(graph, runner, "run_to_here", "r")["blocks"]] == ["r"]


def test_streaming_still_flags_a_source_taken_whole(tmp_path, monkeypatch):
    graph, runner = _graph(tmp_path, "one_hot_encode")
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "1")
    monkeypatch.setattr(preflight, "estimate_block", lambda b: preflight.SizeEstimate(10**7, 5 * 1024**2, True))
    [flag] = preflight.preflight(graph, runner, "run_all_streaming")["blocks"]
    assert not flag["can_stream"] and flag["whole_consumers"] == ["c"]
    # A source feeding nothing is its own result: collected too.
    graph2, runner2 = _graph(tmp_path, None)
    assert len(preflight.preflight(graph2, runner2, "run_all_streaming")["blocks"]) == 1


def test_nothing_flagged_under_the_limit_in_sample_mode_or_with_the_guard_off(tmp_path, monkeypatch):
    graph, runner = _graph(tmp_path, "filter")
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "1000")
    assert preflight.preflight(graph, runner, "force_run_all")["blocks"] == []
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "1")
    monkeypatch.setattr(preflight, "estimate_block", lambda b: preflight.SizeEstimate(10**7, 5 * 1024**2, True))
    assert len(preflight.preflight(graph, runner, "force_run_all")["blocks"]) == 1
    runner.sample_rows = 1000
    assert preflight.preflight(graph, runner, "force_run_all")["blocks"] == []
    runner.sample_rows = None
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "0")
    assert preflight.preflight(graph, runner, "force_run_all")["blocks"] == []


def test_run_all_skips_a_source_already_current(tmp_path, monkeypatch):
    graph, runner = _graph(tmp_path, "filter")
    assert runner.run_block("r") == "green"
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "1")
    monkeypatch.setattr(preflight, "estimate_block", lambda b: preflight.SizeEstimate(10**7, 5 * 1024**2, True))
    assert preflight.preflight(graph, runner, "run_all")["blocks"] == []
    # Force run all forces only the blocks after the sources: a current one isn't re-read.
    assert preflight.preflight(graph, runner, "force_run_all")["blocks"] == []


@pytest.mark.parametrize("state", ["unread", "current", "stale"])
@pytest.mark.parametrize("mode", ["run_all", "force_run_all", "run_all_streaming"])
def test_scope_is_exactly_the_sources_the_run_reads(tmp_path, monkeypatch, mode, state):
    graph, runner = _graph(tmp_path, "filter")
    if state != "unread":
        assert runner.run_block("r") == "green"
    if state == "stale":
        other = tmp_path / "other.parquet"
        pl.DataFrame({"id": [1], "code": ["01"]}).write_parquet(other)
        graph.blocks["r"].params["path"] = str(other)
    expected = preflight._in_scope(graph, runner, mode, None)
    read: list[str] = []
    real_run_block, real_group = runner.run_block, runner._run_fused_group

    def run_block(bid, plan=None):
        if graph.blocks[bid].block_type == "input":
            read.append(bid)
        return real_run_block(bid, plan)

    def run_group(group, plan, report):
        read.extend(b for b in group.members if graph.blocks[b].block_type == "input")
        return real_group(group, plan, report)

    monkeypatch.setattr(runner, "run_block", run_block)
    monkeypatch.setattr(runner, "_run_fused_group", run_group)
    getattr(runner, mode)()
    assert expected == read, (mode, state)


# ---- advice on a block stopped for memory -------------------------------------------


def test_out_of_memory_errors_are_recognized():
    assert preflight.is_out_of_memory_error("OutputTooBig: this block's output is too big to hold in memory: ...")
    assert preflight.is_out_of_memory_error(
        "RuntimeError: worker process was killed by the system (signal 9) -- likely out of memory: ..."
    )
    assert not preflight.is_out_of_memory_error("ValueError: no column named x")
    # Any other crash: its message mentions memory for the assistants, but
    # streaming or sampling wouldn't fix a segfault.
    assert not preflight.is_out_of_memory_error(
        "RuntimeError: worker process exited unexpectedly (code -11) -- likely out of memory or a crash. ..."
    )
    assert not preflight.is_out_of_memory_error(None)


def test_api_preflight_and_memory_advice(tmp_path, monkeypatch):
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    client = TestClient(api_module.app)
    path = tmp_path / "t.parquet"
    pl.DataFrame({"id": range(N)}).write_parquet(path)
    r = client.post("/api/blocks", json={"category": "read_parquet", "params": {"path": str(path)}}).json()
    c = client.post("/api/blocks", json={"category": "one_hot_encode", "x": 300}).json()
    client.post("/api/wires", json={"from_block": r["id"], "from_port": "out", "to_block": c["id"], "to_port": c["inputs"][0]["name"]})
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "1000")
    assert client.post("/api/run_preflight", json={"mode": "run_all"}).json()["blocks"] == []
    monkeypatch.setattr(preflight, "estimate_block", lambda b: preflight.SizeEstimate(10**7, 2 * 1024**3, True))
    [flag] = client.post("/api/run_preflight", json={"mode": "force_run_all"}).json()["blocks"]
    assert flag["id"] == r["id"] and flag["whole_consumers"] == [c["name"]]
    assert client.post("/api/run_preflight", json={"mode": "run_block", "block_id": "nope"}).status_code == 404
    assert client.post("/api/run_preflight", json={"mode": "sideways"}).status_code == 422

    # A red block stopped for memory carries advice; any other block doesn't.
    runner = api_module.SESSION.runner
    assert client.get("/api/graph").json()["blocks"][r["id"]]["memory_advice"] is None
    monkeypatch.setenv(run_worker.COLLECT_LIMIT_ENV, "0")
    monkeypatch.setattr(runner, "status", lambda bid: "red")
    st = runner._st(r["id"])
    st.last_error = "OutputTooBig: this block's output is too big to hold in memory: ..."
    advice = client.get("/api/graph").json()["blocks"][r["id"]]["memory_advice"]
    assert advice == {"can_stream": False, "whole_consumers": [c["name"]], "sample_rows": preflight.SUGGESTED_SAMPLE_ROWS}


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGKILL"), reason="no SIGKILL on this platform")
def test_a_worker_killed_by_the_system_says_so():
    code = (
        "import os, signal\n"
        "import polars as pl\n\n"
        "def killed() -> pl.DataFrame:\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    graph = Graph(blocks={"k": make_block("k", "", code=code, block_type="input")}, wires={})
    runner = Runner(graph, CacheStore())
    assert runner.run_block("k") == "red"
    error = runner.state["k"].last_error
    assert "killed by the system (signal 9)" in error and preflight.is_out_of_memory_error(error)
