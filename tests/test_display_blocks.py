"""Display blocks (display_table, display_value, display_kpi): what they
compute, the CSV a table display saves, the result their canvas node shows
(api._display_out), and their re-run right after their upstream runs."""

from __future__ import annotations

import math

import polars as pl
import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker import block_library as bl
from modelmaker.blocks.base import BLOCK_REGISTRY
from modelmaker.blocks.library import display_kpi, display_table, display_value
from modelmaker.cache import CacheStore
from modelmaker.compiler import compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner
from modelmaker.session import ProjectSession

from .helpers import make_block

# ---- the block functions ---------------------------------------------------------------


def test_display_kpi_reduces_a_column_over_the_rows():
    df = pl.DataFrame({"segment": ["retail", "sme", "corp"], "rwa": [120.0, 80.0, 50.5]})
    assert display_kpi(df, column="rwa", agg="sum", label="Total RWA") == {
        "label": "Total RWA",
        "value": 250.5,
        "column": "rwa",
        "agg": "sum",
        "number_format": "number",
        "decimals": 2,
        "unit": "",
    }
    assert display_kpi(df, column="rwa")["value"] == 120.0  # agg defaults to the first row
    assert display_kpi(df, column="rwa", agg="last")["value"] == 50.5
    assert display_kpi(df, column="rwa", agg="max")["value"] == 120.0
    assert display_kpi(df, column="segment", agg="count")["value"] == 3


def test_display_kpi_picks_the_only_numeric_column_and_labels_it():
    one_row_total = pl.DataFrame({"portfolio": ["all"], "total_rwa": [1_234_567.89]})
    kpi = display_kpi(one_row_total)
    assert (kpi["column"], kpi["label"], kpi["value"]) == ("total_rwa", "total_rwa", 1_234_567.89)


def test_display_kpi_asks_for_a_column_when_it_is_ambiguous_or_wrong():
    df = pl.DataFrame({"ead": [1.0], "rwa": [2.0]})
    with pytest.raises(ValueError, match="set `column`"):
        display_kpi(df)
    with pytest.raises(ValueError, match="no column 'pd'"):
        display_kpi(df, column="pd")
    with pytest.raises(ValueError, match="agg must be one of"):
        display_kpi(df, column="rwa", agg="median")
    with pytest.raises(TypeError, match="isn't a number"):
        display_kpi(pl.DataFrame({"s": ["x"]}), column="s")


def test_display_kpi_reads_a_metric_dict_or_a_plain_number():
    metric = {"kind": "auc_gini", "auc": 0.71, "gini": 0.42}
    assert display_kpi(metric, column="gini", label="Gini")["value"] == 0.42
    with pytest.raises(ValueError, match="set `column`"):
        display_kpi(metric)  # two numeric keys
    assert display_kpi(7, label="n")["value"] == 7


def test_display_kpi_shows_an_empty_or_non_finite_result_as_no_value():
    assert display_kpi(pl.DataFrame({"x": []}, schema={"x": pl.Float64}), column="x")["value"] is None
    assert display_kpi({"x": math.nan}, column="x")["value"] is None


def test_display_table_saves_the_whole_table_as_csv(tmp_path):
    df = pl.DataFrame({"grade": list("ABCDEFGHIJKL"), "rwa": list(range(12))})
    out = tmp_path / "out"
    assert display_table(df, output_dir=str(out), block_id="b_rwa") is df
    assert pl.read_csv(out / "b_rwa.csv").equals(df)  # every row, not just the ones the node shows

    display_table(df, filename="rwa_by_grade.csv", output_dir=str(out))
    assert pl.read_csv(out / "rwa_by_grade.csv").height == 12

    display_table(df, save_csv=False, output_dir=str(tmp_path / "none"))
    assert not (tmp_path / "none").exists()


def test_display_value_saves_only_a_table(tmp_path):
    display_value({"gini": 0.4}, output_dir=str(tmp_path), block_id="b_v")
    assert not (tmp_path / "b_v.csv").exists()
    display_value(pl.DataFrame({"a": [1]}), output_dir=str(tmp_path), block_id="b_v")
    assert (tmp_path / "b_v.csv").exists()


# ---- declaration -----------------------------------------------------------------------


def test_display_blocks_declare_how_they_show_and_validate():
    for category, kind in (("display_table", "table"), ("display_value", "value"), ("display_kpi", "kpi")):
        d = BLOCK_REGISTRY[category]
        assert d.display == kind
        assert d.block_type == "output" and "output" in d.tags
        assert bl.validate_definition(d) == []


def test_validator_rejects_an_unknown_display_kind_and_a_display_without_outputs():
    source = '''"""x."""

import polars as pl

BLOCKS = {
    "show_a": {"display": "chart", "inputs": [{"name": "df"}], "outputs": [{"name": "out"}]},
    "show_b": {"display": "table", "inputs": [{"name": "df"}], "outputs": []},
}


def show_a(df):
    return df


def show_b(df):
    return None
'''
    parsed = bl.validate(bl.parse_source(source, "show.py"))
    assert any("display must be one of" in e for e in parsed.errors["show_a"])
    assert any("needs an output port" in e for e in parsed.errors["show_b"])


# ---- runner and compiler ---------------------------------------------------------------


def _rwa_graph(tmp_path, rows: int = 3) -> Graph:
    csv_path = tmp_path / "rwa.csv"
    pl.DataFrame({"grade": [f"G{i}" for i in range(rows)], "rwa": [float(i) for i in range(rows)]}).write_csv(csv_path)
    return Graph(
        blocks={
            "b_read": make_block("b_read", "read_csv", params={"path": str(csv_path)}),
            "b_table": make_block("b_table", "display_table", x=1),
            "b_kpi": make_block("b_kpi", "display_kpi", params={"column": "rwa", "agg": "sum", "label": "Total RWA"}, x=2),
        },
        wires={
            "w1": Wire("w1", "b_read", "out", "b_table", "df"),
            "w2": Wire("w2", "b_read", "out", "b_kpi", "value"),
        },
    )


def test_display_blocks_rerun_as_soon_as_their_upstream_does(tmp_path):
    runner = Runner(_rwa_graph(tmp_path), CacheStore(), output_dir=str(tmp_path / "out"))
    runner.refresh("b_read")
    # No click on either display block: running the upstream ran them.
    assert runner.status("b_table") == "green"
    assert runner.status("b_kpi") == "green"
    kpi = runner.cache.get(runner.state["b_kpi"].last_successful_key).outputs["kpi"]
    assert kpi["value"] == 3.0
    assert (tmp_path / "out" / "b_table.csv").exists()

    # ...and again after the upstream changes.
    _rwa_graph(tmp_path, rows=5)  # rewrites rwa.csv with five rows
    runner.refresh("b_read")
    kpi = runner.cache.get(runner.state["b_kpi"].last_successful_key).outputs["kpi"]
    assert kpi["value"] == 10.0


def test_a_graph_with_display_blocks_compiles_and_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "compiled"))
    graph = _rwa_graph(tmp_path)
    runner = Runner(graph, CacheStore(), output_dir=str(tmp_path / "out"))
    runner.refresh("b_read")
    source = compile_graph(graph, runner=runner)
    assert "display_table(df=" in source and "display_kpi(value=" in source
    exec(compile(source, "<compiled>", "exec"), {})
    assert pl.read_csv(tmp_path / "compiled" / "b_table.csv").height == 3


# ---- what the canvas node shows --------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    api_module.SESSION.runner.output_dir = str(tmp_path / "out")
    return TestClient(api_module.app)


def _graph_display(client, block_id):
    return client.get("/api/graph").json()["blocks"][block_id]["display"]


def _read_csv_into(client, tmp_path, df: pl.DataFrame, *targets: tuple[str, str]):
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    for target_id, port in targets:
        client.post("/api/wires", json={"from_block": read["id"], "from_port": "out", "to_block": target_id, "to_port": port})
    client.post(f"/api/blocks/{read['id']}/refresh")
    return read


def test_node_shows_a_small_table_whole_and_a_large_one_as_first_rows_and_counts(client, tmp_path):
    small = client.post("/api/blocks", json={"category": "display_table"}).json()
    big = client.post("/api/blocks", json={"category": "display_table"}).json()
    kpi = client.post("/api/blocks", json={"category": "display_kpi", "params": {"column": "c0", "agg": "sum"}}).json()
    assert small["display"] is None  # nothing to show before a run

    df = pl.DataFrame({f"c{i}": list(range(25)) for i in range(20)})
    read = _read_csv_into(client, tmp_path, df, (big["id"], "df"), (kpi["id"], "value"))
    filt = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "c0 < 2"}}).json()
    client.post("/api/wires", json={"from_block": read["id"], "from_port": "out", "to_block": filt["id"], "to_port": "df"})
    client.post("/api/wires", json={"from_block": filt["id"], "from_port": "out", "to_block": small["id"], "to_port": "df"})
    client.post(f"/api/blocks/{filt['id']}/run")

    shown = _graph_display(client, small["id"])
    assert shown["kind"] == "table"
    assert (shown["row_count"], shown["column_count"]) == (2, 20)
    assert len(shown["rows"]) == 2

    shown = _graph_display(client, big["id"])
    assert (shown["row_count"], shown["column_count"]) == (25, 20)
    assert len(shown["rows"]) == api_module.DISPLAY_ROWS
    assert [c["name"] for c in shown["columns"]] == [f"c{i}" for i in range(api_module.DISPLAY_COLUMNS)]
    assert shown["rows"][1] == [1] * api_module.DISPLAY_COLUMNS

    shown = _graph_display(client, kpi["id"])
    assert shown["kind"] == "kpi" and shown["value"] == sum(range(25)) and shown["label"] == "c0"

    # Other blocks never carry a display.
    assert client.get("/api/graph").json()["blocks"][filt["id"]]["display"] is None


def test_node_display_survives_values_plain_json_cannot_carry(client, tmp_path):
    table = client.post("/api/blocks", json={"category": "display_table"}).json()
    df = pl.DataFrame({"x": [1.0, None], "when": ["2024-01-31", "2024-02-29"]})
    _read_csv_into(client, tmp_path, df, (table["id"], "df"))
    graph = client.get("/api/graph")
    assert graph.status_code == 200
    assert graph.json()["blocks"][table["id"]]["display"]["rows"][1][0] is None


def test_display_value_node_shows_a_metric_as_key_value_pairs(client, tmp_path):
    viewer = client.post("/api/blocks", json={"category": "display_value"}).json()
    api_module.SESSION.runner.cache.set("k", {"value": {"kind": "auc_gini", "auc": 0.7, "gini": float("nan")}})
    api_module.SESSION.runner._st(viewer["id"]).last_successful_key = "k"
    api_module.SESSION.runner.compute_key = lambda bid: "k"  # pretend it's current
    shown = _graph_display(client, viewer["id"])
    assert shown == {"kind": "value", "items": [["kind", "auc_gini"], ["auc", 0.7], ["gini", None]]}
