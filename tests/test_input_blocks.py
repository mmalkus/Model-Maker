"""New input blocks (read_parquet/read_json/read_excel/read_sql): each
should behave like read_csv -- infer_dtypes metadata, a "check for changes"
probe, and sample_rows support where a lazy/streaming reader exists. See
modelmaker/blocks/library.py.
"""

import sqlite3

import polars as pl

from modelmaker.cache import CacheStore
from modelmaker.compiler import compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def _single_input_graph(bid, category, params):
    block = make_block(bid, category, params=params)
    graph = Graph(blocks={bid: block}, wires={})
    return graph


def _output_of(runner, bid):
    return runner.cache.get(runner.state[bid].last_successful_key).outputs["out"].data


def test_read_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_parquet(path)

    graph = _single_input_graph("b_read", "read_parquet", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.status("b_read") == "green"
    assert _output_of(runner, "b_read")["a"].to_list() == [1, 2, 3]


def test_read_parquet_probe_flags_mtime_change(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)

    graph = _single_input_graph("b_read", "read_parquet", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.check_for_changes("b_read") is False  # first probe just baselines

    pl.DataFrame({"a": [1, 2]}).write_parquet(path)
    assert runner.check_for_changes("b_read") is True


def test_read_json_array(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')

    graph = _single_input_graph("b_read", "read_json", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.status("b_read") == "green"
    assert _output_of(runner, "b_read")["a"].to_list() == [1, 2]


def test_read_json_ndjson(tmp_path):
    path = tmp_path / "data.ndjson"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')

    graph = _single_input_graph("b_read", "read_json", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert _output_of(runner, "b_read")["a"].to_list() == [1, 2, 3]


def test_read_excel(tmp_path):
    path = tmp_path / "data.xlsx"
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_excel(path)

    graph = _single_input_graph("b_read", "read_excel", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.status("b_read") == "green"
    assert _output_of(runner, "b_read")["a"].to_list() == [1, 2, 3]


def test_read_excel_named_sheet(tmp_path):
    import xlsxwriter

    path = tmp_path / "data.xlsx"
    wb = xlsxwriter.Workbook(str(path))
    pl.DataFrame({"a": [1, 2]}).write_excel(wb, worksheet="Sheet1")
    pl.DataFrame({"a": [9, 9, 9]}).write_excel(wb, worksheet="Other")
    wb.close()

    graph = _single_input_graph("b_read", "read_excel", {"path": str(path), "sheet": "Other"})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert _output_of(runner, "b_read")["a"].to_list() == [9, 9, 9]


def test_input_block_chains_into_a_standard_block(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2, 3, 4]}).write_parquet(path)

    read = make_block("b_read", "read_parquet", params={"path": str(path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 2"}, x=1)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt},
        wires={"w1": Wire("w1", "b_read", "out", "b_filter", "df")},
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_block("b_filter")

    assert runner.status("b_filter") == "green"
    out = runner.cache.get(runner.state["b_filter"].last_successful_key).outputs["out"]
    assert out.data["a"].to_list() == [3, 4]


def test_read_parquet_compiles_and_matches_engine_output(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)

    graph = _single_input_graph("b_read", "read_parquet", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    source = compile_graph(graph, runner=runner)
    assert "# === Function: read_parquet | type=input | category=read_parquet ===" in source

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert ns["b_read_b_read"].to_dicts() == _output_of(runner, "b_read").to_dicts()


def test_read_excel_compiles_and_matches_engine_output(tmp_path):
    path = tmp_path / "data.xlsx"
    pl.DataFrame({"a": [1, 2, 3]}).write_excel(path)

    graph = _single_input_graph("b_read", "read_excel", {"path": str(path)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    source = compile_graph(graph, runner=runner)
    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert ns["b_read_b_read"].to_dicts() == _output_of(runner, "b_read").to_dicts()


def _sqlite_db(tmp_path, rows):
    db_path = tmp_path / "data.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE loans (id INTEGER, amount REAL)")
    con.executemany("INSERT INTO loans VALUES (?, ?)", rows)
    con.commit()
    con.close()
    return f"sqlite://{db_path}"


def test_read_sql(tmp_path, monkeypatch):
    uri = _sqlite_db(tmp_path, [(1, 100.0), (2, 200.0), (3, 300.0)])
    monkeypatch.setenv("MM_TEST_DB_URL", uri)

    graph = _single_input_graph(
        "b_read", "read_sql", {"connection_env": "MM_TEST_DB_URL", "query": "SELECT * FROM loans ORDER BY id"}
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.status("b_read") == "green"
    assert _output_of(runner, "b_read")["id"].to_list() == [1, 2, 3]


def test_read_sql_never_persists_the_connection_string_in_params(tmp_path, monkeypatch):
    # The whole point of connection_env: params (which get written into
    # project files and baked into compiled scripts) hold only an env var
    # *name*, never the connection string/password itself.
    uri = _sqlite_db(tmp_path, [(1, 100.0)])
    monkeypatch.setenv("MM_TEST_DB_URL", uri)

    block = make_block("b_read", "read_sql", params={"connection_env": "MM_TEST_DB_URL", "query": "SELECT * FROM loans"})
    assert uri not in str(block.params)
    assert "MM_TEST_DB_URL" in str(block.params)


def test_read_sql_probe_uses_probe_query(tmp_path, monkeypatch):
    uri = _sqlite_db(tmp_path, [(1, 100.0)])
    monkeypatch.setenv("MM_TEST_DB_URL", uri)

    graph = _single_input_graph(
        "b_read",
        "read_sql",
        {"connection_env": "MM_TEST_DB_URL", "query": "SELECT * FROM loans", "probe_query": "SELECT COUNT(*) AS n FROM loans"},
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.check_for_changes("b_read") is False  # first probe just baselines

    con = sqlite3.connect(uri.removeprefix("sqlite://"))
    con.execute("INSERT INTO loans VALUES (2, 200.0)")
    con.commit()
    con.close()

    assert runner.check_for_changes("b_read") is True


def test_read_sql_without_probe_query_never_flags_changed(tmp_path, monkeypatch):
    uri = _sqlite_db(tmp_path, [(1, 100.0)])
    monkeypatch.setenv("MM_TEST_DB_URL", uri)

    graph = _single_input_graph("b_read", "read_sql", {"connection_env": "MM_TEST_DB_URL", "query": "SELECT * FROM loans"})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.check_for_changes("b_read") is False
    assert runner.check_for_changes("b_read") is False


def test_read_sql_compiles_and_matches_engine_output(tmp_path, monkeypatch):
    uri = _sqlite_db(tmp_path, [(1, 100.0), (2, 200.0)])
    monkeypatch.setenv("MM_TEST_DB_URL", uri)

    graph = _single_input_graph(
        "b_read", "read_sql", {"connection_env": "MM_TEST_DB_URL", "query": "SELECT * FROM loans ORDER BY id"}
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    source = compile_graph(graph, runner=runner)
    assert "os.environ" in source
    assert uri not in source  # the compiled script must reference the env var, never bake in the connection string

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert ns["b_read_b_read"].to_dicts() == _output_of(runner, "b_read").to_dicts()
