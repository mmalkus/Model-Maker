"""Runner.run_all_streaming: fusing compatible blocks (filter/select/
groupby_agg/join/read_csv -- see BlockSpec.lazy_fn) into a single polars
query per group. See runner.py's _build_fusion_groups/_run_fused_group for
the design; these tests exercise it end to end through the public Runner
API, same as the rest of tests/test_runner.py.
"""

from modelmaker.blocks.base import PortSpec
from modelmaker.cache import CacheStore
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def _chain_graph(csv_path):
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 5"}, x=1)
    sel = make_block("b_select", "select", params={"cols": ["a", "b"]}, x=2)
    agg = make_block("b_agg", "groupby_agg", params={"by": ["b"], "aggs": {"a": "sum"}}, x=3)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_select": sel, "b_agg": agg},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_select", "df"),
            "w3": Wire("w3", "b_select", "out", "b_agg", "df"),
        },
    )
    return graph


def _write_csv(tmp_path, n=20):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "".join(f"{i},{'x' if i % 2 == 0 else 'y'}\n" for i in range(1, n + 1)))
    return csv_path


def test_streaming_chain_matches_normal_run(tmp_path):
    csv_path = _write_csv(tmp_path)

    normal = Runner(_chain_graph(csv_path), CacheStore())
    normal.refresh("b_read")
    normal.run_all()
    normal_entry = normal.cache.get(normal.state["b_agg"].last_successful_key)

    streaming = Runner(_chain_graph(csv_path), CacheStore())
    report = streaming.run_all_streaming()
    streaming_entry = streaming.cache.get(streaming.state["b_agg"].last_successful_key)

    assert normal_entry.outputs["out"].data.sort("b").to_dicts() == streaming_entry.outputs["out"].data.sort(
        "b"
    ).to_dicts()
    normal_dtypes = {k: v.dtype for k, v in normal_entry.outputs["out"].schema_meta.items()}
    streaming_dtypes = {k: v.dtype for k, v in streaming_entry.outputs["out"].schema_meta.items()}
    assert normal_dtypes == streaming_dtypes

    # The whole chain -- including the source read -- fused into one group;
    # only the exit block (b_agg) is independently cached/green.
    assert report == {"b_read": "fused", "b_filter": "fused", "b_select": "fused", "b_agg": "green"}
    assert streaming.status("b_read") == "grey"
    assert streaming.status("b_filter") == "grey"
    assert streaming.status("b_select") == "grey"
    assert streaming.status("b_agg") == "green"


def test_streaming_exit_key_matches_normal_run_block_key(tmp_path):
    csv_path = _write_csv(tmp_path)
    # Both runners refresh b_read first, so its read_counter -- part of its
    # own basis, and thus part of everything downstream's "upstream" basis
    # -- is identical in both; only the execution path (fused vs. one
    # block at a time) differs from here.
    streaming = Runner(_chain_graph(csv_path), CacheStore())
    streaming.refresh("b_read")
    streaming.run_all_streaming()
    streamed_key = streaming.state["b_agg"].last_successful_key

    normal = Runner(_chain_graph(csv_path), CacheStore())
    normal.refresh("b_read")
    normal.run_all()
    normal_key = normal.state["b_agg"].last_successful_key

    # Same basis (params/code/upstream) regardless of which execution path
    # produced the cache entry -- a later ordinary run recognizes the
    # streamed result as current with no special-casing anywhere else.
    assert streamed_key == normal_key


def test_fan_out_forces_a_checkpoint_and_both_branches_get_correct_data(tmp_path):
    csv_path = _write_csv(tmp_path)
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 5"}, x=1)
    sel1 = make_block("b_sel1", "select", params={"cols": ["a"]}, x=2, y=0)
    sel2 = make_block("b_sel2", "select", params={"cols": ["b"]}, x=2, y=1)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_sel1": sel1, "b_sel2": sel2},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_sel1", "df"),
            "w3": Wire("w3", "b_filter", "out", "b_sel2", "df"),
        },
    )
    runner = Runner(graph, CacheStore())
    report = runner.run_all_streaming()

    # b_filter has two not-yet-current consumers -- it must become its own
    # checkpoint (independently cached/green), not fused into either branch.
    assert report["b_filter"] == "green"
    assert report["b_sel1"] == "green"
    assert report["b_sel2"] == "green"

    sel1_out = runner.cache.get(runner.state["b_sel1"].last_successful_key).outputs["out"]
    sel2_out = runner.cache.get(runner.state["b_sel2"].last_successful_key).outputs["out"]
    assert sorted(sel1_out.data["a"].to_list()) == list(range(6, 21))
    assert sorted(sel2_out.data["b"].to_list()) == sorted(["x" if i % 2 == 0 else "y" for i in range(6, 21)])


def test_role_tag_propagates_through_fused_chain(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,target\n" + "".join(f"{i},{i % 2}\n" for i in range(1, 21)))
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    read.column_role_overrides = {"target": "target"}
    filt = make_block("b_filter", "filter", params={"expr": "a > 5"}, x=1)
    sel = make_block("b_select", "select", params={"cols": ["a", "target"]}, x=2)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_select": sel},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_select", "df"),
        },
    )
    runner = Runner(graph, CacheStore())
    runner.run_all_streaming()

    out = runner.cache.get(runner.state["b_select"].last_successful_key).outputs["out"]
    assert out.schema_meta["target"].role.value == "target"
    assert out.schema_meta["a"].role.value == "unassigned"


def test_error_in_fused_chain_names_the_failing_step(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "nonexistent_col > 1"}, x=1)
    sel = make_block("b_select", "select", params={"cols": ["a"]}, x=2)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_select": sel},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_select", "df"),
        },
    )
    runner = Runner(graph, CacheStore())
    report = runner.run_all_streaming()

    assert report["b_select"] == "red"
    assert runner.status("b_select") == "red"
    assert "filter" in runner.state["b_select"].last_error
    assert "nonexistent_col" in runner.state["b_select"].last_error
    assert report["b_read"] == "fused (group failed)"
    assert report["b_filter"] == "fused (group failed)"


def test_streaming_raises_when_sample_mode_is_on(tmp_path):
    csv_path = _write_csv(tmp_path)
    runner = Runner(_chain_graph(csv_path), CacheStore(), sample_rows=5)
    try:
        runner.run_all_streaming()
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "sample mode" in str(e)


def test_custom_block_splits_the_fusion_chain(tmp_path):
    csv_path = _write_csv(tmp_path, n=10)
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 2"}, x=1)
    custom_code = "def double_a(df: pl.DataFrame) -> pl.DataFrame:\n    return df.with_columns((pl.col('a') * 2).alias('a'))\n"
    custom = make_block(
        "b_custom",
        "ai_double",
        params={},
        code=custom_code,
        block_type="standard",
        x=2,
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
    )
    sel = make_block("b_select", "select", params={"cols": ["a"]}, x=3)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_custom": custom, "b_select": sel},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_custom", "df"),
            "w3": Wire("w3", "b_custom", "out", "b_select", "df"),
        },
    )
    runner = Runner(graph, CacheStore())
    report = runner.run_all_streaming()

    # b_filter's only consumer is the custom block, which never fuses --
    # so b_filter is its own checkpoint, b_custom and b_select run normally.
    assert report["b_filter"] == "green"
    assert report["b_custom"] == "green"
    assert report["b_select"] == "green"

    out = runner.cache.get(runner.state["b_select"].last_successful_key).outputs["out"]
    assert sorted(out.data["a"].to_list()) == [6, 8, 10, 12, 14, 16, 18, 20]


def test_join_extends_a_fused_chain_with_the_other_side_read_from_cache(tmp_path):
    left_path = tmp_path / "left.csv"
    left_path.write_text("id,a\n" + "".join(f"{i},{i * 10}\n" for i in range(1, 11)))
    right_path = tmp_path / "right.csv"
    right_path.write_text("id,label\n" + "".join(f"{i},{'even' if i % 2 == 0 else 'odd'}\n" for i in range(1, 11)))

    read_left = make_block("b_left", "read_csv", params={"path": str(left_path)})
    read_right = make_block("b_right", "read_csv", params={"path": str(right_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 30"}, x=1)
    joined = make_block("b_join", "join", params={"on": ["id"], "how": "inner"}, x=2)
    sel = make_block("b_select", "select", params={"cols": ["id", "label"]}, x=3)
    graph = Graph(
        blocks={
            "b_left": read_left,
            "b_right": read_right,
            "b_filter": filt,
            "b_join": joined,
            "b_select": sel,
        },
        wires={
            "w1": Wire("w1", "b_left", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_join", "left"),
            "w3": Wire("w3", "b_right", "out", "b_join", "right"),
            "w4": Wire("w4", "b_join", "out", "b_select", "df"),
        },
    )
    # b_right is read via a normal refresh -- already cached and current --
    # so the join's "right" side is a checkpoint read, while "left" chains
    # in from the still-fusing b_left -> b_filter stretch.
    runner = Runner(graph, CacheStore())
    runner.refresh("b_right")
    report = runner.run_all_streaming()

    assert report["b_left"] == "fused"
    assert report["b_filter"] == "fused"
    assert report["b_join"] == "fused"
    assert report["b_select"] == "green"

    out = runner.cache.get(runner.state["b_select"].last_successful_key).outputs["out"]
    assert sorted(out.data["id"].to_list()) == [4, 5, 6, 7, 8, 9, 10]
