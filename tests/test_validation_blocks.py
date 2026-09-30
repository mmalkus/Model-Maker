"""Validation additions: categorical PSI, continuous rank-ordering
metrics, the grade-level PD back-test, and the train/test/OOT comparison
table -- plus the clearer data_quality_rules error."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from modelmaker.blocks.data_quality import data_quality_rules
from modelmaker.blocks.stat_tests import compare_samples, continuous_accuracy, grade_backtest, psi_test
from modelmaker.cache import CacheStore
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def test_psi_test_on_a_categorical_column():
    exp = pl.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
    act = pl.DataFrame({"c": ["a"] * 50 + ["z"] * 40 + [None] * 10})
    out = psi_test(exp, act, "c")
    assert {b["category"] for b in out["buckets"]} == {"a", "b", "z", "__missing__"}
    assert out["psi"] > 0.25
    assert psi_test(exp, exp, "c")["psi"] == pytest.approx(0.0)


def test_psi_test_numeric_with_nulls_gets_a_null_bucket():
    exp = pl.DataFrame({"x": [float(i) for i in range(100)]})
    act = pl.DataFrame({"x": [float(i) for i in range(80)] + [None] * 20})
    out = psi_test(exp, act, "x", bins=5)
    assert len(out["buckets"]) == 6
    assert out["buckets"][-1]["actual_pct"] == pytest.approx(0.2)


def test_continuous_accuracy_rank_ordering_and_bias():
    actual = np.linspace(0, 1, 50)
    df = pl.DataFrame({"a": actual, "p": actual * 0.5 + 0.3})
    out = continuous_accuracy(df, "a", "p")
    assert out["spearman"] == pytest.approx(1.0)
    assert out["kendall_tau"] == pytest.approx(1.0)
    assert out["accuracy_ratio"] == pytest.approx(1.0)
    assert out["bias"] == pytest.approx(0.55 - 0.5)
    reversed_ = continuous_accuracy(df.with_columns(p=-pl.col("p")), "a", "p")
    assert reversed_["accuracy_ratio"] == pytest.approx(-1.0)


def test_grade_backtest_flags_an_underestimated_grade():
    rng = np.random.default_rng(0)
    rows = []
    for grade, pd_, true_dr in (("1", 0.01, 0.01), ("2", 0.05, 0.05), ("3", 0.10, 0.25)):
        n = 400
        rows += [{"grade": grade, "grade_pd": pd_, "y": int(rng.random() < true_dr)} for _ in range(n)]
    out = grade_backtest(pl.DataFrame(rows), "grade", "y", "grade_pd", confidence=0.99)
    lights = {g["grade"]: g["traffic_light"] for g in out["grades"]}
    assert lights["3"] == "red"
    assert lights["1"] != "red" and lights["2"] != "red"
    assert out["n_red"] == 1
    assert out["herfindahl"] == pytest.approx(1 / 3)
    assert out["monotonic"] is True
    assert out["portfolio"]["n"] == 1200


def test_compare_samples_binary_and_continuous():
    rng = np.random.default_rng(1)
    score = rng.random(500)
    train = pl.DataFrame({"y": (rng.random(500) < score).astype(int), "p": score})
    test = train.sample(200, seed=2)
    table = compare_samples(train, "y", "p", sample_2=test)
    assert table["sample"].to_list() == ["train", "test"]
    assert {"auc", "gini", "ks"} <= set(table.columns)
    assert table["gini"][0] == pytest.approx(2 * table["auc"][0] - 1)

    cont = pl.DataFrame({"lgd": np.linspace(0, 1, 30), "pred": np.linspace(0.1, 0.9, 30)})
    table = compare_samples(cont, "lgd", "pred", sample_3=cont, labels=["dev", "x", "oot"])
    assert table["sample"].to_list() == ["dev", "oot"]
    assert table["spearman"][0] == pytest.approx(1.0)


def test_compare_samples_with_an_unwired_optional_input_runs_in_the_engine(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("y,p\n1,0.9\n0,0.2\n1,0.7\n0,0.4\n")
    graph = Graph(
        blocks={
            "b_read": make_block("b_read", "read_csv", params={"path": str(csv)}),
            "b_cmp": make_block("b_cmp", "compare_samples", params={"target_col": "y", "score_col": "p"}),
        },
        wires={"w1": Wire("w1", "b_read", "out", "b_cmp", "sample_1")},
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_block("b_cmp") == "green", runner.state["b_cmp"].last_error


def test_data_quality_rules_explains_a_malformed_rule():
    with pytest.raises(ValueError, match="apply_exclusions"):
        data_quality_rules(pl.DataFrame({"x": [1]}), [{"name": "bad", "expr": "x > 0"}])
