"""Holdout leakage in a graph's wiring (modelmaker/agent/leakage.py): the
rule on small graphs, then how a build meets it -- a rejected stage plan,
a refused wire, and a concern for leakage among the user's own blocks."""

from __future__ import annotations

import pytest

from modelmaker.agent import catalogue
from modelmaker.agent.leakage import Edge, Node, find_leaks
from modelmaker.blocks.base import BLOCK_REGISTRY
from tests.test_agent_build import prepared, staged  # noqa: F401 -- prepared is a fixture

catalogue.ensure_blocks_registered()


class G:
    """A tiny graph: g.add("fit", "fit_binning", df=("data", "out"))."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add(self, nid, category, **inputs):
        spec = BLOCK_REGISTRY[category]
        self.nodes[nid] = Node(nid, nid, category, {p.name: p.type for p in spec.inputs}, {p.name: p.type for p in spec.outputs})
        for port, (src, src_port) in inputs.items():
            self.edges.append(Edge(src, src_port, nid, port))
        return self

    def leaks(self):
        return {(lk.block, lk.fitter, lk.split) for lk in find_leaks(self.nodes, self.edges)}


def base():
    return G().add("data", "read_csv").add("prep", "select", df=("data", "out")).add("split", "train_test_split", df=("prep", "out"))


def test_binning_fitted_before_the_split_and_applied_to_test_leaks():
    # Run 8's wiring: the binning and the split both fed by the same data.
    g = base().add("fit", "fit_binning", df=("prep", "out"))
    g.add("on_train", "apply_binning", df=("split", "train"), binning=("fit", "binning"))
    assert g.leaks() == set()  # fitting on everything to look at it, or applying to train, is fine
    g.add("on_test", "apply_binning", df=("split", "test"), binning=("fit", "binning"))
    assert g.leaks() == {("on_test", "fit", "split")}


def test_fitting_on_train_is_clean():
    g = base().add("fit", "fit_binning", df=("split", "train"))
    g.add("on_test", "apply_binning", df=("split", "test"), binning=("fit", "binning"))
    g.add("model", "logistic_regression", df=("split", "train"))
    g.add("score", "predict", df=("on_test", "out"), model=("model", "model"))
    assert g.leaks() == set()


def test_a_fit_is_followed_through_calibration():
    g = base().add("model", "logistic_regression", df=("split", "train"))
    g.add("cal", "calibrate_model", df=("prep", "out"), model=("model", "model"))  # calibrated on everything
    g.add("score", "predict", df=("split", "test"), model=("cal", "model"))
    assert g.leaks() == {("score", "cal", "split")}


def test_a_model_fitted_on_the_test_sample_leaks_there():
    g = base().add("model", "logistic_regression", df=("split", "test"))
    g.add("score", "predict", df=("split", "test"), model=("model", "model"))
    assert g.leaks() == {("score", "model", "split")}


def test_filters_keep_the_rows_lineage():
    g = base().add("fit", "fit_binning", df=("prep", "out"))
    g.add("some_test", "filter", df=("split", "test"))
    g.add("on_test", "apply_binning", df=("some_test", "out"), binning=("fit", "binning"))
    assert g.leaks() == {("on_test", "fit", "split")}


def test_nested_splits():
    # Out-of-time first, then train/test on the development sample.
    g = G().add("data", "read_csv").add("oot", "time_split", df=("data", "out"))
    g.add("split", "train_test_split", df=("oot", "development"))
    g.add("model", "logistic_regression", df=("split", "train"))
    g.add("on_test", "predict", df=("split", "test"), model=("model", "model"))
    g.add("on_oot", "predict", df=("oot", "out_of_time"), model=("model", "model"))
    assert g.leaks() == set()
    # A binning on the whole development sample saw the test rows, not the out-of-time ones.
    g.add("fit", "fit_binning", df=("oot", "development"))
    g.add("b_test", "apply_binning", df=("split", "test"), binning=("fit", "binning"))
    g.add("b_oot", "apply_binning", df=("oot", "out_of_time"), binning=("fit", "binning"))
    assert g.leaks() == {("b_test", "fit", "split")}


def test_a_separate_dataset_is_not_a_holdout_of_another():
    g = base().add("other", "read_csv").add("model", "logistic_regression", df=("other", "out"))
    g.add("score", "predict", df=("split", "test"), model=("model", "model"))
    assert g.leaks() == set()


def test_woe_transform_on_a_holdout_leaks():
    g = base().add("woe", "woe_transform", df=("split", "test"))
    assert g.leaks() == {("woe", "woe", "split")}
    assert base().add("woe", "woe_transform", df=("split", "train")).leaks() == set()


# ---- in a build -----------------------------------------------------------------------


@pytest.fixture
def staged_build(prepared):  # noqa: F811
    session, anchor = prepared
    return staged(session, anchor), anchor


def _leaky_steps(anchor):
    return [
        {"ref": "lk1", "category": "fit_binning", "name": "univariate", "why": "", "params": {"features": ["credit_score"]},
         "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]},
        {"ref": "lk2", "category": "train_test_split", "name": "split", "why": "", "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]},
        {"ref": "lk3", "category": "apply_binning", "name": "bin test", "why": "",
         "inputs": [{"port": "df", "from": "lk2", "from_port": "test"}, {"port": "binning", "from": "lk1", "from_port": "binning"}]},
    ]


def test_a_leaky_stage_plan_is_rejected(staged_build):
    b, anchor = staged_build
    r = b.call_tool("plan_stage", {"steps": _leaky_steps(anchor)})
    assert "step lk3:" in r["error"] and "fitted on data that includes that sample" in r["error"]
    assert "the split's train output" in r["error"]
    assert not b.owned_blocks  # nothing built

    fixed = _leaky_steps(anchor)
    fixed[0]["inputs"] = [{"port": "df", "from": "lk2", "from_port": "train"}]
    fixed = [fixed[1], fixed[0], fixed[2]]
    r = b.call_tool("plan_stage", {"steps": fixed})
    assert r.get("ok"), r


def test_a_leaky_wire_is_refused(staged_build):
    b, anchor = staged_build
    fit = b.call_tool("add_block", {"category": "fit_binning", "lane": "est", "params": {"features": ["credit_score"]}})["block"]
    split = b.call_tool("add_block", {"category": "train_test_split", "lane": "est"})["block"]
    apply = b.call_tool("add_block", {"category": "apply_binning", "lane": "est"})["block"]
    for blk in (fit, split):
        assert "wire" in b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": blk, "to_port": "df"})
    assert "wire" in b.call_tool("connect", {"from_block": fit, "from_port": "binning", "to_block": apply, "to_port": "binning"})
    r = b.call_tool("connect", {"from_block": split, "from_port": "test", "to_block": apply, "to_port": "df"})
    assert "includes that sample" in r["error"] and "wire it themselves" in r["error"]
    assert "wire" in b.call_tool("connect", {"from_block": split, "from_port": "train", "to_block": apply, "to_port": "df"})


def test_leakage_among_the_users_blocks_is_a_concern(staged_build, prepared):  # noqa: F811
    b, anchor = staged_build
    session, _ = prepared
    with session.edit():  # the user's own (leaky) wiring
        fit = session.add_block("fit_binning", name="my binning", lane="lane_prep", params={"features": ["credit_score"]})
        split = session.add_block("train_test_split", name="my split", lane="lane_prep")
        apply = session.add_block("apply_binning", name="my test bins", lane="lane_prep")
        session.add_wire(anchor, "out", fit.id, "df")
        session.add_wire(anchor, "out", split.id, "df")
        session.add_wire(fit.id, "binning", apply.id, "binning")
        session.add_wire(split.id, "test", apply.id, "df")
    r = b.call_tool("plan_stage", {"steps": [
        {"ref": "u1", "category": "train_test_split", "name": "split", "why": "", "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]}]})
    assert r.get("ok"), r
    assert any("'my binning'" in c["what"] and "'my test bins'" in c["what"] for c in b.concerns)


def test_an_outline_must_split_before_it_learns_from_the_target(prepared):  # noqa: F811
    from modelmaker.agent.build import PLANNING
    from tests.test_agent_build import building

    session, anchor = prepared
    b = building(session, anchor)
    b.phase = PLANNING

    def outline(*stages):
        return {"plan": {"summary": "x", "stages": [
            {"key": k, "name": k, "goal": "g", "blocks": blocks} for k, blocks in stages]}}

    r = b.call_tool("submit_plan", outline(("uni", ["fit_binning"]), ("split", ["train_test_split"]), ("est", ["logistic_regression"])))
    assert "stage uni: fit_binning learn(s) from the target" in r["error"] and "later stage (split)" in r["error"]
    # Split first -- or in the same stage, or not at all -- is fine.
    for ok in (
        outline(("split", ["train_test_split"]), ("uni", ["fit_binning"])),
        outline(("est", ["train_test_split", "fit_binning", "logistic_regression"])),
        outline(("prep", ["select"]), ("uni", ["fit_binning"])),
    ):
        b.plan = None
        b.phase = PLANNING
        assert b.call_tool("submit_plan", ok).get("ok"), ok
