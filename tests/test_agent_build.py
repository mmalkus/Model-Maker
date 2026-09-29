"""Phase 1 of the AI model builder (see /agent-builder-proposal.md): the tool
layer, its guards, and the build lifecycle, driven by scripted "model"
turns instead of an LLM -- against the real runner and the PD sample data."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelmaker.agent.build import (
    AWAITING_APPROVAL,
    AWAITING_INPUT,
    BUILDING,
    DISCARDED,
    DONE,
    PLANNING,
    PREFLIGHT,
    STOPPED,
    AgentBuild,
    BuildOptions,
    LLMChoice,
)
from modelmaker.agent.controller import BuildController, BuildError
from modelmaker.agent.loop import ScriptedLoop
from modelmaker.runslot import RunSlot
from modelmaker.session import ProjectSession

DATA = Path(__file__).resolve().parents[1] / "sample_data" / "credit_risk_data.csv"

FEATURES = ["credit_score", "dti", "revolving_utilization", "num_late_payments_2yr", "employment_years"]


@pytest.fixture
def prepared(tmp_path):
    """What the user does before a build: an input block, run, roles tagged
    (target, id, and one 'leaky' column excluded)."""
    session = ProjectSession(recovery_path=tmp_path / "recovery.json")
    with session.edit():
        session.set_lane("lane_prep", "Data prep", 0)
        load = session.add_block("read_csv", name="applications", lane="lane_prep", params={"path": str(DATA)})
    assert session.runner.run_block(load.id) == "green"
    with session.edit():
        session.set_column_role(load.id, "default_flag", "target")
        session.set_column_role(load.id, "application_id", "id")
        session.set_column_role(load.id, "interest_rate", "excluded")
    assert session.runner.run_block(load.id) == "green"
    return session, load.id


def make_controller(session, scripts: dict[str, list]):
    """scripts: {"plan": [turn, ...], "build": [turn, ...]}."""
    loops = {}

    def factory(choice, phase):
        loops[phase] = ScriptedLoop(scripts.get(phase, []))
        return loops[phase]

    controller = BuildController(lambda: session, RunSlot(), factory)
    controller.loops = loops
    return controller


def start(controller, anchor, goal="PD model, logistic regression", **options):
    return controller.start(goal, [anchor], LLMChoice("stub"), LLMChoice("stub"), BuildOptions(**options), token="t")


def building(session, anchor) -> AgentBuild:
    """A build already in the building phase, for calling tools directly."""
    b = AgentBuild(session, RunSlot(), "test", [anchor])
    b.phase = BUILDING
    return b


# ---- plan turns used by several tests ---------------------------------------------


def plan_turn(call):
    assert "error" not in call("get_graph", {})
    call("list_block_types", {"group": "modelling"})
    call("describe_block_type", {"category": "logistic_regression"})
    result = call(
        "submit_plan",
        {
            "plan": {
                "summary": "Split, fit a logistic regression, measure Gini on the test set.",
                "assumptions": ["default_flag is the default indicator"],
                "questions": [],
                "lanes": [{"key": "est", "name": "Estimation"}, {"key": "val", "name": "Validation"}],
                "steps": [
                    {"ref": "s1", "category": "train_test_split", "lane": "est", "name": "split",
                     "inputs": [{"port": "df", "from": ANCHOR[0], "from_port": "out"}], "params": {"seed": 1}, "why": "holdout"},
                    {"ref": "s2", "category": "logistic_regression", "lane": "est", "name": "pd model",
                     "inputs": [{"port": "df", "from": "s1", "from_port": "train"}], "params": {"features": FEATURES}, "why": "goal"},
                    {"ref": "s3", "category": "predict", "lane": "val", "name": "score test",
                     "inputs": [{"port": "df", "from": "s1", "from_port": "test"}, {"port": "model", "from": "s2", "from_port": "model"}], "why": "oos"},
                    {"ref": "s4", "category": "auc_gini", "lane": "val", "name": "test gini",
                     "inputs": [{"port": "df", "from": "s3", "from_port": "predictions"}], "why": "discrimination"},
                ],
                "changes_to_existing": [],
            }
        },
    )
    assert result.get("ok"), result
    return "Plan submitted."


ANCHOR: list[str] = []  # set per test so plan_turn can reference the anchor id


def build_turn(call):
    split = call("add_block", {"category": "train_test_split", "lane": "est", "name": "split", "params": {"seed": 1}, "plan_step": "s1"})["block"]
    call("connect", {"from_block": ANCHOR[0], "from_port": "out", "to_block": split, "to_port": "df"})
    assert call("run_to", {"block": split})["status"] == "green"
    fit = call("add_block", {"category": "logistic_regression", "lane": "est", "params": {"features": FEATURES}, "plan_step": "s2"})["block"]
    call("connect", {"from_block": split, "from_port": "train", "to_block": fit, "to_port": "df"})
    ran = call("run_to", {"block": fit})
    assert ran["status"] == "green", ran
    score = call("add_block", {"category": "predict", "lane": "val", "plan_step": "s3"})["block"]
    call("connect", {"from_block": split, "from_port": "test", "to_block": score, "to_port": "df"})
    call("connect", {"from_block": fit, "from_port": "model", "to_block": score, "to_port": "model"})
    gini = call("add_block", {"category": "auc_gini", "lane": "val", "name": "test gini", "plan_step": "s4"})["block"]
    call("connect", {"from_block": score, "from_port": "predictions", "to_block": gini, "to_port": "df"})
    ran = call("run_to", {"block": gini})
    assert ran["status"] == "green", ran
    call("finish", {"report": "Built a logistic PD model.", "key_outputs": [{"block": gini, "port": "metric", "label": "Test Gini"}]})
    return "Done."


# ---- lifecycle ------------------------------------------------------------------------


def test_full_build_plans_builds_and_reports(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    blocks_before = set(session.graph.blocks)
    controller = make_controller(session, {"plan": [plan_turn], "build": [build_turn]})

    b = start(controller, anchor)
    # Warns (no excluded? no -- it has one; target present) -> nothing blocks,
    # so planning starts straight away.
    assert b.preflight["blocking"] == []
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL
    assert b.plan["layout"]["s1"]["x"] >= 0
    assert set(b.plan["lane_layout"]) == {"est", "val"}
    assert set(session.graph.blocks) == blocks_before  # planning changes nothing

    controller.approve()
    assert controller.canvas_locked() or b.phase == DONE
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)
    assert not controller.canvas_locked()

    new = set(session.graph.blocks) - blocks_before
    assert new == b.owned_blocks and len(new) == 4
    for bid in new:
        prov = session.graph.blocks[bid].provenance
        assert prov["source"] == "agent" and prov["build_id"] == b.id and prov["plan_step"]
        assert session.runner.status(bid) == "green"
    assert b.results and b.results[0]["label"] == "Test Gini" and "value" in b.results[0]
    reports = [a for a in session.graph.artifacts.values() if a.kind == "build_report"]
    assert len(reports) == 1 and "Test Gini" in reports[0].document

    # One undo takes the whole build back out, lanes included.
    lanes_before_undo = set(session.graph.lanes)
    assert session.undo() is True
    assert set(session.graph.blocks) == blocks_before
    assert set(session.graph.lanes) == lanes_before_undo - b.owned_lanes


def test_plan_feedback_replans_with_the_feedback(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [plan_turn, plan_turn]})
    b = start(controller, anchor)
    controller.join(60)
    first = b.plan
    controller.feedback("Use seed 1 for the split.")
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL and b.plan is not None and b.plan is not first
    assert b.plan_history[0]["feedback"] == "Use seed 1 for the split."
    assert "Use seed 1" in controller.loops["plan"].prompts[-1]


def test_preflight_blocks_until_the_anchor_has_output(tmp_path):
    session = ProjectSession(recovery_path=tmp_path / "r.json")
    with session.edit():
        load = session.add_block("read_csv", params={"path": str(DATA)})
    controller = make_controller(session, {"plan": []})
    b = start(controller, load.id)
    assert b.phase == PREFLIGHT
    assert [i["code"] for i in b.preflight["blocking"]] == ["anchor_not_run"]
    with pytest.raises(BuildError):
        controller.proceed()
    controller.run_upstream()
    controller.join(30)
    assert not b.preflight["blocking"]
    # Still waits: no target tagged / nothing excluded are warnings.
    assert b.phase == PREFLIGHT and {w["code"] for w in b.preflight["warnings"]} >= {"no_target", "no_excluded"}
    controller.proceed()
    controller.join(30)
    assert b.phase == AWAITING_APPROVAL  # scripted planner had no turns: no plan, awaits feedback
    assert b.plan is None


def test_preflight_warns_about_stale_upstream(prepared):
    session, anchor = prepared
    with session.edit():
        clean = session.add_block("filter", name="clean", params={"expr": "dti >= 0"})
        session.add_wire(anchor, "out", clean.id, "df")
    assert session.runner.run_to_here(clean.id) == "green"
    with session.edit():
        session.update_block(clean.id, params={"expr": "dti >= 1"})  # now orange
    controller = make_controller(session, {})
    b = start(controller, clean.id)
    stale = [w for w in b.preflight["warnings"] if w["code"] == "upstream_not_current"]
    assert stale and stale[0]["blocks"] == [clean.id] and "out of date" in stale[0]["message"]


def test_ask_user_pauses_and_the_answer_resumes(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]

    def asks(call):
        call("ask_user", {"question": "Which seed?"})
        return "Waiting."

    def finishes(call):
        call("finish", {"report": "ok"})
        return "done"

    controller = make_controller(session, {"plan": [plan_turn], "build": [asks, finishes]})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(60)
    assert b.phase == AWAITING_INPUT and b.pending_question == "Which seed?"
    assert controller.canvas_locked()
    controller.feedback("Seed 7.")
    controller.join(120)
    assert b.phase == DONE
    assert "Seed 7." in controller.loops["build"].prompts[-1]


def test_a_turn_that_ends_without_finishing_asks_the_user(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [plan_turn], "build": [lambda call: "I got tired."]})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(60)
    assert b.phase == AWAITING_INPUT and "I got tired." in b.pending_question


def test_discard_puts_the_graph_back(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]

    def partial(call):
        blk = call("add_block", {"category": "train_test_split", "lane": "est"})["block"]
        call("connect", {"from_block": anchor, "from_port": "out", "to_block": blk, "to_port": "df"})
        call("ask_user", {"question": "continue?"})
        return ""

    controller = make_controller(session, {"plan": [plan_turn], "build": [partial]})
    before = (set(session.graph.blocks), set(session.graph.lanes), set(session.graph.wires))
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(60)
    assert len(session.graph.blocks) == len(before[0]) + 1
    controller.discard()
    assert b.phase == DISCARDED
    assert (set(session.graph.blocks), set(session.graph.lanes), set(session.graph.wires)) == before
    assert session.undo() is True  # the discard itself is undoable
    assert len(session.graph.blocks) == len(before[0]) + 1


def test_stop_ends_the_build_and_keeps_what_was_built(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [plan_turn], "build": []})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(60)
    controller.stop()
    assert b.phase == STOPPED and not controller.canvas_locked()
    # Further tool calls are refused.
    assert "stopped" in b.call_tool("get_graph", {})


def test_build_on_a_sample_then_full_run(tmp_path, prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    seen = {}

    def sampled_build(call):
        blk = call("add_block", {"category": "select", "lane": "est", "params": {"cols": ["dti", "default_flag"]}})["block"]
        call("connect", {"from_block": anchor, "from_port": "out", "to_block": blk, "to_port": "df"})
        seen["run"] = call("run_to", {"block": blk})
        call("finish", {"report": "ok", "key_outputs": [{"block": blk}]})
        return ""

    controller = make_controller(session, {"plan": [plan_turn], "build": [sampled_build]})
    b = start(controller, anchor, sample_rows=200)
    controller.join(60)
    controller.approve()
    controller.join(180)
    assert b.phase == DONE, (b.phase, b.pending_question, b.error)
    assert seen["run"]["outputs"]["out"]["row_count"] == 200
    assert session.runner.sample_rows is None  # restored
    assert b.results[0]["row_count"] == 5000  # key output refreshed on the full data
    assert session.runner.status(anchor) == "green"


# ---- guards ------------------------------------------------------------------------------


def test_user_blocks_are_read_only(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    assert "belongs to the user" in b.call_tool("set_params", {"block": anchor, "params": {"path": "x.csv"}})["error"]
    assert "belongs to the user" in b.call_tool("delete_block", {"block": anchor})["error"]
    assert "read-only" in b.call_tool("set_column_role", {"block": anchor, "column": "dti", "role": "excluded"})["error"]
    assert session.graph.blocks[anchor].params == {"path": str(DATA)}


def test_the_build_never_reads_input_blocks_itself(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    assert "input blocks" in b.call_tool("run_to", {"block": anchor})["error"]


def test_approved_changes_are_allowed_and_recorded(prepared):
    session, anchor = prepared
    with session.edit():
        clean = session.add_block("filter", name="clean", params={"expr": "dti >= 0"})
    b = building(session, anchor)
    b.approved_changes[clean.id] = [{"block": clean.id, "change": "tighten filter", "why": "x"}]
    assert b.call_tool("set_params", {"block": clean.id, "params": {"expr": "dti >= 1"}})["ok"]
    prov = session.graph.blocks[clean.id].provenance
    assert "source" not in prov  # still the user's block
    assert prov["changes"][0]["build_id"] == b.id
    assert session.graph.blocks[clean.id].params["expr"] == "dti >= 1"


def test_wiring_into_a_user_block_needs_approval(prepared):
    session, anchor = prepared
    with session.edit():
        clean = session.add_block("filter", name="clean", params={"expr": "dti >= 0"})
    b = building(session, anchor)
    r = b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": clean.id, "to_port": "df"})
    assert "belongs to the user" in r["error"]


def test_excluded_columns_cannot_be_features(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    r = b.call_tool("add_block", {"category": "logistic_regression", "lane": "est", "params": {"features": ["dti", "interest_rate"]}})
    assert "excluded" in r["error"] and "interest_rate" in r["error"]
    code = 'def f(df, a_col: str = "dti"):\n    return df.with_columns(pl.col("interest_rate").alias("ir"))\n'
    r = b.call_tool("add_custom_block", {"lane": "est", "name": "leak", "code": code, "metadata_transform": {"kind": "passthrough"}})
    assert "excluded" in r["error"]


def test_disallowed_and_unknown_blocks_are_refused(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    assert "user" in b.call_tool("add_block", {"category": "read_csv", "lane": "x"})["error"]
    assert "no params" in b.call_tool("add_block", {"category": "filter", "lane": "x", "params": {"nope": 1}})["error"]


def test_output_summary_never_contains_rows_or_text_values(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    out = b.call_tool("get_output_summary", {"block": anchor})
    assert out["row_count"] == 5000 and "rows" not in out
    cols = {c["name"]: c for c in out["columns"]}
    assert "min" not in cols["application_id"] and "min" not in cols["region"]
    assert cols["credit_score"]["min"] > 0
    assert "APP000001" not in str(out) and "West" not in str(out)


def test_custom_block_contract_is_checked(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    bad = b.call_tool("add_custom_block", {"lane": "x", "name": "n", "code": "def f(data):\n    return data\n", "metadata_transform": {"kind": "passthrough"}})
    assert "first parameters" in bad["error"]
    good = b.call_tool(
        "add_custom_block",
        {
            "lane": "Feature engineering",
            "name": "income per year",
            "code": 'def income_ratio(df, income_col: str = "annual_income", loan_col: str = "loan_amount"):\n'
            '    return df.with_columns((pl.col(loan_col) / pl.col(income_col)).alias("loan_to_income"))\n',
            "metadata_transform": {"kind": "declared", "base": "df", "drops": [], "adds": [{"name": "loan_to_income", "dtype": "Float64", "role": "feature"}]},
        },
    )
    assert "block" in good, good
    blk = session.graph.blocks[good["block"]]
    assert blk.is_custom and blk.params == {"income_col": "annual_income", "loan_col": "loan_amount"}
    assert session.graph.lanes[blk.lane].name == "Feature engineering"
    b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": blk.id, "to_port": "df"})
    ran = b.call_tool("run_to", {"block": blk.id})
    assert ran["status"] == "green" and "loan_to_income" in ran["outputs"]["out"]["columns"]


def test_repeated_failures_force_ask_user(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    blk = b.call_tool("add_block", {"category": "filter", "lane": "x", "params": {"expr": "no_such_col > 1"}})["block"]
    b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": blk, "to_port": "df"})
    results = [b.call_tool("run_to", {"block": blk}) for _ in range(3)]
    assert all(r["status"] == "red" for r in results)
    assert "must_ask_user" not in results[1] and "must_ask_user" in results[2]


def test_plan_validation_reports_every_problem(prepared):
    session, anchor = prepared
    b = AgentBuild(session, RunSlot(), "g", [anchor])
    b.phase = PLANNING
    r = b.call_tool(
        "submit_plan",
        {
            "plan": {
                "summary": "x",
                "steps": [
                    {"ref": "s1", "category": "read_csv", "lane": "nowhere", "name": "a", "inputs": [], "why": ""},
                    {"ref": "s2", "category": "auc_gini", "lane": "nowhere", "name": "b", "inputs": [{"port": "df", "from": anchor, "from_port": "nope"}], "why": ""},
                ],
            }
        },
    )
    err = r["error"]
    assert "read_csv can't be added" in err and "lane 'nowhere'" in err and "no output port 'nope'" in err
    assert b.phase == PLANNING and b.plan is None


def test_tool_call_limit_stops_the_build(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    b.options.limits.build_tool_calls = 2
    b.call_tool("get_graph", {})
    b.call_tool("get_graph", {})
    r = b.call_tool("get_graph", {})
    assert r.get("stopped") and b.stop_requested


def test_approve_while_the_ai_is_still_finishing_its_turn_changes_nothing(prepared):
    """Regression: approving while the planner's turn was still running used
    to flip the phase and open the undo transaction before refusing, leaving
    a build stuck in 'building' with nothing running."""
    import threading

    session, anchor = prepared
    ANCHOR[:] = [anchor]
    release = threading.Event()

    def slow_plan(call):
        plan_turn(call)
        release.wait(30)  # a slow model still writing its closing remark
        return "done"

    controller = make_controller(session, {"plan": [slow_plan]})
    b = start(controller, anchor)
    for _ in range(300):
        if b.phase == AWAITING_APPROVAL:
            break
        threading.Event().wait(0.1)
    lanes_before = set(session.graph.lanes)
    with pytest.raises(BuildError, match="finishing its turn"):
        controller.approve()
    assert b.phase == AWAITING_APPROVAL and set(session.graph.lanes) == lanes_before
    assert controller._transaction is None and session._transaction_depth == 0  # nothing opened
    release.set()
    controller.join(30)
    controller.approve()  # now fine
    assert b.phase in (BUILDING, AWAITING_INPUT)
