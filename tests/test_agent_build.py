"""Phase 1 of the AI model builder (see /agent-builder-proposal.md): the tool
layer, its guards, and the build lifecycle, driven by scripted "model"
turns instead of an LLM -- against the real runner and the PD sample data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelmaker.agent.build import (
    AWAITING_APPROVAL,
    AWAITING_INPUT,
    AWAITING_STAGE_REVIEW,
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
from modelmaker.agent.loop import LLMUnavailable, ScriptedLoop
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


ANCHOR: list[str] = []  # set per test so the turns can reference the anchor id
BUILT: dict[str, str] = {}  # plan step ref -> block id, for later stages' wiring


def plan_turn(call):
    assert "error" not in call("get_graph", {})
    call("list_block_types", {"tag": "regression"})
    call("describe_block_type", {"category": "logistic_regression"})
    result = call(
        "submit_plan",
        {
            "plan": {
                "summary": "Split, fit a logistic regression, measure Gini on the test set.",
                "assumptions": ["default_flag is the default indicator"],
                "questions": [],
                "stages": [
                    {"key": "est", "name": "Estimation", "goal": "70/30 split; logistic regression on the numeric drivers"},
                    {"key": "val", "name": "Validation", "goal": "Score the test set and report its Gini"},
                ],
                "changes_to_existing": [],
            }
        },
    )
    assert result.get("ok"), result
    return "Plan submitted."


def plan_one_stage(call):
    result = call(
        "submit_plan",
        {"plan": {"summary": "Split it.", "stages": [{"key": "est", "name": "Estimation", "goal": "holdout split"}]}},
    )
    assert result.get("ok"), result
    return "Plan submitted."


EST_STEPS = [
    {"ref": "s1", "category": "train_test_split", "name": "split", "params": {"seed": 1}, "why": "holdout",
     "inputs": [{"port": "df", "from": "ANCHOR", "from_port": "out"}]},
    {"ref": "s2", "category": "logistic_regression", "name": "pd model", "params": {"features": FEATURES}, "why": "goal",
     "inputs": [{"port": "df", "from": "s1", "from_port": "train"}]},
]


def est_steps():
    steps = json.loads(json.dumps(EST_STEPS))
    steps[0]["inputs"][0]["from"] = ANCHOR[0]
    return steps


def build_est(call):
    # plan_stage builds the plan too: add, wire and run each step in order.
    built = call("plan_stage", {"steps": est_steps()})
    assert built.get("ok"), built
    assert [(st["ref"], st["status"]) for st in built["steps"]] == [("s1", "green"), ("s2", "green")]
    BUILT.update({st["ref"]: st["block"] for st in built["steps"]})
    done = call("complete_stage", {"summary": "Split 70/30 and fitted the PD model on 5 drivers."})
    assert done.get("ok"), done
    return "Estimation done."


def build_val(call):
    split, fit = BUILT["s1"], BUILT["s2"]
    built = call(
        "plan_stage",
        {
            "steps": [
                {"ref": "s3", "category": "predict", "name": "score test", "why": "oos",
                 "inputs": [{"port": "df", "from": split, "from_port": "test"}, {"port": "model", "from": fit, "from_port": "model"}]},
                {"ref": "s4", "category": "auc_gini", "name": "test gini", "why": "discrimination",
                 "inputs": [{"port": "df", "from": "s3", "from_port": "predictions"}]},
            ]
        },
    )
    assert built.get("ok"), built
    gini = built["steps"][-1]["block"]
    call("finish", {"report": "Built a logistic PD model.", "key_outputs": [{"block": gini, "port": "metric", "label": "Test Gini"}]})
    return "Done."


BUILD_TURNS = [build_est, build_val]


# ---- lifecycle ------------------------------------------------------------------------


def test_full_build_plans_builds_and_reports(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    blocks_before = set(session.graph.blocks)
    controller = make_controller(session, {"plan": [plan_turn], "build": list(BUILD_TURNS)})

    b = start(controller, anchor)
    # Warns (no excluded? no -- it has one; target present) -> nothing blocks,
    # so planning starts straight away.
    assert b.preflight["blocking"] == []
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL
    assert set(b.plan["lane_layout"]) == {"est", "val"}
    assert set(session.graph.blocks) == blocks_before  # planning changes nothing

    controller.approve()
    controller.join(300)
    # The first stage is built, then the build waits for the user's review.
    assert b.phase == AWAITING_STAGE_REVIEW, (b.phase, b.error, b.pending_question)
    assert controller.canvas_locked()
    est, val = b.stages
    assert est["status"] == "done" and "5 drivers" in est["summary"] and val["status"] == "pending"
    assert [s["ref"] for s in est["plan"]["steps"]] == ["s1", "s2"]
    # Each stage's lane exists from the start.
    assert {session.graph.lanes[b.lane_map[k]].name for k in ("est", "val")} == {"Estimation", "Validation"}
    # Each block landed where its ghost was drawn.
    for ref in ("s1", "s2"):
        ghost = est["plan"]["layout"][ref]
        real = session.graph.blocks[BUILT[ref]]
        assert (real.position.x, real.position.y, real.lane) == (ghost["x"], ghost["y"], ghost["lane"])

    controller.approve()  # on to the next stage
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)
    assert not controller.canvas_locked()
    assert val["status"] == "done"
    assert "Stage 2 of 2: Validation" in controller.loops["build"].prompts[-1]

    new = set(session.graph.blocks) - blocks_before
    assert new == b.owned_blocks and len(new) == 4
    for bid in new:
        prov = session.graph.blocks[bid].provenance
        assert prov["source"] == "agent" and prov["build_id"] == b.id and prov["plan_step"]
        assert prov["stage"] == ("est" if prov["plan_step"] in ("s1", "s2") else "val")
        assert session.runner.status(bid) == "green"
    assert b.results and b.results[0]["label"] == "Test Gini" and "value" in b.results[0]
    reports = [a for a in session.graph.artifacts.values() if a.kind == "build_report"]
    assert len(reports) == 1 and "Test Gini" in reports[0].document and "5 drivers" in reports[0].document

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

    controller = make_controller(session, {"plan": [plan_one_stage], "build": [asks, finishes]})
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

    controller = make_controller(session, {"plan": [plan_one_stage], "build": [sampled_build]})
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
                "stages": [
                    {"key": "est", "name": "Estimation", "goal": "fit"},
                    {"key": "est", "name": "Again", "goal": ""},
                    {"key": "val", "name": "Validation", "goal": "check", "lane": "lane_nowhere"},
                ],
                "changes_to_existing": [{"block": "blk_missing", "change": "x", "why": "y"}],
            }
        },
    )
    err = r["error"]
    assert "key must be unique" in err and "needs a goal" in err and "'lane_nowhere'" in err and "'blk_missing'" in err
    assert b.phase == PLANNING and b.plan is None
    assert "no stages" in b.call_tool("submit_plan", {"plan": {"summary": "x", "stages": []}})["error"]


def staged(session, anchor, stages=("est", "val")) -> AgentBuild:
    """A build in its first stage, for calling stage tools directly."""
    b = building(session, anchor)
    b.plan = {"summary": "x", "stages": [{"key": k, "name": k.title(), "goal": "g"} for k in stages]}
    b.stages = [{**st, "status": "pending", "plan": None, "summary": None} for st in b.plan["stages"]]
    b.stages[0]["status"] = "active"
    return b


def test_stage_plan_validation_reports_every_problem(prepared):
    session, anchor = prepared
    b = staged(session, anchor)
    r = b.call_tool(
        "plan_stage",
        {
            "steps": [
                {"ref": "s1", "category": "read_csv", "lane": "nowhere", "name": "a", "inputs": [], "why": ""},
                {"ref": "s2", "category": "auc_gini", "name": "b", "inputs": [{"port": "df", "from": anchor, "from_port": "nope"}], "why": ""},
            ]
        },
    )
    err = r["error"]
    assert "read_csv can't be added" in err and "lane 'nowhere'" in err and "no output port 'nope'" in err
    assert b.stages[0]["plan"] is None

    # A step without a lane goes into the stage's own.
    ok = b.call_tool("plan_stage", {"steps": [{"ref": "s1", "category": "train_test_split", "name": "split", "why": "",
                                               "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]}]})
    assert ok.get("ok"), ok
    assert b.stages[0]["plan"]["steps"][0]["lane"] == "est"

    # Refs stay unique across stages, and earlier stages' steps are wired by block id.
    b.stages[0]["status"], b.stages[1]["status"], b.stage_index = "done", "active", 1
    r = b.call_tool("plan_stage", {"steps": [{"ref": "s1", "category": "auc_gini", "name": "g", "why": "",
                                              "inputs": [{"port": "df", "from": "s1", "from_port": "test"}]}]})
    assert "used by an earlier stage" in r["error"] and "wire from the block it built" in r["error"]


def test_stage_tools_keep_to_the_outline(prepared):
    session, anchor = prepared
    b = staged(session, anchor)
    assert "plan_stage for this stage" in b.call_tool("complete_stage", {"summary": "x"})["error"]
    r = b.call_tool("finish", {"report": "early"})
    assert "stages still to build after this one: Val" in r["error"] and not b.finished

    b.call_tool("plan_stage", {"steps": [{"ref": "s1", "category": "train_test_split", "name": "split", "why": "",
                                          "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]}]})
    assert b.call_tool("complete_stage", {"summary": "split done"})["ok"]
    assert b.stages[0]["status"] == "done" and b.turn_over
    # Nothing more until the user has reviewed the stage.
    assert "waiting for the user's review" in b.call_tool("plan_stage", {"steps": []})["error"]
    assert "waiting for the user's review" in b.call_tool("add_block", {"category": "filter", "lane": "val"})["error"]

    b.stages[1]["status"], b.stage_index = "active", 1
    assert "call finish instead" in b.call_tool("complete_stage", {"summary": "x"})["error"]
    assert b.call_tool("finish", {"report": "ok"})["ok"] and b.stages[1]["status"] == "done"


def test_finish_can_drop_stages_the_user_agreed_to_drop(prepared):
    session, anchor = prepared
    b = staged(session, anchor, stages=("est", "val", "cal"))
    r = b.call_tool("finish", {"report": "ok", "dropped_stages_reason": "the user only wants the fit for now"})
    assert r["ok"] and [st["status"] for st in b.stages] == ["done", "skipped", "skipped"]
    assert [d["what"] for d in b.deviations] == ["dropped stage Val", "dropped stage Cal"]


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


# ---- auto-build and the persisted log ---------------------------------------------------


def test_auto_build_builds_without_waiting_for_approval(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    blocks_before = set(session.graph.blocks)
    controller = make_controller(session, {"plan": [plan_turn], "build": list(BUILD_TURNS)})
    b = start(controller, anchor, auto_build=True)
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)
    kinds = [e["kind"] for e in b.events_since()]
    assert "auto_approved" in kinds and "auto_continued" in kinds  # no stops for the plan or the stage review
    assert len(set(session.graph.blocks) - blocks_before) == 4
    # Still one undo step, like an approved build.
    assert session.undo() is True
    assert set(session.graph.blocks) == blocks_before


def test_auto_build_waits_when_the_plan_has_questions(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]

    def asks_in_plan(call):
        result = call(
            "submit_plan",
            {
                "plan": {
                    "summary": "split",
                    "questions": ["Which seed?"],
                    "stages": [{"key": "est", "name": "Estimation", "goal": "holdout split"}],
                }
            },
        )
        assert result.get("ok"), result
        return "planned"

    controller = make_controller(session, {"plan": [asks_in_plan, plan_turn], "build": list(BUILD_TURNS)})
    b = start(controller, anchor, auto_build=True)
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL
    assert any(e["kind"] == "auto_build_paused" for e in b.events_since())
    # Answering re-plans; the new plan has no questions, so it builds.
    controller.feedback("Seed 1.")
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)


def test_the_build_log_is_saved_with_models_and_timings(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [plan_turn], "build": list(BUILD_TURNS)})
    b = controller.start("PD model", [anchor], LLMChoice("anthropic", "some-model"), LLMChoice("openai", "other-model"), BuildOptions(), token="t")
    controller.join(60)
    # Written while the build is still waiting for approval, not only at the end.
    assert b.log_path is not None and b.log_path.is_file()
    controller.approve()
    controller.join(300)
    controller.approve()  # past the first stage's review
    controller.join(300)
    assert b.phase == DONE

    log = json.loads(b.log_path.read_text(encoding="utf-8"))
    assert log["id"] == b.id and log["goal"] == "PD model" and log["phase"] == DONE
    assert log["models"]["plan"] == {"provider": "anthropic", "model": "some-model", "label": "anthropic/some-model"}
    assert log["models"]["build"]["model"] == "other-model"
    assert log["created_at"] and log["ended_at"] and log["duration_seconds"] >= 0
    assert log["options"]["auto_build"] is False
    assert log["plan"]["summary"] and log["results"] and log["usage"] is not None
    assert [st["status"] for st in log["stages"]] == ["done", "done"] and log["stages"][0]["summary"]
    kinds = [e["kind"] for e in log["events"]]
    assert "tool" in kinds and kinds[-1] == "phase"
    assert all("at" in e for e in log["events"])


def test_the_build_log_lives_in_the_project_folder_once_saved(prepared, tmp_path):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [plan_turn]})
    b = start(controller, anchor)
    controller.join(60)
    cache_copy = b.log_path
    assert cache_copy is not None and cache_copy.is_file()

    session.save(tmp_path / "proj" / "model.json")
    controller.stop()
    assert b.log_path == tmp_path / "proj" / "ai_builds" / f"{b.id}.json"
    assert json.loads(b.log_path.read_text(encoding="utf-8"))["phase"] == STOPPED
    assert not cache_copy.exists()


def _model_times_out(call):
    raise LLMUnavailable("local model server at http://x/v1 did not respond within 300s")


def test_a_model_timeout_while_planning_pauses_for_a_retry(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [_model_times_out, plan_turn]})
    b = start(controller, anchor)
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL and b.plan is None
    assert "did not respond" in b.pending_question
    controller.feedback("continue")
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL and b.plan is not None and b.pending_question is None


def test_a_model_timeout_while_building_keeps_the_build(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]

    def adds_then_times_out(call):
        split = call("add_block", {"category": "train_test_split", "lane": "est", "name": "split", "plan_step": "s1"})["block"]
        call("connect", {"from_block": ANCHOR[0], "from_port": "out", "to_block": split, "to_port": "df"})
        _model_times_out(call)

    controller = make_controller(session, {"plan": [plan_one_stage], "build": [adds_then_times_out, lambda call: call("finish", {"report": "ok"}) and ""]})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(60)
    assert b.phase == AWAITING_INPUT and "did not respond" in b.pending_question
    assert len(b.owned_blocks) == 1  # what was built so far is kept
    controller.feedback("continue")
    controller.join(120)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)


# ---- stage reviews ------------------------------------------------------------------------


def test_stage_review_feedback_reworks_the_stage(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]

    def rework(call):
        fit = BUILT["s2"]
        assert call("set_params", {"block": fit, "params": {"features": FEATURES[:3]}})["ok"]
        assert call("run_to", {"block": fit})["status"] == "green"
        assert call("complete_stage", {"summary": "Refitted on 3 drivers."})["ok"]
        return "Reworked."

    controller = make_controller(session, {"plan": [plan_turn], "build": [build_est, rework, build_val]})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(300)
    assert b.phase == AWAITING_STAGE_REVIEW
    with pytest.raises(BuildError):
        controller.feedback("   ")

    controller.feedback("Keep only the first three drivers.")
    controller.join(300)
    # Still the same stage, back for review with the new summary.
    assert b.phase == AWAITING_STAGE_REVIEW and b.stage_index == 0
    assert b.stages[0]["summary"] == "Refitted on 3 drivers."
    prompt = controller.loops["build"].prompts[-1]
    assert "Keep only the first three drivers." in prompt and "'Estimation'" in prompt

    controller.approve()
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)


def test_stop_during_a_stage_review(prepared):
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    controller = make_controller(session, {"plan": [plan_turn], "build": list(BUILD_TURNS)})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(300)
    assert b.phase == AWAITING_STAGE_REVIEW
    controller.stop()
    assert b.phase == STOPPED and not controller.canvas_locked()
    # The empty lane of the stage never built is cleaned up; the built one stays.
    assert b.lane_map["val"] not in session.graph.lanes and b.lane_map["est"] in session.graph.lanes


def test_waiting_on_the_user_doesnt_count_against_the_time_limit(prepared):
    session, anchor = prepared
    b = AgentBuild(session, RunSlot(), "g", [anchor])
    b.started_monotonic -= 3600  # an hour in preflight (waiting on the user)
    b._waiting_since -= 3600
    b.set_phase(PLANNING)
    assert b.active_seconds() < 60
    b.check_stop()  # within the limit


# ---- statistics tables -------------------------------------------------------------------


def _binned(session, anchor, b):
    fit = b.call_tool("add_block", {"category": "fit_binning", "lane": "est", "params": {"features": FEATURES + ["region"]}})["block"]
    b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": fit, "to_port": "df"})
    ran = b.call_tool("run_to", {"block": fit})
    assert ran["status"] == "green", ran
    return fit, ran


def test_statistics_tables_show_their_rows(prepared):
    session, anchor = prepared
    b = building(session, anchor)
    fit, ran = _binned(session, anchor, b)
    assert ran["outputs"]["summary"]["type"] == "statistics_table"

    summary = b.call_tool("get_output_summary", {"block": fit, "port": "summary"})
    assert summary["type"] == "statistics_table" and summary["row_count"] == len(FEATURES) + 1
    by_feature = {r["feature"]: r for r in summary["rows"]}
    assert set(by_feature) == set(FEATURES) | {"region"}
    assert by_feature["credit_score"]["iv"] > 0 and by_feature["credit_score"]["iv_band"]
    # Category labels are shown in the bins table: fit_binning pools any
    # category under its minimum share, so a label is never one customer's.
    bins = b.call_tool("get_output_summary", {"block": fit, "port": "bins"})
    assert any(r["feature"] == "region" and "West" in str(r["label"]) for r in bins["rows"])
    # ...but the data itself still never reaches the model.
    assert "West" not in str(b.call_tool("get_output_summary", {"block": anchor}))


def test_statistics_tables_are_capped(prepared, monkeypatch):
    from modelmaker.agent import tools

    session, anchor = prepared
    b = building(session, anchor)
    fit, _ = _binned(session, anchor, b)
    monkeypatch.setattr(tools, "TABLE_ROW_LIMIT", 3)
    bins = b.call_tool("get_output_summary", {"block": fit, "port": "bins"})
    assert len(bins["rows"]) == 3 and "the first 3 of" in bins["rows_shown"]
    monkeypatch.setattr(tools, "TABLE_ROW_LIMIT", 500)
    monkeypatch.setattr(tools, "TABLE_CHAR_LIMIT", 600)
    bins = b.call_tool("get_output_summary", {"block": fit, "port": "bins"})
    assert 0 < len(bins["rows"]) < bins["row_count"] and len(json.dumps(bins["rows"])) <= 600


def test_custom_blocks_never_count_as_statistics_tables(prepared):
    """A custom block can return anything -- df.head(10) included -- so its
    rows stay hidden even when it only reshapes a statistics table."""
    session, anchor = prepared
    b = building(session, anchor)
    fit, _ = _binned(session, anchor, b)
    code = 'def top_iv(summary, iv_col: str = "iv"):\n    return summary.filter(pl.col(iv_col) > 0.02)\n'
    custom = b.call_tool("add_custom_block", {"lane": "est", "name": "strong", "inputs": ["summary"], "code": code,
                                              "metadata_transform": {"kind": "passthrough"}})["block"]
    b.call_tool("connect", {"from_block": fit, "from_port": "summary", "to_block": custom, "to_port": "summary"})
    assert b.call_tool("run_to", {"block": custom})["status"] == "green"
    out = b.call_tool("get_output_summary", {"block": custom})
    assert out["type"] == "dataframe" and "rows" not in out and "credit_score" not in json.dumps(out)


def test_only_dataframe_outputs_can_be_statistics_tables():
    from modelmaker.blocks.base import BlockSpec, PortSpec, register_block

    spec = BlockSpec("x_bad", "standard", "x", [PortSpec("df")], [PortSpec("model", type="model")], fn=lambda df: df,
                     metadata_transform=lambda *a: None, aggregate_outputs=("model",))
    with pytest.raises(ValueError, match="aren't dataframe outputs"):
        register_block(spec)


def test_the_catalogue_marks_statistics_tables():
    from modelmaker.agent import catalogue

    outs = {p["name"]: p for p in catalogue.describe_block_type("fit_binning")["outputs"]}
    assert outs["summary"].get("statistics_table") and "statistics_table" not in outs["binning"]
    assert "statistics_table" not in catalogue.describe_block_type("logistic_regression")["outputs"][0]


# ---- custom blocks turned off ------------------------------------------------------------


def test_custom_blocks_can_be_turned_off(prepared):
    from modelmaker.agent import prompts
    from modelmaker.agent.tools import tools_for_phase

    session, anchor = prepared
    b = staged(session, anchor)
    b.options.allow_custom_blocks = False
    names = {t.name for t in tools_for_phase(BUILDING, allow_custom=False)}
    assert "add_block" in names and not names & {"add_custom_block", "update_custom_block"}
    code = 'def f(df):\n    return df\n'
    r = b.call_tool("add_custom_block", {"lane": "est", "name": "n", "code": code, "metadata_transform": {"kind": "passthrough"}})
    assert "turned off" in r["error"]
    r = b.call_tool("plan_stage", {"steps": [{"ref": "s1", "category": "custom", "instruction": "x", "name": "n", "why": "",
                                              "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]}]})
    assert "turned off" in r["error"]
    system = prompts.build_system(b)
    assert "Custom blocks are turned off" in system and "add_custom_block" not in system
    assert "add_custom_block" not in prompts.plan_system(b)


def test_a_suspicious_feature_in_a_model_is_flagged_by_the_app(prepared):
    session, anchor = prepared
    b = staged(session, anchor)
    fit, _ = _binned(session, anchor, b)
    summary = b.call_tool("get_output_summary", {"block": fit, "port": "summary"})
    assert {r["feature"]: r["iv_band"] for r in summary["rows"]}["credit_score"] == "suspicious"

    # Selecting the column isn't using it as a driver.
    picked = b.call_tool("add_block", {"category": "select", "lane": "est", "params": {"cols": ["credit_score", "dti"]}})
    assert "warning" not in picked and b.concerns == []

    model = b.call_tool("add_block", {"category": "logistic_regression", "lane": "est", "name": "pd model",
                                      "params": {"features": ["credit_score_woe", "dti_woe"]}})
    assert "credit_score" in model["warning"] and "ask_user" in model["warning"]
    assert len(b.concerns) == 1 and b.concerns[0]["stage"] == "est"
    assert "pd model uses credit_score (IV" in b.concerns[0]["what"]
    # Flagged once per block, however often its params change.
    again = b.call_tool("set_params", {"block": model["block"], "params": {"features": ["credit_score_woe"]}})
    assert "warning" in again and len(b.concerns) == 1
    assert b.call_tool("set_params", {"block": model["block"], "params": {"features": ["dti_woe"]}}).get("warning") is None
    assert any(e["kind"] == "concern" for e in b.events_since())



# ---- the app builds the stage plan ----------------------------------------------------


def test_plan_stage_build_stops_at_a_failure_and_build_stage_resumes(prepared):
    session, anchor = prepared
    b = staged(session, anchor)
    steps = [
        {"ref": "est1", "category": "filter", "name": "bad filter", "params": {"expr": "no_such_col > 1"}, "why": "",
         "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]},
        {"ref": "est2", "category": "select", "name": "pick", "params": {"cols": ["dti", "default_flag"]}, "why": "",
         "inputs": [{"port": "df", "from": "est1", "from_port": "out"}]},
    ]
    r = b.call_tool("plan_stage", {"steps": steps})
    assert r["ok"] is False and r["stopped_at"] == "est1" and r["steps"][0]["status"] == "red" and r["steps"][0]["error"]
    assert len(b.owned_blocks) == 1  # est2 wasn't started

    bad = r["steps"][0]["block"]
    b.call_tool("set_params", {"block": bad, "params": {"expr": "dti > 1"}})
    r = b.call_tool("build_stage", {})
    assert r["ok"] and [st["status"] for st in r["steps"]] == ["green", "green"], r
    assert r["steps"][1]["outputs"]["out"]["columns"] == ["dti", "default_flag"]
    assert [e["status"] for e in b.events_since() if e["kind"] == "step"] == ["red", "green", "green"]

    # Re-planning keeps built steps, applying changed params.
    steps[1]["params"] = {"cols": ["dti", "credit_score", "default_flag"]}
    r = b.call_tool("plan_stage", {"steps": steps})
    assert r["ok"] and len(b.owned_blocks) == 2
    assert session.graph.blocks[r["steps"][1]["block"]].params["cols"] == ["dti", "credit_score", "default_flag"]
    assert "credit_score" in r["steps"][1]["outputs"]["out"]["columns"]


def test_a_custom_step_hands_back_to_the_model(prepared):
    session, anchor = prepared
    b = staged(session, anchor)
    steps = [
        {"ref": "est1", "category": "custom", "instruction": "loan to income", "name": "ratio", "why": "",
         "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]},
        {"ref": "est2", "category": "select", "name": "pick", "params": {"cols": ["loan_to_income"]}, "why": "",
         "inputs": [{"port": "df", "from": "est1", "from_port": "out"}]},
    ]
    r = b.call_tool("plan_stage", {"steps": steps})
    assert r["stopped_at"] == "est1" and r["steps"][0]["status"] == "needs_code" and "add_custom_block" in r["next"]
    code = ('def ratio(df, a_col: str = "loan_amount", b_col: str = "annual_income"):\n'
            '    return df.with_columns((pl.col(a_col) / pl.col(b_col)).alias("loan_to_income"))\n')
    custom = b.call_tool("add_custom_block", {"lane": "est", "name": "ratio", "code": code, "plan_step": "est1",
                                              "metadata_transform": {"kind": "declared", "base": "df", "drops": [],
                                                                     "adds": [{"name": "loan_to_income", "dtype": "Float64", "role": "feature"}]}})
    b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": custom["block"], "to_port": "df"})
    r = b.call_tool("build_stage", {})
    assert r["ok"] and [st["status"] for st in r["steps"]] == ["green", "green"], r


def test_plan_stage_can_just_plan(prepared):
    session, anchor = prepared
    b = staged(session, anchor)
    r = b.call_tool("plan_stage", {"build": False, "steps": [
        {"ref": "est1", "category": "train_test_split", "name": "split", "why": "",
         "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]}]})
    assert r["ok"] and "steps" not in r and not b.owned_blocks
    assert b.call_tool("build_stage", {})["steps"][0]["status"] == "green"


def test_the_build_prompt_carries_only_the_block_tags(prepared):
    from modelmaker.agent import prompts

    session, anchor = prepared
    b = staged(session, anchor)
    system = prompts.build_system(b)
    assert "## Registry block tags" in system and "## Registry blocks\n" not in system
    assert "## Registry blocks\n" in prompts.plan_system(b)


# ---- what a stage starts with ------------------------------------------------------------


def test_a_stage_starts_with_what_is_built_and_the_blocks_it_needs(prepared):
    from modelmaker.agent import prompts

    session, anchor = prepared
    b = staged(session, anchor)
    b.stages[0]["goal"] = "Split, then fit_binning on train"
    b.call_tool("plan_stage", {"steps": [
        {"ref": "est1", "category": "train_test_split", "name": "split", "why": "",
         "inputs": [{"port": "df", "from": anchor, "from_port": "out"}]},
        {"ref": "est2", "category": "fit_binning", "name": "bin", "params": {"features": FEATURES}, "why": "",
         "inputs": [{"port": "df", "from": "est1", "from_port": "train"}]},
    ]})
    b.stages[0]["status"], b.stages[1]["status"], b.stage_index = "done", "active", 1
    b.stages[1]["blocks"] = ["apply_binning", "logistic_regression"]
    b.stages[1]["goal"] = "Fit a model, then compare_samples"

    prompt = prompts.stage_prompt(b)
    # Every block so far, with its outputs: columns once, then "same as".
    assert f"(id {anchor}; read_csv, anchor; green)" in prompt and "interest_rate [excluded]" in prompt
    assert "default_flag [target]" in prompt and "application_id [id]" in prompt
    assert "train: 4,000 rows; same columns as applications.out" in prompt
    assert "test: 1,000 rows; same columns as applications.out" in prompt
    # A small statistics table inline, a fitted artifact cut short.
    assert '"iv_band": "suspicious"' in prompt and "summary: statistics table, 5 rows" in prompt
    assert "binning (binning):" in prompt and "get_output_summary for all" in prompt
    # Docs for the listed blocks, and for any block the goal names.
    for category in ("apply_binning", "logistic_regression", "compare_samples"):
        assert f"### {category}" in prompt
    assert "features: list[str] (required)" in prompt and "auto-fills from the target role" in prompt
    # Still no rows of the data itself.
    assert "West" not in prompt and "APP000001" not in prompt


def test_the_outline_names_real_blocks_per_stage(prepared):
    session, anchor = prepared
    b = AgentBuild(session, RunSlot(), "g", [anchor])
    b.phase = PLANNING
    r = b.call_tool("submit_plan", {"plan": {"summary": "x", "stages": [
        {"key": "est", "name": "Estimation", "goal": "fit", "blocks": ["logistic_regression", "made_up", "read_csv"]}]}})
    assert "['made_up', 'read_csv']" in r["error"]
    r = b.call_tool("submit_plan", {"plan": {"summary": "x", "stages": [
        {"key": "est", "name": "Estimation", "goal": "fit", "blocks": ["logistic_regression"]}]}})
    assert r["ok"] and b.plan["stages"][0]["blocks"] == ["logistic_regression"]


def test_a_step_report_carries_what_the_step_needs_checking(prepared):
    """run_to (and so each plan_stage step) reports a small statistics
    table whole, a dataframe's target rate and the columns the block
    added -- so checking a step needs no get_output_summary round trip."""
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    b = staged(session, anchor)
    built = b.call_tool("plan_stage", {"steps": est_steps()})
    assert built.get("ok"), built
    split = built["steps"][0]["outputs"]
    for sample in ("train", "test"):
        assert 0 < split[sample]["default_flag_mean"] < 1
        assert "new_columns" not in split[sample]  # a split adds no columns
    scored = b.call_tool("add_block", {"category": "predict", "lane": "est"})["block"]
    b.call_tool("connect", {"from_block": built["steps"][0]["block"], "from_port": "test", "to_block": scored, "to_port": "df"})
    b.call_tool("connect", {"from_block": built["steps"][1]["block"], "from_port": "model", "to_block": scored, "to_port": "model"})
    ran = b.call_tool("run_to", {"block": scored})
    new = ran["outputs"]["predictions"]["new_columns"]
    assert new and all(0 <= c["min"] <= c["max"] <= 1 for c in new if c.get("role") == "predicted")

    fit, ran = _binned(session, anchor, b)
    summary = ran["outputs"]["summary"]
    assert summary["type"] == "statistics_table" and {r["feature"] for r in summary["rows"]} == set(FEATURES) | {"region"}


def test_a_big_statistics_table_points_to_get_output_summary(prepared, monkeypatch):
    import modelmaker.agent.tools as T

    session, anchor = prepared
    monkeypatch.setattr(T, "REPORT_TABLE_CHARS", 300)
    b = building(session, anchor)
    fit, ran = _binned(session, anchor, b)
    assert "rows" not in ran["outputs"]["summary"] and "get_output_summary" in ran["outputs"]["summary"]["note"]


@pytest.mark.parametrize("small", [False, True])
def test_small_context_starts_each_stage_fresh_with_the_decisions(prepared, small):
    session, anchor = prepared
    ANCHOR[:] = [anchor]

    def asks(call):
        call("ask_user", {"question": "Which seed should the split use?"})
        return "Waiting."

    def builds_as_told(call):
        assert call("note_deviation", {"what": "Used seed 1", "why": "the user asked"})["ok"]
        return build_est(call)

    controller = make_controller(session, {"plan": [plan_turn], "build": [asks, builds_as_told, build_val]})
    b = start(controller, anchor, small_context=small)
    controller.join(60)
    controller.approve()
    controller.join(120)
    controller.feedback("Seed 1, and keep all five drivers.")
    controller.join(300)
    assert b.phase == AWAITING_STAGE_REVIEW
    controller.approve()
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)

    stage2 = controller.loops["build"].prompts[-1]
    assert "Stage 2 of 2" in stage2
    if small:
        # A fresh conversation: the whole outline again, plus what the
        # first stage decided -- its summary, the user's answer, the deviation.
        assert "The user approved this outline" in stage2
        assert "Which seed should the split use?" in stage2 and "Seed 1, and keep all five drivers." in stage2
        assert "Split 70/30 and fitted the PD model" in stage2 and "Used seed 1" in stage2
        assert any(e["kind"] == "fresh_conversation" for e in b.events_since())
    else:
        assert "The user approved this outline" not in stage2 and "fresh conversation" not in stage2
    assert b.log_record()["options"]["small_context"] is small
