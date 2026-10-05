"""Guided builds (agent/guided.py): the app asks one narrow question at a
time and the model answers with one tool call -- driven here by scripted
answers against the real runner and the PD sample data."""

from __future__ import annotations

from modelmaker.agent.build import AWAITING_APPROVAL, AWAITING_STAGE_REVIEW, DONE, BuildOptions, LLMChoice
from modelmaker.agent.guided import GuidedLoop, is_guided_provider
from modelmaker.agent.loop import Answer
from modelmaker.agent.tools import tools_for_phase

from .test_agent_build import FEATURES, make_controller, prepared  # noqa: F401 -- fixture

PLAN = {
    "summary": "Split, fit a logistic regression, measure Gini on the test set.",
    "stages": [
        {"key": "est", "name": "Estimation", "goal": "70/30 split; logistic regression on the development sample",
         "blocks": ["train_test_split", "logistic_regression"]},
        {"key": "val", "name": "Validation", "goal": "Score the test sample and report its Gini", "blocks": ["auc_gini"]},
    ],
}


def start(controller, anchor, goal="PD model, logistic regression", **options):
    return controller.start(goal, [anchor], LLMChoice("stub"), LLMChoice("stub"), BuildOptions(guided=True, **options), token="t")


def test_a_guided_build_asks_one_question_at_a_time(prepared):
    session, anchor = prepared
    seen: dict[str, str] = {}

    def plan(prompt):
        seen["plan"] = prompt
        return "submit_plan", PLAN

    def empty_done(prompt):
        seen["est1"] = prompt
        return "stage_done", {"summary": "nothing yet"}

    def split_without_settings(prompt):
        seen["est2"] = prompt
        # A label that isn't there, as the first try.
        return "place_block", {"block": "train_test_split", "inputs": ["D9"]}

    def split(prompt):
        seen["est3"] = prompt
        # "target": the role word, for the one target column.
        return "place_block", {"block": "train_test_split", "inputs": ["D1"], "settings": {"test_size": 0.3, "seed": 1, "stratify_col": "target"}}

    def fit(prompt):
        seen["est4"] = prompt
        # No features: the app fills them from the sample's profile.
        return "place_block", {"block": "logistic_regression", "inputs": ["D2"], "settings": {"features": "the good ones"}}

    def est_done(prompt):
        return "stage_done", {"summary": "Split 70/30 and fitted the PD model on 5 drivers."}

    def no_tool(prompt):
        return Answer(text="I think we should score the test sample.")

    def score(prompt):
        seen["val2"] = prompt
        # predict isn't in the stage's blocks: allowed, and recorded.
        return "place_block", {"block": "predict", "inputs": ["D3", "M1"], "why": "Gini needs scores"}

    def gini(prompt):
        return "place_block", {"block": "auc_gini", "inputs": ["D5"]}

    def val_done(prompt):
        return "stage_done", {"summary": "Test Gini measured."}

    controller = make_controller(session, {"plan": [plan], "build": [empty_done, split_without_settings, split, fit,
                                                                      est_done, no_tool, score, gini, val_done]})
    b = start(controller, anchor)
    controller.join(60)
    assert b.phase == AWAITING_APPROVAL and b.plan["stages"][0]["key"] == "est"
    # One prompt: the goal, the profiled data (text dates read as dates), the catalogue, the rules.
    assert "Goal: PD model, logistic regression" in seen["plan"]
    assert "default_flag (binary, 2 levels, rate of 1 = " in seen["plan"] and "application_date 20" in seen["plan"]
    assert "- train_test_split: " in seen["plan"] and "Split the sample before" in seen["plan"]

    controller.approve()
    controller.join(300)
    assert b.phase == AWAITING_STAGE_REVIEW, (b.phase, b.error, b.pending_question)
    # The app parsed the dates before the first question; the anchor gave way to it.
    parsed = [blk for blk in session.graph.blocks.values() if blk.category == "parse_dates"]
    assert len(parsed) == 1 and "D1 = out of \"applications (dates parsed)\"" in seen["est1"]
    assert "reference_date 20" in seen["est1"] and "75% of rows before" in seen["est1"]
    assert "best practice:" in seen["est1"]  # the stage's blocks' docs carry it
    # Refusals and failures come back as the next question's "last step".
    assert "not yet placed: train_test_split, logistic_regression" in seen["est2"]
    assert "'D9' isn't one of the labels above" in seen["est3"]
    assert "placed train_test_split on D1 -- it ran OK" in seen["est4"]
    assert "D2 = train of" in seen["est4"] and "D3 = test of" in seen["est4"]

    controller.approve()
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)
    assert "you answered without a tool call" in seen["val2"] and "M1 = model of" in seen["val2"]
    by_cat = {blk.category: blk for blk in session.graph.blocks.values()}
    wires = {(w.from_block, w.from_port, w.to_block, w.to_port) for w in session.graph.wires.values()}
    split_id, fit_id, pred_id = by_cat["train_test_split"].id, by_cat["logistic_regression"].id, by_cat["predict"].id
    assert (split_id, "test", pred_id, "df") in wires and (fit_id, "model", pred_id, "model") in wires
    assert session.runner.status(by_cat["auc_gini"].id) == "green"
    assert by_cat["train_test_split"].params["stratify_col"] == "default_flag"
    features = by_cat["logistic_regression"].params["features"]
    assert "credit_score" in features and "default_flag" not in features and "application_id" not in features
    assert "credit_limit" not in features  # has missing values
    assert any("used predict, not in stage val's plan" in d["what"] for d in b.deviations)


def test_a_block_that_fails_to_run_is_removed_and_explained(prepared):
    session, anchor = prepared
    seen = {}

    def bad_split(prompt):
        # Not a date: time_split fails when it runs.
        return "place_block", {"block": "time_split", "inputs": ["D1"], "settings": {"date_col": "reference_date", "cutoff": "soon"}}

    def look(prompt):
        seen["after"] = prompt
        return Answer(text="")

    plan = {"summary": "s", "stages": [{"key": "oot", "name": "Split", "goal": "out-of-time split", "blocks": ["time_split"]}]}
    controller = make_controller(session, {"plan": [lambda p: ("submit_plan", plan)], "build": [bad_split, look]})
    b = start(controller, anchor, goal="Split the data")
    controller.join(60)
    controller.approve()
    controller.join(300)
    assert "time_split failed to run and was removed" in seen["after"]
    assert "### time_split" in seen["after"]  # its docs, to get it right
    assert not [blk for blk in session.graph.blocks.values() if blk.category == "time_split"]


def test_guided_is_the_default_for_local_models_only():
    assert is_guided_provider("lmstudio") and not is_guided_provider("anthropic") and not is_guided_provider("openai")
    # stage_done is the guided builds' own: never offered to a normal build.
    assert "stage_done" not in {t.name for t in tools_for_phase("building")}


def test_guided_wraps_only_a_loop_that_can_ask(prepared):
    session, anchor = prepared
    controller = make_controller(session, {"plan": [lambda p: ("submit_plan", PLAN)]})
    start(controller, anchor)
    controller.join(60)
    assert isinstance(controller._plan_loop, GuidedLoop)


def test_the_observation_date_role_fills_date_col(prepared):
    session, anchor = prepared
    with session.edit():
        session.set_column_role(anchor, "reference_date", "date")
    assert session.runner.run_block(anchor) == "green"
    seen = {}

    def split(prompt):
        seen["first"] = prompt
        # Not a column: dropped, so date_col fills from the date role.
        return "place_block", {"block": "time_split", "inputs": ["D1"], "settings": {"date_col": "observation_date", "cutoff": "2024-10-08"}}

    def done(prompt):
        seen["after"] = prompt
        return "stage_done", {"summary": "split"}

    plan = {"summary": "s", "stages": [{"key": "oot", "name": "Split", "goal": "out-of-time split", "blocks": ["time_split"]}]}
    controller = make_controller(session, {"plan": [lambda p: ("submit_plan", plan)], "build": [split, done]})
    start(controller, anchor, goal="Out-of-time split of the data")
    controller.join(60)
    controller.approve()
    controller.join(300)
    assert "observation date (date_col params fill from it): reference_date 20" in seen["first"]
    assert "placed time_split on D1 -- it ran OK" in seen["after"] and "left out: date_col=observation_date" in seen["after"]


def test_an_early_stage_done_is_told_once_what_isnt_placed_and_bad_features_are_left_out(prepared):
    session, anchor = prepared
    seen = {}

    def split(prompt):
        return "place_block", {"block": "train_test_split", "inputs": ["D1"], "settings": {"seed": 1}}

    def early(prompt):
        return "stage_done", {"summary": "done"}

    def fit(prompt):
        seen["warned"] = prompt
        # credit_limit has missing values: logistic_regression needs complete numeric features.
        return "place_block", {"block": "logistic_regression", "inputs": ["D2"], "settings": {"features": FEATURES + ["credit_limit"]}}

    def done(prompt):
        seen["fitted"] = prompt
        return "stage_done", {"summary": "fitted"}

    plan = {"summary": "s", "stages": [{"key": "est", "name": "Estimation", "goal": "split and fit",
                                        "blocks": ["train_test_split", "logistic_regression"]}]}
    controller = make_controller(session, {"plan": [lambda p: ("submit_plan", plan)], "build": [split, early, fit, done]})
    b = start(controller, anchor)
    controller.join(60)
    controller.approve()
    controller.join(300)
    assert b.phase == DONE, (b.phase, b.error, b.pending_question)
    assert "not yet placed: logistic_regression" in seen["warned"]
    assert "credit_limit (missing values or not numeric)" in seen["fitted"]
    fit_block = next(blk for blk in session.graph.blocks.values() if blk.category == "logistic_regression")
    assert fit_block.params["features"] == FEATURES


def test_a_plan_that_stops_short_of_the_goal_is_sent_back(prepared):
    session, anchor = prepared
    short = {"summary": "s", "stages": [{"key": "prep", "name": "Prep", "goal": "prepare", "blocks": ["select"]}]}
    seen = {}

    def again(prompt):
        seen["retry"] = prompt
        return "submit_plan", PLAN

    controller = make_controller(session, {"plan": [lambda p: ("submit_plan", short), again]})
    b = start(controller, anchor, goal="PD model with validation (Gini)")
    controller.join(60)
    assert "plan needs a stage that fits the model" in seen["retry"] and "plan needs a stage that validates it" not in seen["retry"].split("It was rejected")[0]
    assert b.plan["stages"][0]["key"] == "est"
