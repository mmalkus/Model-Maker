"""Decision-based planning (modelmaker/agent/decisions): the phase menus,
the per-block questions, the context budget and the decision log -- all with
the rule policy or a scripted backend, no model calls."""

from __future__ import annotations

import polars as pl
import pytest

from modelmaker.agent.decisions import Decider, Workspace, plan
from modelmaker.agent.decisions.backends import RuleBackend, _answer
from modelmaker.agent.decisions.bench import scenarios
from modelmaker.agent.decisions.core import (
    Answer,
    ContextBudgetError,
    choice,
    fit,
    noul,
    score,
)
from modelmaker.agent.decisions.phases import DONE, screen_rule
from modelmaker.agent.decisions.workspace import column_facts


@pytest.fixture(scope="module")
def scen():
    return scenarios()


def _run(scen, name, backend=None):
    df, roles = scen[name]
    ws = Workspace(df, dict(roles))
    decider = Decider(backend or RuleBackend())
    return ws, decider, plan(ws, decider)


def _path(result):
    return [s["option"] for s in result.steps]


# ---- the menus vary with the data --------------------------------------------------------


@pytest.mark.parametrize(
    "name, path",
    [
        (
            "snapshot",
            [
                "profile",
                "screen",
                "dq_rules",
                "missing",
                "trend",
                "drop_immature",
                "time_split",
                "train_test",
            ],
        ),
        ("no_dates", ["profile", "screen", "dq_rules", "missing", "train_test"]),
        (
            "few_defaults",
            ["profile", "screen", "dq_rules", "missing", "trend", "train_test"],
        ),
        (
            "messy",
            [
                "profile",
                "screen",
                "dq_rules",
                "exclusions",
                "missing",
                "trend",
                "drop_immature",
                "time_split",
                "train_test",
            ],
        ),
        # ~4% negative balances in the panel: bad values, excluded.
        (
            "panel",
            [
                "profile",
                "screen",
                "dq_rules",
                "exclusions",
                "forward_flag",
                "trend",
                "drop_immature",
                "time_split",
                "train_test",
            ],
        ),
    ],
)
def test_rule_policy_paths(scen, name, path):
    _, decider, result = _run(scen, name)
    assert result.model_type == "pd"
    assert _path(result) == path
    assert not any(s.get("error") for s in result.steps)
    assert all(r.agrees and not r.needs_user for r in decider.records)


def test_every_decision_fits_layas_context(scen):
    for name in scen:
        _, decider, _ = _run(scen, name)
        assert max(r.tokens for r in decider.records) <= 512


def test_screen_excludes_ids_dates_and_post_outcome_columns(scen):
    _, _, result = _run(scen, "snapshot")
    excluded = next(s for s in result.steps if s["option"] == "screen")["params"]["excluded"]
    assert excluded["reference_date"] == "exclude_date"
    for col in ("default_date", "recovery_amount", "cure_flag"):
        assert excluded[col] == "exclude_post_outcome"
    assert "annual_income" not in excluded


def test_panel_builds_the_target_from_the_status_that_never_reverts(scen):
    ws, _, result = _run(scen, "panel")
    step = next(s for s in result.steps if s["option"] == "forward_flag")
    assert step["params"]["default_col"] == "in_default"  # not has_overdraft
    assert ws.col("target") == "default_12m"
    assert ws.roles["in_default"] == "excluded"


def test_messy_data_gets_exclusions_with_a_waterfall(scen):
    ws, _, result = _run(scen, "messy")
    step = next(s for s in result.steps if s["option"] == "exclusions")
    assert {r["name"] for r in step["params"]["rules"]} == {
        "target missing",
        "dti out of range",
    }
    assert ws.df["default_flag"].null_count() == 0


def test_guardrail_no_min_max_for_ids_or_text():
    df = pl.DataFrame(
        {
            "acct": list(range(100)),
            "name": [f"n{i}" for i in range(100)],
            "x": [1.5] * 100,
        }
    )
    ws = Workspace(df, {})
    assert "min" not in column_facts(ws, "acct")
    assert "min" not in column_facts(ws, "name")
    assert column_facts(ws, "x")["min"] == 1.5


def test_screen_rule_order():
    assert (
        screen_rule(
            {
                "dtype": "Float64",
                "fill_rate": 0.1,
                "distinct_share": 0.9,
                "fill_events": 1.0,
                "fill_non_events": 0.0,
            }
        )
        == "exclude_post_outcome"
    )
    assert screen_rule({"dtype": "String", "fill_rate": 1, "distinct_share": 1.0}) == "exclude_id"
    assert screen_rule({"dtype": "Float64", "fill_rate": 1, "distinct_share": 1.0}) == "use"


# ---- a model backend: low confidence, DONE, slips -------------------------------------------


class Scripted:
    """Answers every question with its first label, except the menu, where
    it says DONE when that's on offer and otherwise picks the last option
    (out of the usual order); `conf` sets how sure it is."""

    name = "scripted"

    def __init__(self, conf=0.9, menu=DONE):
        self.conf, self.menu = conf, menu

    def decide(self, items):
        out = []
        for _, questions in items:
            answers = {}
            for qid, q in questions.items():
                if qid == "next":
                    answers[qid] = Answer(
                        self.menu if self.menu in q["criteria"] else list(q["criteria"])[-1], self.conf
                    )
                elif q["type"] == "noul":
                    answers[qid] = Answer(True, self.conf)
                else:
                    first = next(iter(q["criteria"])) if isinstance(q["criteria"], dict) else q["criteria"][0]
                    answers[qid] = Answer(first, self.conf)
            out.append(answers)
        return out


def test_done_only_once_nothing_required_is_left(scen):
    # Picks DONE whenever it's offered, else the last option (out of order).
    _, decider, result = _run(scen, "snapshot", Scripted(menu=DONE))
    assert _path(result) == ["profile", "screen", "missing", "dq_rules", DONE, "train_test", "trend", DONE]
    # Phase 2 ended with the optional immature-period check skipped, and the
    # out-of-time split was gone once train/test had run.
    assert result.steps[-1]["skipped"] == ["drop_immature"]
    for r in decider.records:
        if r.qid == "next" and DONE in r.question["criteria"]:
            assert all("(optional)" in v for k, v in r.question["criteria"].items() if k != DONE)


def test_low_confidence_goes_to_the_person(scen):
    _, decider, _ = _run(scen, "no_dates", Scripted(conf=0.5, menu="__none__"))
    assert decider.records and all(r.needs_user for r in decider.records)


# ---- answers and the budget ------------------------------------------------------------------


def test_answer_normalises_probabilities():
    a = _answer(choice("q", {"a": "", "b": ""}), {"a": 2, "b": 6})
    assert a.value == "b" and a.confidence == pytest.approx(0.75)
    assert _answer(noul("q"), {"true": 0.2}).value is False
    assert _answer(noul("q"), {"true": 0.2}).confidence == pytest.approx(0.8)
    # Score: the expected level (0.2 + 1.4 = 1.6), reported as the nearest level.
    assert _answer(score("q", ["low", "mid", "high"]), {"low": 0.1, "mid": 0.2, "high": 0.7}).value == "high"


def test_fit_drops_low_priority_keys_then_refuses():
    q = {"q": noul("Is it?")}
    state = {"keep": 1, "bulky": "x" * 2000}
    fitted, tokens = fit(state, q, 512, ["bulky"])
    assert "bulky" not in fitted and tokens <= 512
    with pytest.raises(ContextBudgetError):
        fit(state, q, 512, [])
