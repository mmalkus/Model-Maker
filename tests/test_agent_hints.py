"""Decision hints (BuildOptions.decision_hints, see modelmaker/agent/hints.py):
the rules on their own, and their place in run_to, plan_stage and the
prompts -- only when the option is on."""

from __future__ import annotations

import polars as pl

from modelmaker.agent import prompts
from modelmaker.agent.hints import decision_hints
from modelmaker.packet import DataFramePacket

from .test_agent_build import ANCHOR, FEATURES, _binned, est_steps, prepared, staged  # noqa: F401 -- fixture


def _table(rows: list[dict]) -> DataFramePacket:
    return DataFramePacket(pl.DataFrame(rows), {})


# ---- the rules -------------------------------------------------------------------


def test_binning_sorts_features_by_iv_band():
    summary = _table(
        [
            {"feature": "leaky", "iv": 0.9, "monotonic": True},
            {"feature": "good", "iv": 0.25, "monotonic": True},
            {"feature": "bumpy", "iv": 0.12, "monotonic": False},
            {"feature": "noise", "iv": 0.005, "monotonic": True},
        ]
    )
    hints = decision_hints("fit_binning", {"summary": summary})
    assert any("Candidate" in h and "good" in h and "bumpy" in h and "leaky" not in h for h in hints)
    assert any("Leave out" in h and "noise" in h for h in hints)
    assert any("leaky" in h and "ask_user" in h for h in hints)
    assert any("Not monotonic" in h and "bumpy" in h and "good" not in h for h in hints)


def test_binning_with_a_continuous_target_uses_r2():
    summary = _table([{"feature": "a", "r2_binned": 0.08}, {"feature": "b", "r2_binned": 0.001}])
    hints = decision_hints("fit_binning", {"summary": summary})
    assert any("Candidate" in h and "a" in h for h in hints)
    assert any("Leave out" in h and "b" in h for h in hints)


def test_coefficients_flag_wrong_signs_and_insignificance():
    model = {
        "coefficients": {"x_woe": 0.4, "y_woe": -0.8, "z_woe": -0.1},
        "statistics": {
            "intercept": {"p_value": 0.9},
            "x_woe": {"p_value": 0.001},
            "y_woe": {"p_value": 0.001},
            "z_woe": {"p_value": 0.4},
        },
    }
    hints = decision_hints("logistic_regression", {"model": model})
    assert any("wrong sign" in h and "x_woe" in h and "y_woe" not in h for h in hints)
    assert any("Not significant" in h and "z_woe" in h and "intercept" not in h for h in hints)

    sound = {"coefficients": {"y_woe": -0.8}, "statistics": {"y_woe": {"p_value": 0.001}}}
    assert decision_hints("logistic_regression", {"model": sound}) == [
        "Coefficients look sound: signs as expected and all significant."
    ]


def test_gini_bands():
    def hint(g):
        return decision_hints("auc_gini", {"metric": {"auc": (g + 1) / 2, "gini": g}})[0]

    assert "wrong way" in hint(-0.1)
    assert "weak" in hint(0.1)
    assert "usual range" in hint(0.5)
    assert "leakage" in hint(0.95)


def test_compare_samples_flags_overfitting_and_calibration():
    table = _table(
        [
            {"sample": "train", "gini": 0.62, "mean_actual": 0.05, "mean_predicted": 0.05},
            {"sample": "test", "gini": 0.41, "mean_actual": 0.05, "mean_predicted": 0.08},
        ]
    )
    hints = decision_hints("compare_samples", {"table": table})
    assert any("overfitting" in h and "train" in h and "test" in h for h in hints)
    assert any("calibrate" in h and "test" in h and "above" in h for h in hints)

    steady = _table(
        [
            {"sample": "train", "gini": 0.6, "mean_actual": 0.05, "mean_predicted": 0.05},
            {"sample": "test", "gini": 0.58, "mean_actual": 0.05, "mean_predicted": 0.051},
        ]
    )
    assert "holds" in decision_hints("compare_samples", {"table": steady})[0]


def test_psi_and_rating_summary():
    assert "stable" in decision_hints("psi_test", {"metric": {"psi": 0.05}})[0]
    assert "moderate" in decision_hints("psi_test", {"metric": {"psi": 0.15}})[0]
    assert "ask_user" in decision_hints("psi_test", {"metric": {"psi": 0.4}})[0]
    assert "refit" in decision_hints("rating_summary", {"metric": {"monotonic": False}})[0]


def test_blocks_without_rules_and_odd_outputs_get_no_hints():
    assert decision_hints("train_test_split", {"train": _table([{"a": 1}])}) == []
    assert decision_hints("auc_gini", {"metric": "not a dict"}) == []
    assert decision_hints("fit_binning", {}) == []


# ---- in the build ------------------------------------------------------------------


def test_run_to_carries_hints_only_when_on(prepared):  # noqa: F811
    session, anchor = prepared
    b = staged(session, anchor)
    _, ran = _binned(session, anchor, b)
    assert "next" not in ran

    b.options.decision_hints = True
    fit = b.call_tool("add_block", {"category": "fit_binning", "lane": "est", "params": {"features": FEATURES}})["block"]
    b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": fit, "to_port": "df"})
    ran = b.call_tool("run_to", {"block": fit})
    assert ran["status"] == "green", ran
    # credit_score is banded 'suspicious' in the sample data (see
    # test_a_suspicious_feature_in_a_model_is_flagged_by_the_app).
    assert any("credit_score" in h and "ask_user" in h for h in ran["next"])


def test_plan_stage_steps_and_the_next_stage_prompt_carry_hints(prepared):  # noqa: F811
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    b = staged(session, anchor)
    b.options.decision_hints = True
    built = b.call_tool("plan_stage", {"steps": est_steps()})
    assert built.get("ok"), built
    split, fit = built["steps"]
    assert "next" not in split  # no rule for a split
    assert fit["next"] and fit["next"] == ["All coefficients are significant."]
    assert "  next: " in prompts.built_so_far(b)
    assert "## Hints" in prompts.build_system(b)

    b.options.decision_hints = False
    assert "  next: " not in prompts.built_so_far(b)
    assert "## Hints" not in prompts.build_system(b)
