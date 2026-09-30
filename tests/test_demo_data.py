"""The demo dataset and how it reaches a pip-installed user: the generator
must be deterministic, the data must actually support
PD, LGD and CCF work with the library's own blocks, and a wheel must carry
the demo projects plus the generator (not the CSV)."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest

from modelmaker import demo_data
from modelmaker.blocks.modelling import compute_ccf, compute_lgd, lgd_regression
from modelmaker.session import ProjectSession

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = REPO_ROOT / "sample_data" / demo_data.FILENAME  # written by conftest.py

DEFAULT_ONLY = ["default_date", "balance_at_default", "recovery_amount", "workout_cost", "workout_months", "cure_flag"]


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.read_csv(DEMO_CSV)


def test_generator_is_deterministic(tmp_path):
    a = demo_data.write_csv(tmp_path / "a.csv").read_bytes()
    b = demo_data.write_csv(tmp_path / "b.csv").read_bytes()
    assert a == b
    assert demo_data.write_csv(tmp_path / "c.csv", seed=demo_data.SEED + 1).read_bytes() != a


def test_default_only_fields_are_filled_exactly_on_defaults(df):
    defaulted = df["default_flag"] == 1
    for col in DEFAULT_ONLY:
        assert (df[col].is_null() == ~defaulted).all(), col
    # Only term loans lack a limit; only cards lack a term.
    assert (df["credit_limit"].is_null() == (df["product_type"] == "term_loan")).all()
    assert (df["loan_term_months"].is_null() == (df["product_type"] == "credit_card")).all()


def test_pd_has_a_plausible_default_rate(df):
    assert 0.05 < df["default_flag"].mean() < 0.15


def test_lgd_is_computable_on_every_default_and_bimodal(df):
    defaults = df.filter(pl.col("default_flag") == 1)
    assert defaults.height >= 400
    lgd = compute_lgd(defaults, "balance_at_default", "recovery_amount", "workout_cost")["lgd"]
    assert lgd.null_count() == 0
    assert (lgd <= 0.1).mean() > 0.1 and (lgd >= 0.9).mean() > 0.1

    secured = compute_lgd(defaults.filter(pl.col("is_secured") == 1), "balance_at_default", "recovery_amount", "workout_cost")
    unsecured = compute_lgd(defaults.filter(pl.col("is_secured") == 0), "balance_at_default", "recovery_amount", "workout_cost")
    assert secured["lgd"].mean() < unsecured["lgd"].mean()

    _, model = lgd_regression(
        compute_lgd(defaults, "balance_at_default", "recovery_amount", "workout_cost"),
        "lgd",
        ["is_secured", "credit_score"],
    )
    assert model["converged"]
    assert model["coefficients"]["is_secured"] < 0


def test_ccf_is_computable_on_defaulted_cards(df):
    cards = df.filter((pl.col("default_flag") == 1) & (pl.col("product_type") == "credit_card"))
    assert cards.height >= 150
    raw = compute_ccf(cards, "credit_limit", "balance_at_reference", "balance_at_default", floor=None, cap=None)["ccf"]
    # A few observations outside [0, 1], so the floor/cap have something to do.
    assert (raw < 0).any() and (raw > 1).any()
    ccf = compute_ccf(cards, "credit_limit", "balance_at_reference", "balance_at_default")
    assert 0.4 < ccf["ccf"].mean() < 0.9

    ccf = ccf.with_columns((pl.col("balance_at_reference") / pl.col("credit_limit")).alias("utilisation"))
    _, model = lgd_regression(ccf, "ccf", ["utilisation"])
    assert model["converged"]
    assert model["coefficients"]["utilisation"] < 0


def test_workspace_runs_the_demo_green_and_keeps_user_edits(tmp_path, monkeypatch):
    ws = demo_data.create_workspace(tmp_path / "demo")
    assert (ws / "sample_data" / demo_data.FILENAME).is_file()
    project = ws / "projects" / "demo_pd_model" / "model.json"

    monkeypatch.chdir(ws)
    session = ProjectSession(recovery_path=None)
    session.load(project)
    session.runner.run_all()
    assert all(session.runner.status(bid) == "green" for bid in session.graph.blocks)

    project.write_text("edited", encoding="utf-8")
    demo_data.create_workspace(ws)
    assert project.read_text(encoding="utf-8") == "edited"


def test_modelmaker_demo_cli(tmp_path, capsys):
    demo_data.workspace_main([str(tmp_path / "ws")])
    assert "Demo ready" in capsys.readouterr().out
    assert (tmp_path / "ws" / "projects" / "demo_pd_model" / "model.json").is_file()


def test_wheel_ships_demo_projects_and_generator_but_not_the_csv(tmp_path):
    pytest.importorskip("setuptools", reason="building a wheel offline needs setuptools installed")
    env = {**os.environ, "MODELMAKER_SKIP_FRONTEND_BUILD": "1"}
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(REPO_ROOT), "--no-deps", "--no-build-isolation", "-w", str(tmp_path), "-q"],
        check=True,
        env=env,
    )
    (wheel,) = tmp_path.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert "modelmaker/demo_data.py" in names
    assert "modelmaker/examples/demo_pd_model/model.json" in names
    assert not [n for n in names if n.endswith(".csv")]
