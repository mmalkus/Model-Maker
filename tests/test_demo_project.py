"""The demo project is the first thing a new user opens, so it is checked
the only way that means anything: by loading the shipped file and running
every block in it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelmaker.session import ProjectSession

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "projects" / "demo_pd_model.json"


@pytest.fixture
def in_repo_root(monkeypatch):
    # The demo references its data as a repo-relative path, the way it will
    # resolve when the server is started from the project root.
    monkeypatch.chdir(REPO_ROOT)
    yield
    os.chdir(REPO_ROOT)


def test_demo_project_runs_green_end_to_end(in_repo_root):
    session = ProjectSession(recovery_path=None)
    session.load(DEMO)
    assert session.graph.blocks, "demo project has no blocks"

    # Just Run all, the way the README tells a new user to do it -- its
    # sources have never been read, so run_all() reads them itself (see
    # Runner._sweep) instead of leaving the whole pipeline blocked.
    session.runner.run_all()

    failures = {
        block.name: session.runner.state[bid].last_error
        for bid, block in session.graph.blocks.items()
        if session.runner.status(bid) != "green"
    }
    assert not failures, f"demo pipeline is not green: {failures}"


def test_demo_project_produces_the_headline_validation_metrics(in_repo_root):
    session = ProjectSession(recovery_path=None)
    session.load(DEMO)
    session.runner.run_all()

    metrics = {}
    for bid, block in session.graph.blocks.items():
        if block.category in ("auc_gini", "ks_test", "psi_test"):
            entry = session.runner.cache.get(session.runner.state[bid].last_successful_key)
            metrics[block.category] = entry.outputs["metric"]

    assert set(metrics) == {"auc_gini", "ks_test", "psi_test"}
    # Loose bounds: this guards against a demo that silently degrades into
    # a coin-flip model, not against small changes in the numbers.
    assert 0.5 < metrics["auc_gini"]["auc"] <= 1.0
    assert 0.0 < metrics["ks_test"]["ks_statistic"] <= 1.0
    assert metrics["psi_test"]["psi"] >= 0.0
