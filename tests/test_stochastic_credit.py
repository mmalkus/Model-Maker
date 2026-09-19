"""Tests for the closed-form single-factor ASRF credit EC block (see
stochastic-engine-proposal.md S3.2) -- modelmaker.stochastic.credit and
its BlockSpec wrapper in blocks/stochastic.py.

No hand-derived reference numbers here (nothing to check them against
independently) -- instead, the formula's own well-known mathematical
properties, each individually verifiable from the Vasicek/ASRF definition
itself: zero correlation collapses to zero capital, capital is never
negative, and capital is monotone in both confidence and correlation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from modelmaker.blocks import stochastic as blocks
from modelmaker.stochastic import credit


def test_basel_corporate_correlation_asymptotes():
    assert credit.basel_corporate_correlation(np.array([1e-6]))[0] == pytest.approx(0.24, abs=1e-4)
    assert credit.basel_corporate_correlation(np.array([0.999999]))[0] == pytest.approx(0.12, abs=1e-4)


def test_capital_rate_is_zero_when_correlation_is_zero():
    pd = np.array([0.01, 0.05, 0.2])
    rate = credit.asrf_capital_rate(pd, correlation=np.zeros_like(pd), confidence=0.999)
    assert rate == pytest.approx(np.zeros_like(pd), abs=1e-10)


def test_capital_rate_is_never_negative():
    rng = np.random.default_rng(0)
    pd = rng.uniform(0.001, 0.3, size=200)
    corr = rng.uniform(0.01, 0.5, size=200)
    rate = credit.asrf_capital_rate(pd, corr, confidence=0.999)
    assert (rate >= -1e-12).all()


def test_capital_rate_increases_with_confidence():
    pd = np.array([0.02])
    corr = np.array([0.2])
    low = credit.asrf_capital_rate(pd, corr, confidence=0.95)[0]
    high = credit.asrf_capital_rate(pd, corr, confidence=0.999)[0]
    assert high > low


def test_capital_rate_increases_with_correlation():
    pd = np.array([0.02])
    low = credit.asrf_capital_rate(pd, np.array([0.05]), confidence=0.999)[0]
    high = credit.asrf_capital_rate(pd, np.array([0.5]), confidence=0.999)[0]
    assert high > low


def test_portfolio_summary_sums_obligor_level_values():
    pd = np.array([0.01, 0.02, 0.03])
    lgd = np.array([0.4, 0.45, 0.5])
    ead = np.array([100.0, 200.0, 300.0])
    corr = credit.basel_corporate_correlation(pd)
    summary, ec, el = credit.asrf_economic_capital(pd, lgd, ead, corr, confidence=0.999)
    assert summary["expected_loss"] == pytest.approx(float(el.sum()))
    assert summary["economic_capital"] == pytest.approx(float(ec.sum()))
    assert summary["capital_requirement"] == pytest.approx(summary["expected_loss"] + summary["economic_capital"])
    assert summary["total_ead"] == pytest.approx(600.0)
    assert summary["n_obligors"] == 3


def test_asrf_economic_capital_block_end_to_end():
    df = pl.DataFrame(
        {
            "obligor": ["a", "b", "c"],
            "pd": [0.01, 0.02, 0.05],
            "lgd": [0.4, 0.45, 0.5],
            "ead": [1000.0, 2000.0, 500.0],
        }
    )
    summary, detail = blocks.asrf_economic_capital(df, pd_col="pd", lgd_col="lgd", ead_col="ead", confidence=0.999)
    assert summary["kind"] == "simulation_result"
    assert summary["n_obligors"] == 3
    assert detail.height == 3
    assert set(detail.columns) >= {"obligor", "pd", "lgd", "ead", "correlation", "expected_loss", "economic_capital"}
    assert detail["economic_capital"].sum() == pytest.approx(summary["economic_capital"])
    # a fixed correlation should still run and differ from the Basel default
    summary_fixed, detail_fixed = blocks.asrf_economic_capital(df, pd_col="pd", lgd_col="lgd", ead_col="ead", correlation=0.15)
    assert detail_fixed["correlation"].to_list() == [0.15, 0.15, 0.15]
    assert summary_fixed["economic_capital"] != summary["economic_capital"]
