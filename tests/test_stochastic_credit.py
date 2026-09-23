"""Tests for the closed-form single-factor ASRF credit EC block and the
multi-factor Monte Carlo credit portfolio simulation (see
stochastic-engine-proposal.md S3.2) -- modelmaker.stochastic.credit and
their BlockSpec wrappers in blocks/stochastic.py.

No hand-derived reference numbers for the closed-form block (nothing to
check them against independently) -- instead, the formula's own
well-known mathematical properties, each individually verifiable from the
Vasicek/ASRF definition itself: zero correlation collapses to zero
capital, capital is never negative, and capital is monotone in both
confidence and correlation.

The Monte Carlo simulation *does* get a real numerical reference: with a
single sector and matching correlation assumptions, its VaR_alpha of
portfolio loss converges to the closed-form block's own
`capital_requirement` as n_paths grows -- not a coincidence, a direct
consequence of both being the same Merton conditional-independence model,
one integrated in closed form and the other by Monte Carlo over the same
single factor. This is *why* the closed-form block was built first (see
the proposal's own S10 sequencing): it's the simulation's benchmark."""

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


# ---------------------------------------------------------------------------
# Multi-factor Monte Carlo credit portfolio simulation
# ---------------------------------------------------------------------------


def _portfolio_df(n_obligors: int, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "obligor": [f"o{i}" for i in range(n_obligors)],
            "pd": rng.uniform(0.005, 0.05, size=n_obligors),
            "lgd": rng.uniform(0.3, 0.6, size=n_obligors),
            "ead": rng.uniform(50.0, 500.0, size=n_obligors),
            "sector": ["only_sector"] * n_obligors,
        }
    )


def test_multi_factor_simulation_converges_to_the_asrf_closed_form():
    df = _portfolio_df(20)
    dependency = {"kind": "dependency", "type": "gaussian", "labels": ["only_sector"], "corr": [[1.0]]}

    asrf_summary, _ = blocks.asrf_economic_capital(df, pd_col="pd", lgd_col="lgd", ead_col="ead", confidence=0.999)

    mc_result, mc_table, _ = blocks.simulate_credit_portfolio(
        df,
        dependency,
        pd_col="pd",
        lgd_col="lgd",
        ead_col="ead",
        sector_col="sector",
        n_paths=500_000,
        chunk_size=100_000,
        alpha_levels=[0.999],
        compute_contributions=False,
        seed=42,
    )

    assert mc_result["quantiles"][0.999] == pytest.approx(asrf_summary["capital_requirement"], rel=0.05)
    assert mc_table["var"][0] == pytest.approx(asrf_summary["capital_requirement"], rel=0.05)


def test_multi_factor_simulation_is_deterministic_given_seed():
    df = _portfolio_df(10)
    dependency = {"kind": "dependency", "type": "gaussian", "labels": ["only_sector"], "corr": [[1.0]]}
    kwargs = dict(pd_col="pd", lgd_col="lgd", ead_col="ead", sector_col="sector", n_paths=10_000, chunk_size=3_000, seed=7)
    result_a, _, _ = blocks.simulate_credit_portfolio(df, dependency, block_id="b1", **kwargs)
    result_b, _, _ = blocks.simulate_credit_portfolio(df, dependency, block_id="b1", **kwargs)
    result_c, _, _ = blocks.simulate_credit_portfolio(df, dependency, block_id="b2", **kwargs)
    assert result_a["quantiles"] == result_b["quantiles"]
    assert result_a["quantiles"] != result_c["quantiles"]


def test_multi_factor_simulation_chunking_does_not_bias_the_estimate():
    df = _portfolio_df(15, seed=3)
    dependency = {"kind": "dependency", "type": "gaussian", "labels": ["only_sector"], "corr": [[1.0]]}
    kwargs = dict(
        df=df,
        dependency=dependency,
        pd_col="pd",
        lgd_col="lgd",
        ead_col="ead",
        sector_col="sector",
        n_paths=200_000,
        alpha_levels=[0.999],
        seed=5,
        compute_contributions=False,
    )
    result_one_chunk, _, _ = blocks.simulate_credit_portfolio(chunk_size=200_000, block_id="b_a", **kwargs)
    result_many_chunks, _, _ = blocks.simulate_credit_portfolio(chunk_size=17_000, block_id="b_b", **kwargs)
    assert result_one_chunk["mean"] == pytest.approx(result_many_chunks["mean"], rel=0.03)
    assert result_one_chunk["quantiles"][0.999] == pytest.approx(result_many_chunks["quantiles"][0.999], rel=0.1)


def test_multi_factor_simulation_sector_correlation_increases_the_tail():
    df = pl.DataFrame(
        {
            "obligor": [f"o{i}" for i in range(30)],
            "pd": np.full(30, 0.02),
            "lgd": np.full(30, 0.45),
            "ead": np.full(30, 100.0),
            "sector": ["a"] * 15 + ["b"] * 15,
        }
    )
    kwargs = dict(
        df=df,
        pd_col="pd",
        lgd_col="lgd",
        ead_col="ead",
        sector_col="sector",
        n_paths=200_000,
        chunk_size=50_000,
        alpha_levels=[0.999],
        compute_contributions=False,
        seed=9,
    )
    independent = {"kind": "dependency", "type": "gaussian", "labels": ["a", "b"], "corr": [[1.0, 0.0], [0.0, 1.0]]}
    correlated = {"kind": "dependency", "type": "gaussian", "labels": ["a", "b"], "corr": [[1.0, 0.9], [0.9, 1.0]]}
    result_indep, _, _ = blocks.simulate_credit_portfolio(dependency=independent, block_id="b_a", **kwargs)
    result_corr, _, _ = blocks.simulate_credit_portfolio(dependency=correlated, block_id="b_b", **kwargs)
    assert result_corr["quantiles"][0.999] > result_indep["quantiles"][0.999]


def test_multi_factor_simulation_contributions_sum_to_grand_tvar():
    df = _portfolio_df(8, seed=11)
    dependency = {"kind": "dependency", "type": "gaussian", "labels": ["only_sector"], "corr": [[1.0]]}
    result, _, contributions = blocks.simulate_credit_portfolio(
        df,
        dependency,
        pd_col="pd",
        lgd_col="lgd",
        ead_col="ead",
        sector_col="sector",
        name_col="obligor",
        n_paths=50_000,
        chunk_size=10_000,
        alpha_levels=[0.99],
        seed=13,
        compute_contributions=True,
    )
    assert set(contributions["segment"].to_list()) == set(df["obligor"].to_list())
    assert contributions["euler_contribution"].sum() == pytest.approx(result["tvar"][0.99], rel=1e-6)


def test_simulate_credit_portfolio_rejects_unknown_sector():
    df = _portfolio_df(3)
    df = df.with_columns(pl.Series("sector", ["only_sector", "only_sector", "somewhere_else"]))
    dependency = {"kind": "dependency", "type": "gaussian", "labels": ["only_sector"], "corr": [[1.0]]}
    with pytest.raises(ValueError, match="somewhere_else"):
        blocks.simulate_credit_portfolio(
            df, dependency, pd_col="pd", lgd_col="lgd", ead_col="ead", sector_col="sector", n_paths=1000
        )
