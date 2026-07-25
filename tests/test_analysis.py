"""Backtest analytics: performance math and the calibration verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import (
    baseline_comparison,
    barrier_mix,
    breakdown,
    performance,
    score_calibration,
)


def make_trades(r_multiples, scores=None, regimes=None, symbols=None) -> pd.DataFrame:
    n = len(r_multiples)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "r_multiple": r_multiples,
            "score": scores if scores is not None else np.full(n, 60.0),
            "regime": regimes if regimes is not None else ["bull"] * n,
            "symbol": symbols if symbols is not None else ["BTC/USDT:USDT"] * n,
            "side": ["long"] * n,
            "barrier": ["tp1" if r > 0 else "sl" for r in r_multiples],
            "net_pnl_pct": np.asarray(r_multiples) * 1.14,
        },
        index=idx,
    )


# ----------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------
def test_performance_basics():
    stats = performance(make_trades([2.0, -1.0, 2.0, -1.0]))
    assert stats.trades == 4
    assert stats.win_rate == 0.5
    assert stats.expectancy_r == pytest.approx(0.5)
    assert stats.profit_factor == pytest.approx(2.0)
    assert stats.total_r == pytest.approx(2.0)
    assert stats.avg_win_r == pytest.approx(2.0)
    assert stats.avg_loss_r == pytest.approx(-1.0)


def test_max_drawdown_tracks_the_worst_peak_to_trough():
    # +3 then four straight losses = 4R drawdown from the peak.
    stats = performance(make_trades([3.0, -1.0, -1.0, -1.0, -1.0]))
    assert stats.max_drawdown_r == pytest.approx(4.0)


def test_empty_input_is_safe():
    stats = performance(pd.DataFrame())
    assert stats.trades == 0 and stats.expectancy_r == 0.0


def test_significance_requires_both_edge_and_evidence():
    strong = performance(make_trades([1.0, 1.2, 0.9, 1.1] * 15))
    assert strong.significant

    # A tiny positive mean buried in large variance: the sample cannot
    # distinguish it from zero, so it must not be called an edge.
    rng = np.random.default_rng(0)
    noisy = performance(make_trades(list(rng.normal(0.03, 1.5, 60))))
    assert noisy.expectancy_r > 0
    assert not noisy.significant

    # A negative expectancy is never "significant", however clear the loss.
    losing = performance(make_trades([-1.0, -1.0, 0.5] * 20))
    assert losing.p_value < 0.05
    assert not losing.significant


# ----------------------------------------------------------------------
# Calibration — the decisive test
# ----------------------------------------------------------------------
def test_calibration_detects_an_informative_score():
    """Outcomes improving with score must be reported as informative."""
    scores, rs = [], []
    rng = np.random.default_rng(0)
    for score, mean_r in ((45, -0.5), (55, -0.1), (65, 0.3), (75, 0.8), (85, 1.4)):
        scores += [score] * 40
        rs += list(rng.normal(mean_r, 0.2, 40))
    result = score_calibration(make_trades(rs, scores=scores))
    assert result.spearman_rho > 0.5
    assert result.spearman_p < 0.05
    assert result.monotonic
    assert "INFORMATIVE" in result.verdict


def test_calibration_detects_a_worthless_score():
    """The outcome that must not be dressed up: score explains nothing."""
    rng = np.random.default_rng(1)
    n = 200
    scores = rng.uniform(40, 100, n)
    rs = rng.normal(0.0, 1.0, n)  # independent of score
    result = score_calibration(make_trades(rs, scores=scores))
    assert result.spearman_p >= 0.05
    assert "FLAT" in result.verdict


def test_calibration_detects_an_inverted_score():
    scores, rs = [], []
    rng = np.random.default_rng(2)
    for score, mean_r in ((45, 1.2), (65, 0.2), (85, -0.9)):
        scores += [score] * 50
        rs += list(rng.normal(mean_r, 0.2, 50))
    result = score_calibration(make_trades(rs, scores=scores))
    assert result.spearman_rho < 0
    assert "INVERTED" in result.verdict


def test_thin_buckets_are_flagged_not_trusted():
    scores = [45] * 3 + [85] * 3
    result = score_calibration(make_trades([0.1] * 6, scores=scores))
    assert any("below 20 trades" in n for n in result.notes)


def test_calibration_on_empty_input():
    assert score_calibration(pd.DataFrame()).verdict == "NO DATA"


# ----------------------------------------------------------------------
# Breakdowns
# ----------------------------------------------------------------------
def test_breakdown_splits_by_regime():
    trades = make_trades(
        [2.0, 2.0, -1.0, -1.0],
        regimes=["bull", "bull", "bear", "bear"],
    )
    table = breakdown(trades, "regime").set_index("regime")
    assert table.loc["bull", "expectancy_r"] == pytest.approx(2.0)
    assert table.loc["bear", "expectancy_r"] == pytest.approx(-1.0)


def test_breakdown_on_missing_column_is_empty():
    assert breakdown(make_trades([1.0]), "nonexistent").empty


def test_barrier_mix_shares_sum_to_one():
    mix = barrier_mix(make_trades([1.0, 1.0, -1.0, -1.0]))
    assert mix["share"].sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Random baseline
# ----------------------------------------------------------------------
def test_edge_over_random_is_detected():
    rng = np.random.default_rng(3)
    signal = make_trades(list(rng.normal(0.6, 0.5, 150)))
    random_ = make_trades(list(rng.normal(-0.1, 0.5, 150)))
    result = baseline_comparison(signal, random_)
    assert result["beats_random"] is True
    assert result["expectancy_gap_r"] > 0


def test_no_edge_over_random_is_reported_honestly():
    rng = np.random.default_rng(4)
    signal = make_trades(list(rng.normal(0.0, 1.0, 120)))
    random_ = make_trades(list(rng.normal(0.0, 1.0, 120)))
    assert baseline_comparison(signal, random_)["beats_random"] is False


def test_baseline_needs_both_samples():
    assert baseline_comparison(pd.DataFrame(), make_trades([1.0]))["comparable"] is False
