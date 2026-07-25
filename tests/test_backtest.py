"""Triple-barrier labeling: path resolution, tie handling, and cost math."""

from __future__ import annotations

import pandas as pd
import pytest

from app.backtest import BARRIER_SL, BARRIER_TIMEOUT, BARRIER_TP1, TripleBarrierLabeler

T0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")


@pytest.fixture
def labeler(config) -> TripleBarrierLabeler:
    return TripleBarrierLabeler(config)


def path(bars: list[tuple[float, float, float]]) -> pd.DataFrame:
    """bars: (high, low, close), one minute apart."""
    idx = pd.date_range(T0, periods=len(bars), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [b[2] for b in bars],
            "high": [b[0] for b in bars],
            "low": [b[1] for b in bars],
            "close": [b[2] for b in bars],
            "volume": 1.0,
        },
        index=idx,
    )


LONG = dict(side="long", entry_price=100.0, sl=99.0, tp1=102.0, tp2=104.0)
SHORT = dict(side="short", entry_price=100.0, sl=101.0, tp1=98.0, tp2=96.0)


# ----------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------
def test_long_take_profit_hit_first(labeler):
    out = labeler.resolve_one(**LONG, path=path([
        (100.5, 99.8, 100.2), (101.0, 100.0, 100.8), (102.5, 100.9, 102.2),
    ]))
    assert out["barrier"] == BARRIER_TP1
    assert out["exit_price"] == 102.0
    assert out["gross_pnl_pct"] == pytest.approx(2.0)
    assert out["minutes_held"] == 3


def test_long_stop_hit_first(labeler):
    out = labeler.resolve_one(**LONG, path=path([
        (100.2, 99.5, 99.8), (100.0, 98.7, 98.9), (103.0, 98.8, 102.5),
    ]))
    assert out["barrier"] == BARRIER_SL
    assert out["exit_price"] == 99.0
    assert out["gross_pnl_pct"] == pytest.approx(-1.0)
    assert out["minutes_held"] == 2  # resolved before the later rally


def test_short_side_is_mirrored(labeler):
    win = labeler.resolve_one(**SHORT, path=path([(100.2, 97.5, 97.8)]))
    assert win["barrier"] == BARRIER_TP1
    assert win["gross_pnl_pct"] == pytest.approx(2.0)

    loss = labeler.resolve_one(**SHORT, path=path([(101.6, 99.9, 101.4)]))
    assert loss["barrier"] == BARRIER_SL
    assert loss["gross_pnl_pct"] == pytest.approx(-1.0)


def test_bar_straddling_both_levels_resolves_as_a_loss(labeler):
    """The decisive conservatism: inside one minute the true order is
    unknowable, so the stop is assumed to have come first."""
    out = labeler.resolve_one(**LONG, path=path([(102.5, 98.5, 101.0)]))
    assert out["barrier"] == BARRIER_SL


def test_timeout_closes_at_the_last_price(labeler):
    out = labeler.resolve_one(**LONG, path=path([
        (100.4, 99.6, 100.1), (100.5, 99.7, 100.3), (100.6, 99.8, 100.45),
    ]))
    assert out["barrier"] == BARRIER_TIMEOUT
    assert out["exit_price"] == 100.45
    assert out["gross_pnl_pct"] == pytest.approx(0.45)


def test_empty_path_is_unresolvable(labeler):
    assert labeler.resolve_one(**LONG, path=path([])) is None


def test_tp2_counts_only_when_reached_before_the_stop(labeler):
    reached = labeler.resolve_one(**LONG, path=path([(104.5, 99.9, 104.2)]))
    assert reached["tp2_reached"] is True

    stopped_first = labeler.resolve_one(**LONG, path=path([
        (100.2, 98.5, 98.7), (104.5, 98.6, 104.2),
    ]))
    assert stopped_first["barrier"] == BARRIER_SL
    assert stopped_first["tp2_reached"] is False


# ----------------------------------------------------------------------
# Cost and R-multiple math
# ----------------------------------------------------------------------
def test_costs_are_deducted_from_every_outcome(labeler, config):
    win = labeler.resolve_one(**LONG, path=path([(102.5, 99.9, 102.2)]))
    assert win["net_pnl_pct"] == pytest.approx(
        win["gross_pnl_pct"] - config.round_trip_cost_pct
    )
    loss = labeler.resolve_one(**LONG, path=path([(100.1, 98.5, 98.7)]))
    # Costs make a loss worse, never better.
    assert loss["net_pnl_pct"] < loss["gross_pnl_pct"]


def test_r_multiple_uses_the_same_risk_as_position_sizing(labeler, config):
    out = labeler.resolve_one(**LONG, path=path([(102.5, 99.9, 102.2)]))
    risk_pct = 1.0 + config.round_trip_cost_pct  # 1% stop + round trip
    assert out["r_multiple"] == pytest.approx(out["net_pnl_pct"] / risk_pct)


def test_a_stop_out_is_close_to_minus_one_r(labeler):
    out = labeler.resolve_one(**LONG, path=path([(100.1, 98.5, 98.7)]))
    assert out["r_multiple"] == pytest.approx(-1.0, abs=0.01)


# ----------------------------------------------------------------------
# Batch run
# ----------------------------------------------------------------------
def _signals() -> pd.DataFrame:
    rows = [
        {"evaluated_at": T0, "symbol": "BTC/USDT:USDT", "verdict": "LONG",
         "side": "long", "price": 100.0, "sl": 99.0, "tp1": 102.0, "tp2": 104.0,
         "long_score": 72.0, "short_score": 10.0, "regime": "bull",
         "rr_tp1_net": 1.8, "sl_basis": "sweep_wick", "risk_weight": 0.5},
        {"evaluated_at": T0 + pd.Timedelta(hours=2), "symbol": "BTC/USDT:USDT",
         "verdict": "NEUTRAL", "side": None, "price": 100.0, "sl": None,
         "tp1": None, "tp2": None, "long_score": 12.0, "short_score": 8.0,
         "regime": "bull", "rr_tp1_net": None, "sl_basis": None,
         "risk_weight": None},
    ]
    return pd.DataFrame(rows).set_index("evaluated_at")


def test_run_labels_actionable_signals_only(labeler):
    m1 = path([(100.5, 99.7, 100.2)] * 30)
    m1.index = pd.date_range(
        T0 + pd.Timedelta(minutes=15), periods=30, freq="1min", tz="UTC"
    )
    report = labeler.run(_signals(), m1)
    assert report.signals_in == 1        # the NEUTRAL row is not a trade
    assert report.resolved == 1
    assert report.coverage == 1.0
    trade = report.trades.iloc[0]
    assert trade["side"] == "long"
    assert trade["score"] == 72.0
    assert trade["entry_at"] == T0 + pd.Timedelta(minutes=15)


def test_entry_is_the_signal_bar_close_not_the_signal_bar_open(labeler):
    """A trade cannot be entered before the bar that produced it has closed."""
    m1 = path([(105.0, 95.0, 100.0)] * 40)
    m1.index = pd.date_range(T0, periods=40, freq="1min", tz="UTC")
    report = labeler.run(_signals(), m1)
    assert report.trades.iloc[0]["entry_at"] == T0 + pd.Timedelta(minutes=15)
    # Bars before that close must not have resolved the trade.
    assert report.trades.iloc[0]["exit_at"] >= T0 + pd.Timedelta(minutes=15)


def test_missing_1m_coverage_is_reported_not_silently_dropped(labeler):
    far_away = path([(100.5, 99.7, 100.2)] * 5)
    far_away.index = pd.date_range(
        T0 + pd.Timedelta(days=30), periods=5, freq="1min", tz="UTC"
    )
    report = labeler.run(_signals(), far_away)
    assert report.resolved == 0
    assert report.unresolved_missing_1m == 1
    assert report.coverage == 0.0
    assert report.trades.empty


def test_holding_period_is_capped(labeler):
    flat = path([(100.4, 99.7, 100.1)] * 600)
    flat.index = pd.date_range(
        T0 + pd.Timedelta(minutes=15), periods=600, freq="1min", tz="UTC"
    )
    report = labeler.run(_signals(), flat, max_hold_bars=8)  # 2 hours
    trade = report.trades.iloc[0]
    assert trade["barrier"] == BARRIER_TIMEOUT
    assert trade["minutes_held"] <= 120
