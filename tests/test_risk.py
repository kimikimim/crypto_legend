"""Regime filter, regime penalty, and structural SL/TP/sizing plans."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.fibonacci import FibLevel, SwingLeg
from app.risk import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
    TradePlanner,
    determine_regime,
)
from app.scoring import ScoringContext, ScoringEngine
from app.smc import LiquiditySweep, OrderBlock, SwingLevel
from tests.conftest import make_ohlcv

T0 = pd.Timestamp("2026-01-01", tz="UTC")


# ----------------------------------------------------------------------
# Regime determination
# ----------------------------------------------------------------------
def test_rising_daily_market_is_bull(config):
    df = make_ohlcv(np.linspace(100, 400, 300), "1D")
    assert determine_regime(df, config) == REGIME_BULL


def test_falling_daily_market_is_bear(config):
    df = make_ohlcv(np.linspace(400, 100, 300), "1D")
    assert determine_regime(df, config) == REGIME_BEAR


def test_flat_daily_market_is_chop(config):
    df = make_ohlcv(np.full(300, 100.0), "1D")
    assert determine_regime(df, config) == REGIME_CHOP


def test_short_history_defaults_to_chop(config):
    df = make_ohlcv(np.linspace(100, 400, 100), "1D")
    assert determine_regime(df, config) == REGIME_CHOP


# ----------------------------------------------------------------------
# Regime penalty in scoring
# ----------------------------------------------------------------------
def test_long_in_bear_regime_loses_20(config, base_row):
    engine = ScoringEngine(config)
    row = dict(base_row)
    row.update(
        ema120_4h=105.0, ema200_4h=100.0, close_4h=110.0,
        ema120_1h=104.0, ema200_1h=100.0, close_1h=108.0,
        ema120=105.0, close=107.0, vol_ratio=2.5,
    )
    bear = engine.score(row, ScoringContext(regime=REGIME_BEAR))
    chop = engine.score(row, ScoringContext(regime=REGIME_CHOP))
    assert bear.long.penalty == 20.0
    assert bear.long.total == max(chop.long.total - 20.0, 0.0)
    assert bear.short.penalty == 0.0   # short is WITH the bear regime


def test_short_in_bull_regime_loses_20(config, base_row):
    engine = ScoringEngine(config)
    result = engine.score(base_row, ScoringContext(regime=REGIME_BULL))
    assert result.short.penalty == 20.0
    assert result.long.penalty == 0.0


# ----------------------------------------------------------------------
# Trade planner
# ----------------------------------------------------------------------
@pytest.fixture
def m15() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100.2, "high": 100.5, "low": 99.8,
            "close": 100.0, "volume": 1000.0, "atr": 1.0,
        },
        index=idx,
    )
    # Last candle: the sweep candle with a deep stop-hunt wick.
    df.iloc[-1, df.columns.get_loc("low")] = 98.6
    df.iloc[-1, df.columns.get_loc("close")] = 100.0
    return df


@pytest.fixture
def planner(config) -> TradePlanner:
    return TradePlanner(config)


def _plan(planner, m15, **overrides):
    args = dict(
        m15=m15, zones=[], levels=[], order_blocks=[], fvgs=[],
        sweeps=[], swing_levels=[], leg_1h=None,
    )
    args.update(overrides)
    side = args.pop("side", "long")
    return planner.plan(side, **args)


def test_sl_sits_beyond_sweep_wick_plus_buffer(planner, m15):
    sweep = LiquiditySweep(level=99.0, level_timeframe="1h", side="low",
                           time=m15.index[-1])
    plan = _plan(planner, m15, sweeps=[sweep])
    # Wick low 98.6 minus 0.5 x ATR(1.0) buffer.
    assert plan.suggested_sl == pytest.approx(98.1)
    assert plan.sl_basis == "sweep_wick"


def test_sl_fallbacks_walk_the_structure_hierarchy(planner, m15):
    no_structure = _plan(planner, m15)
    assert no_structure.sl_basis == "atr_fallback"
    assert no_structure.suggested_sl == pytest.approx(100.0 - 1.5)

    with_swing = _plan(
        planner, m15,
        swing_levels=[SwingLevel(95.0, "low", "1h", T0)],
    )
    assert with_swing.sl_basis == "swing_level"
    assert with_swing.suggested_sl == pytest.approx(94.5)

    with_ob = _plan(
        planner, m15,
        order_blocks=[OrderBlock(99.2, 99.9, "bullish", "1h", T0)],
    )
    assert with_ob.sl_basis == "poi_band"
    assert with_ob.suggested_sl == pytest.approx(98.7)  # 99.2 - 0.5


def test_tp1_targets_nearest_fib_or_opposing_ob(planner, m15):
    plan = _plan(
        planner, m15,
        levels=[FibLevel(103.0, 0.382, "4h"), FibLevel(101.8, 0.5, "1h")],
        order_blocks=[OrderBlock(104.0, 104.8, "bearish", "1h", T0)],
    )
    assert plan.suggested_tp1 == pytest.approx(101.8)   # closest valid target


def test_tp2_targets_fib_extension_of_1h_leg(planner, m15):
    leg = SwingLeg(low=90.0, high=100.0, direction="up",
                   low_time=T0, high_time=T0)
    plan = _plan(planner, m15, leg_1h=leg,
                 levels=[FibLevel(101.8, 0.5, "1h")])
    # 90 + 1.618 * 10 = 106.18
    assert plan.suggested_tp2 == pytest.approx(106.18)


def test_plan_ordering_is_always_valid(planner, m15):
    for side in ("long", "short"):
        plan = _plan(planner, m15, side=side)
        if side == "long":
            assert plan.suggested_sl < 100.0 < plan.suggested_tp1 < plan.suggested_tp2
        else:
            assert plan.suggested_tp2 < plan.suggested_tp1 < 100.0 < plan.suggested_sl


def test_wider_stop_means_smaller_position(planner, m15):
    tight = _plan(planner, m15,
                  sweeps=[LiquiditySweep(99.0, "1h", "low", m15.index[-1])])
    wide = _plan(planner, m15,
                 swing_levels=[SwingLevel(92.0, "low", "1h", T0)])
    assert abs(100.0 - wide.suggested_sl) > abs(100.0 - tight.suggested_sl)
    assert wide.risk_weight < tight.risk_weight
    assert wide.suggested_leverage < tight.suggested_leverage


def test_leverage_capped_and_risk_weight_bounded(planner, m15, config):
    plans = [
        _plan(planner, m15),
        _plan(planner, m15, sweeps=[LiquiditySweep(99.9, "1h", "low", m15.index[-1])]),
        _plan(planner, m15, side="short"),
    ]
    for plan in plans:
        assert 0.0 < plan.risk_weight <= 1.0
        assert 0.0 < plan.suggested_leverage <= config.max_leverage


def test_constant_account_risk_math(planner, m15, config):
    plan = _plan(planner, m15)  # SL 98.5 -> risk 1.5%
    expected_leverage = min(
        config.max_leverage, (config.account_risk_pct / 100.0) / (1.5 / 100.0)
    )
    assert plan.suggested_leverage == pytest.approx(expected_leverage, abs=0.01)
    assert plan.risk_weight == pytest.approx(
        expected_leverage / config.max_leverage, abs=0.01
    )


def test_plan_none_when_atr_unusable(planner, m15):
    broken = m15.copy()
    broken["atr"] = float("nan")
    assert _plan(planner, broken) is None
