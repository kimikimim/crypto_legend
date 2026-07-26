"""Regime filter, regime penalty, and structural SL/TP/sizing plans."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.fibonacci import FibLevel, SwingLeg
from app.config import EngineConfig
from app.risk import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
    MarketState,
    TradePlanner,
    decide_verdict,
    determine_regime,
    round_trip_cost_pct,
    slippage_pct,
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


def test_realized_risk_equals_the_account_limit(planner, m15, config):
    """The invariant that matters: leverage x stop distance (incl. costs)
    always lands on the configured account risk, whatever the stop width."""
    for overrides in (
        {},
        {"sweeps": [LiquiditySweep(99.0, "1h", "low", m15.index[-1])]},
        {"swing_levels": [SwingLevel(92.0, "low", "1h", T0)]},
    ):
        plan = _plan(planner, m15, **overrides)
        price = 100.0
        cost = price * plan.cost_pct / 100.0
        stop_frac = (abs(price - plan.suggested_sl) + cost) / price
        realized_risk_pct = plan.suggested_leverage * stop_frac * 100.0
        assert realized_risk_pct == pytest.approx(config.account_risk_pct, abs=0.02)
        assert plan.risk_weight == pytest.approx(
            plan.suggested_leverage / config.max_leverage, abs=0.01
        )


def test_plan_none_when_atr_unusable(planner, m15):
    broken = m15.copy()
    broken["atr"] = float("nan")
    assert _plan(planner, broken) is None


# ----------------------------------------------------------------------
# Cost model & net-RR gate
# ----------------------------------------------------------------------
def _calm_cost(config, price=100.0, atr=1.0) -> float:
    return round_trip_cost_pct(MarketState(price=price, atr=atr), config)


def test_net_rr_is_gross_rr_minus_round_trip_costs(planner, m15, config):
    # SL 98.5 (risk 1.5), TP1 forced to the 2x ATR fallback -> 102.0 (reward 2.0).
    plan = _plan(planner, m15)
    cost = 100.0 * _calm_cost(config) / 100.0
    assert plan.rr_tp1 == pytest.approx(2.0 / 1.5, abs=0.01)
    assert plan.rr_tp1_net == pytest.approx((2.0 - cost) / (1.5 + cost), abs=0.01)
    assert plan.rr_tp1_net < plan.rr_tp1  # costs always hurt
    assert plan.cost_pct == pytest.approx(_calm_cost(config), abs=1e-4)


def test_thin_reward_setup_is_rejected(planner, m15):
    # TP1 barely above price: gross RR ~0.13 -> nowhere near the 1.5 gate.
    plan = _plan(planner, m15, levels=[FibLevel(100.6, 0.382, "1h")])
    assert plan.tradeable is False
    assert plan.reject_reasons
    assert "net RR at TP1" in plan.reject_reasons[0]


def test_generous_reward_setup_is_tradeable(planner, m15):
    plan = _plan(planner, m15, levels=[FibLevel(105.0, 0.382, "1h")])
    assert plan.tradeable is True
    assert plan.reject_reasons == ()
    assert plan.rr_tp1_net >= 1.5


def test_sizing_accounts_for_costs(planner, m15, config):
    plan = _plan(planner, m15)  # SL 1.5 wide on a 100 price
    cost = 100.0 * _calm_cost(config) / 100.0
    expected_lev = min(
        config.max_leverage,
        (config.account_risk_pct / 100.0) / ((1.5 + cost) / 100.0),
    )
    assert plan.suggested_leverage == pytest.approx(expected_lev, abs=0.01)


def test_zero_cost_config_makes_net_equal_gross(m15):
    free = EngineConfig(
        taker_fee_pct=0.0, slippage_base_pct=0.0, slippage_atr_coef=0.0
    )
    plan = _plan(TradePlanner(free), m15)
    assert plan.rr_tp1_net == pytest.approx(plan.rr_tp1, abs=0.01)
    assert plan.cost_pct == 0.0


# ----------------------------------------------------------------------
# Dynamic slippage: a flat assumption is wrong exactly where this engine
# trades — sweeps and squeezes clear the book out.
# ----------------------------------------------------------------------
def test_slippage_scales_with_volatility(config):
    quiet = slippage_pct(MarketState(price=100.0, atr=0.1), config)
    volatile = slippage_pct(MarketState(price=100.0, atr=2.0), config)
    assert volatile > quiet
    # 0.1% ATR -> base + 0.12 * 0.1; 2% ATR -> base + 0.12 * 2.
    assert quiet == pytest.approx(config.slippage_base_pct + 0.12 * 0.1, abs=1e-6)
    assert volatile == pytest.approx(config.slippage_base_pct + 0.12 * 2.0, abs=1e-6)


def test_stressed_tape_multiplies_slippage(config):
    calm = MarketState(price=100.0, atr=1.0, stressed=False)
    thin = MarketState(price=100.0, atr=1.0, stressed=True)
    assert slippage_pct(thin, config) == pytest.approx(
        slippage_pct(calm, config) * config.slippage_stress_mult
    )


def test_slippage_is_capped(config):
    absurd = slippage_pct(MarketState(price=100.0, atr=50.0, stressed=True), config)
    assert absurd == config.slippage_cap_pct


def test_unusable_atr_falls_back_to_the_floor(config):
    assert slippage_pct(MarketState(price=100.0, atr=float("nan")), config) == (
        config.slippage_base_pct
    )
    assert slippage_pct(MarketState(price=0.0, atr=1.0), config) == (
        config.slippage_base_pct
    )


def test_a_sweep_entry_is_charged_thin_book_costs(planner, m15):
    """The sweep is the setup and the liquidity hole at the same time; the
    plan must price that, not the quiet-market figure."""
    calm = _plan(planner, m15)
    swept = _plan(
        planner, m15, sweeps=[LiquiditySweep(99.0, "1h", "low", m15.index[-1])]
    )
    assert swept.stressed_entry is True
    assert calm.stressed_entry is False
    assert swept.cost_pct > calm.cost_pct
    assert swept.rr_tp1_net < calm.rr_tp1_net  # same levels, worse economics


def test_volume_spike_alone_triggers_thin_book_pricing(planner, m15):
    spiky = m15.copy()
    spiky["vol_ratio"] = 3.0
    plan = _plan(planner, spiky)
    assert plan.stressed_entry is True
    assert plan.cost_pct > _plan(planner, m15).cost_pct


# ----------------------------------------------------------------------
# Time barrier, enforced live and not only in the labeler
# ----------------------------------------------------------------------
def test_plan_carries_a_time_stop(planner, m15, config):
    plan = _plan(planner, m15)
    expected = m15.index[-1] + pd.Timedelta(minutes=15) * (1 + config.max_hold_bars)
    assert plan.time_stop_at == expected


def test_time_stop_honours_a_shorter_holding_period():
    short = EngineConfig(max_hold_bars=8)
    idx = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.2, "high": 100.5, "low": 99.8, "close": 100.0,
         "volume": 1000.0, "atr": 1.0},
        index=idx,
    )
    plan = _plan(TradePlanner(short), df)
    assert plan.time_stop_at == idx[-1] + pd.Timedelta(minutes=15) * 9


# ----------------------------------------------------------------------
# Verdict: score threshold AND net-RR gate, one shared decision path
# ----------------------------------------------------------------------
def _fake_plan(side: str, tradeable: bool, reasons: tuple[str, ...] = ()):
    from app.risk import TradePlan

    return TradePlan(
        side=side, entry_zone_low=99.0, entry_zone_high=100.0,
        suggested_sl=98.0, suggested_tp1=103.0, suggested_tp2=106.0,
        risk_weight=0.5, suggested_leverage=2.5, rr_tp1=2.5,
        rr_tp1_net=2.2 if tradeable else 0.3, sl_basis="sweep_wick",
        tradeable=tradeable, cost_pct=0.36, reject_reasons=reasons,
    )


def test_verdict_long_when_score_and_rr_both_pass(config):
    verdict, reasons = decide_verdict(
        70.0, _fake_plan("long", True), 20.0, _fake_plan("short", True), config
    )
    assert verdict == "LONG"
    assert reasons == ()


def test_verdict_neutral_below_score_threshold(config):
    verdict, reasons = decide_verdict(
        16.0, _fake_plan("long", True), 12.0, _fake_plan("short", True), config
    )
    assert verdict == "NEUTRAL"
    assert any("below 40 threshold" in r for r in reasons)


def test_verdict_neutral_when_rr_gate_fails_despite_high_score(config):
    verdict, reasons = decide_verdict(
        85.0,
        _fake_plan("long", False, ("net RR at TP1 0.30 < required 1.50",)),
        10.0,
        _fake_plan("short", True),
        config,
    )
    assert verdict == "NEUTRAL"
    assert any("net RR" in r for r in reasons)


def test_verdict_falls_to_other_side_when_top_side_is_untradeable(config):
    # Long scores higher but fails the RR gate; short qualifies outright.
    verdict, _ = decide_verdict(
        75.0,
        _fake_plan("long", False, ("net RR at TP1 0.20 < required 1.50",)),
        60.0,
        _fake_plan("short", True),
        config,
    )
    assert verdict == "SHORT"


def test_verdict_neutral_when_plan_missing(config):
    verdict, reasons = decide_verdict(90.0, None, 88.0, None, config)
    assert verdict == "NEUTRAL"
    assert all("no computable trade plan" in r for r in reasons)
