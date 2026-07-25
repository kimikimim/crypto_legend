"""Scoring engine: exact 0-100 arithmetic under the smart-money weights
(Trend 20 / Location 25 / Whale 25 / Momentum 15 / Volatility 15),
breakout penalty, determinism, and bounds."""

from __future__ import annotations

import pandas as pd
import pytest

from app.fibonacci import ConfluenceZone, FibLevel
from app.liquidations import NO_SIGNAL, LiquidationSignal
from app.scoring import ScoringContext, ScoringEngine
from app.smc import FairValueGap, LiquiditySweep, OrderBlock, RawBreakout

T0 = pd.Timestamp("2026-01-01", tz="UTC")


@pytest.fixture
def engine(config) -> ScoringEngine:
    return ScoringEngine(config)


def perfect_long_row(base_row: dict) -> dict:
    row = dict(base_row)
    row.update(
        # Trend: fully aligned up on 4h, 1h, 15m.
        ema120_4h=105.0, ema200_4h=100.0, close_4h=110.0,
        ema120_1h=104.0, ema200_1h=100.0, close_1h=108.0,
        ema120=105.0, close=107.0,
        # Candle dipped into the demand POIs and closed back above them.
        open=106.3, high=107.5, low=106.2, atr=1.0,
        # Momentum: fresh oversold reclaim + bullish MACD cross.
        rsi=45.0, rsi_cross_up_recent=True, macd_cross_up_recent=True,
        # Volatility: 2.5x volume spike.
        vol_ratio=2.5,
    )
    return row


def perfect_short_row(base_row: dict) -> dict:
    row = dict(base_row)
    row.update(
        ema120_4h=95.0, ema200_4h=100.0, close_4h=90.0,
        ema120_1h=96.0, ema200_1h=100.0, close_1h=92.0,
        ema120=95.0, close=93.0,
        open=93.7, high=93.8, low=92.5, atr=1.0,
        rsi=55.0, rsi_cross_dn_recent=True, macd_cross_dn_recent=True,
        vol_ratio=2.5,
    )
    return row


LONG_ZONE = ConfluenceZone(
    low=106.0, high=106.4, ratios=(0.5, 0.618), sources=("4h:0.618", "1h:0.5")
)
SHORT_ZONE = ConfluenceZone(
    low=93.6, high=94.0, ratios=(0.5, 0.618), sources=("4h:0.5", "1h:0.618")
)
LONG_OB = OrderBlock(low=105.8, high=106.5, side="bullish", timeframe="1h", time=T0)
SHORT_OB = OrderBlock(low=93.5, high=94.2, side="bearish", timeframe="1h", time=T0)
LONG_FVG = FairValueGap(low=106.1, high=106.6, side="bullish", timeframe="15m", time=T0)
SHORT_FVG = FairValueGap(low=93.4, high=93.9, side="bearish", timeframe="15m", time=T0)
LONG_SWEEP = LiquiditySweep(level=106.3, level_timeframe="1h", side="low", time=T0)
SHORT_SWEEP = LiquiditySweep(level=93.7, level_timeframe="1h", side="high", time=T0)
LONG_FLUSH = LiquidationSignal(True, False, "measured", ("long flush detail",))
SHORT_FLUSH = LiquidationSignal(False, True, "measured", ("short flush detail",))

PERFECT_LONG_CTX = ScoringContext(
    zones=[LONG_ZONE], order_blocks=[LONG_OB], fvgs=[LONG_FVG],
    sweeps=[LONG_SWEEP], liquidations=LONG_FLUSH,
)
PERFECT_SHORT_CTX = ScoringContext(
    zones=[SHORT_ZONE], order_blocks=[SHORT_OB], fvgs=[SHORT_FVG],
    sweeps=[SHORT_SWEEP], liquidations=SHORT_FLUSH,
)


def test_perfect_long_setup_scores_exactly_100(engine, base_row):
    result = engine.score(perfect_long_row(base_row), PERFECT_LONG_CTX)
    assert result.long.trend.points == 20.0
    assert result.long.location.points == 25.0
    assert result.long.whale.points == 25.0
    assert result.long.momentum.points == 15.0
    assert result.long.volatility.points == 15.0
    assert result.long.total == 100.0
    # The mirrored short read must be weak.
    assert result.short.total <= 20.0


def test_perfect_short_setup_scores_exactly_100(engine, base_row):
    result = engine.score(perfect_short_row(base_row), PERFECT_SHORT_CTX)
    assert result.short.total == 100.0
    assert result.short.trend.points == 20.0
    assert result.short.location.points == 25.0
    assert result.short.whale.points == 25.0
    assert result.short.momentum.points == 15.0
    assert result.short.volatility.points == 15.0
    assert result.long.total <= 20.0


def test_neutral_chop_scores_zero(engine, base_row):
    result = engine.score(base_row, ScoringContext())
    assert result.long.total == 0.0
    assert result.short.total == 0.0


def test_whale_sweep_alone_gives_15(engine, base_row):
    result = engine.score(base_row, ScoringContext(sweeps=[LONG_SWEEP]))
    assert result.long.whale.points == 15.0
    assert result.short.whale.points == 0.0


def test_liquidation_flush_alone_gives_10(engine, base_row):
    result = engine.score(base_row, ScoringContext(liquidations=SHORT_FLUSH))
    assert result.short.whale.points == 10.0
    assert result.long.whale.points == 0.0


def test_sweep_plus_flush_stack_to_25(engine, base_row):
    ctx = ScoringContext(sweeps=[LONG_SWEEP], liquidations=LONG_FLUSH)
    result = engine.score(base_row, ctx)
    assert result.long.whale.points == 25.0


def test_sweep_direction_is_mean_reversion(engine, base_row):
    # A sweep of the HIGHS rewards the SHORT side, never the long.
    result = engine.score(base_row, ScoringContext(sweeps=[SHORT_SWEEP]))
    assert result.short.whale.points == 15.0
    assert result.long.whale.points == 0.0


def test_raw_breakout_penalizes_breakout_direction(engine, base_row):
    row = dict(base_row)
    row.update(
        ema120_4h=105.0, ema200_4h=100.0, close_4h=110.0,
        ema120_1h=104.0, ema200_1h=100.0, close_1h=108.0,
        ema120=105.0, close=107.0, vol_ratio=2.5,
    )
    breakout = RawBreakout(level=106.5, level_timeframe="1h", side="up", time=T0)
    ctx = ScoringContext(breakouts=[breakout])
    result = engine.score(row, ctx)
    no_penalty = engine.score(row, ScoringContext())
    assert result.long.penalty == 15.0
    assert result.long.total == max(no_penalty.long.total - 15.0, 0.0)
    assert result.short.penalty == 0.0


def test_breakout_penalty_floors_at_zero(engine, base_row):
    breakout = RawBreakout(level=100.5, level_timeframe="4h", side="up", time=T0)
    result = engine.score(base_row, ScoringContext(breakouts=[breakout]))
    assert result.long.total == 0.0  # 0 - 15 floored, never negative


def test_ob_and_fvg_touch_score_without_fib(engine, base_row):
    row = dict(base_row)
    row.update(close=107.0, low=106.2, high=107.5, open=106.3, atr=1.0)
    ctx = ScoringContext(order_blocks=[LONG_OB], fvgs=[LONG_FVG])
    result = engine.score(row, ctx)
    # OB 7 + FVG 5 + reaction 3 (closed above both bands) = 15.
    assert result.long.location.points == 15.0


def test_single_tf_fib_level_scores_5(engine, base_row):
    row = dict(base_row)
    row.update(close=100.0, low=99.5, high=100.2, atr=1.0)
    level = FibLevel(price=99.8, ratio=0.618, timeframe="4h")
    result = engine.score(row, ScoringContext(levels_4h=[level]))
    # 5 (single level) + 3 (reaction close above) = 8.
    assert result.long.location.points == 8.0


def test_wrong_side_pois_give_no_long_credit(engine, base_row):
    row = dict(base_row)
    row.update(close=100.0, low=99.5, high=100.5, atr=1.0)
    ctx = ScoringContext(
        zones=[ConfluenceZone(low=104.0, high=104.4, ratios=(0.5,),
                              sources=("4h:0.5", "1h:0.5"))],
        order_blocks=[OrderBlock(low=93.5, high=94.2, side="bearish",
                                 timeframe="1h", time=T0)],
    )
    result = engine.score(row, ctx)
    assert result.long.location.points == 0.0


def test_volume_spike_at_sr_vs_away(engine, base_row):
    row = dict(base_row)
    row.update(close=100.0, low=99.5, high=100.2, atr=1.0, vol_ratio=2.5)
    at_sr = engine.score(
        row, ScoringContext(levels_1h=[FibLevel(99.8, 0.5, "1h")])
    )
    away = engine.score(row, ScoringContext())
    assert at_sr.long.volatility.points == 15.0
    assert away.long.volatility.points == 5.0


def test_momentum_partial_credit(engine, base_row):
    row = dict(base_row)
    row.update(rsi=55.0, rsi_rising=True, macd_hist=0.5, hist_rising=True)
    result = engine.score(row, ScoringContext())
    assert result.long.momentum.points == 7.0  # 4 (RSI slope) + 3 (MACD hist)


def test_scores_are_deterministic(engine, base_row):
    row = perfect_long_row(base_row)
    assert engine.score(row, PERFECT_LONG_CTX) == engine.score(row, PERFECT_LONG_CTX)


def test_totals_equal_category_sums_and_stay_in_bounds(engine, base_row):
    scenarios = [
        (dict(base_row), ScoringContext()),
        (perfect_long_row(base_row), PERFECT_LONG_CTX),
        (perfect_short_row(base_row), PERFECT_SHORT_CTX),
        (dict(base_row), ScoringContext(
            breakouts=[RawBreakout(100.5, "1h", "up", T0),
                       RawBreakout(99.5, "1h", "down", T0)],
            liquidations=NO_SIGNAL,
        )),
    ]
    for row, ctx in scenarios:
        result = engine.score(row, ctx)
        for d in (result.long, result.short):
            cat_sum = (
                d.trend.points + d.location.points + d.whale.points
                + d.momentum.points + d.volatility.points - d.penalty
            )
            assert d.total == pytest.approx(max(min(cat_sum, 100.0), 0.0))
            assert 0.0 <= d.total <= 100.0


def test_nan_inputs_score_zero_not_crash(engine, base_row):
    row = {k: float("nan") for k in base_row}
    result = engine.score(row, PERFECT_LONG_CTX)
    # Whale category is context-driven (sweep+flush) so it still scores 25,
    # but every row-driven category must be 0 and nothing may crash.
    assert result.long.trend.points == 0.0
    assert result.long.location.points == 0.0
    assert result.long.momentum.points == 0.0
    assert result.long.volatility.points == 0.0
    assert result.long.total == 25.0
