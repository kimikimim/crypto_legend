"""SMC structures: FVGs, order blocks, sweeps, breakouts, squeeze."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.indicators import IndicatorCalculator
from app.smc import (
    SwingLevel,
    detect_squeeze,
    find_fvgs,
    find_order_blocks,
    find_sweeps_and_breakouts,
    major_swing_levels,
)
from tests.conftest import make_ohlcv, zigzag_closes


def build_df(candles: list[dict], freq: str = "15min") -> pd.DataFrame:
    """Hand-crafted candle frame with the enriched columns SMC needs."""
    idx = pd.date_range("2026-01-01", periods=len(candles), freq=freq, tz="UTC")
    idx.name = "timestamp"
    df = pd.DataFrame(candles, index=idx)
    df["volume"] = df.get("volume", 1000.0)
    df["atr"] = df.get("atr", 1.0)
    df["vol_ratio"] = df.get("vol_ratio", 1.0)
    df["f_high"] = df["high"]
    df["f_low"] = df["low"]
    return df


def candle(o, h, l, c, vol_ratio=1.0):
    return {"open": o, "high": h, "low": l, "close": c, "vol_ratio": vol_ratio}


# ----------------------------------------------------------------------
# Fair Value Gaps
# ----------------------------------------------------------------------
def test_bullish_fvg_detected_with_exact_zone(config):
    df = build_df([
        candle(100, 101, 99, 100.5),
        candle(100.5, 103, 100.4, 102.8),   # impulse
        candle(102.9, 104, 102.0, 103.5),   # low 102 > high[0] 101 -> gap
        candle(103.5, 104.5, 102.5, 104.0),  # stays above -> unmitigated
    ])
    fvgs = find_fvgs(df, "15m", config)
    bull = [g for g in fvgs if g.side == "bullish"]
    assert len(bull) == 1
    assert bull[0].low == pytest.approx(101.0)
    assert bull[0].high == pytest.approx(102.0)


def test_filled_fvg_is_mitigated_and_dropped(config):
    df = build_df([
        candle(100, 101, 99, 100.5),
        candle(100.5, 103, 100.4, 102.8),
        candle(102.9, 104, 102.0, 103.5),
        candle(103.5, 104.5, 100.9, 101.2),  # low 100.9 <= 101 fills the gap
    ])
    assert [g for g in find_fvgs(df, "15m", config) if g.side == "bullish"] == []


def test_tiny_gap_below_min_atr_size_ignored(config):
    df = build_df([
        candle(100, 101, 99, 100.5),
        candle(100.5, 101.5, 100.4, 101.2),
        candle(101.2, 102, 101.1, 101.8),   # gap 101 -> 101.1 = 0.1 < 0.3*ATR(1)
    ])
    assert find_fvgs(df, "15m", config) == []


def test_bearish_fvg_detected(config):
    df = build_df([
        candle(100, 101, 99, 99.5),
        candle(99.5, 99.6, 96.5, 96.8),
        candle(96.8, 97.5, 95.5, 96.0),     # high 97.5 < low[0] 99 -> gap
        candle(96.0, 97.0, 95.0, 95.5),
    ])
    bears = [g for g in find_fvgs(df, "15m", config) if g.side == "bearish"]
    assert len(bears) == 1
    assert bears[0].low == pytest.approx(97.5)
    assert bears[0].high == pytest.approx(99.0)


# ----------------------------------------------------------------------
# Order Blocks
# ----------------------------------------------------------------------
def test_bullish_order_block_found_before_impulsive_bos(config):
    # Peak at idx5 (pivot high 106), decline, one bearish candle at idx15,
    # then a high-volume impulse at idx16 closing above the pivot high.
    candles = [
        candle(100, 101.5, 99.5, 101),
        candle(101, 102.5, 100.5, 102),
        candle(102, 103.5, 101.5, 103),
        candle(103, 104.5, 102.5, 104),
        candle(104, 105.5, 103.5, 105),
        candle(105, 106.0, 104.5, 105.5),           # pivot high f_high=106
        candle(105.5, 105.6, 103.9, 104),
        candle(104, 104.2, 102.9, 103),
        candle(103, 103.2, 101.9, 102),
        candle(102, 102.2, 100.9, 101),
        candle(101, 101.2, 99.9, 100),
        candle(100, 100.6, 99.6, 100.2),
        candle(100.2, 100.8, 99.8, 100.4),
        candle(100.4, 101.0, 100.0, 100.6),
        candle(100.6, 101.2, 100.2, 100.8),
        candle(100.8, 101.0, 99.3, 99.5),           # bearish OB candle (idx15)
        candle(99.5, 107.5, 99.4, 107, vol_ratio=3.0),  # impulsive BOS (idx16)
        candle(107, 107.5, 106.0, 106.5),
        candle(106.5, 107.0, 105.8, 106.2),
    ]
    df = build_df(candles)
    obs = find_order_blocks(df, "15m", config)
    bulls = [ob for ob in obs if ob.side == "bullish"]
    assert len(bulls) == 1
    assert bulls[0].low == pytest.approx(99.3)
    assert bulls[0].high == pytest.approx(101.0)
    assert bulls[0].time == df.index[15]


def test_order_block_requires_impulsive_volume(config):
    # Same structure but the breaking leg has no volume spike -> no OB.
    candles = [
        candle(100, 101.5, 99.5, 101),
        candle(101, 102.5, 100.5, 102),
        candle(102, 103.5, 101.5, 103),
        candle(103, 104.5, 102.5, 104),
        candle(104, 105.5, 103.5, 105),
        candle(105, 106.0, 104.5, 105.5),
        candle(105.5, 105.6, 103.9, 104),
        candle(104, 104.2, 102.9, 103),
        candle(103, 103.2, 101.9, 102),
        candle(102, 102.2, 100.9, 101),
        candle(101, 101.2, 99.9, 100),
        candle(100, 100.6, 99.6, 100.2),
        candle(100.2, 100.8, 99.8, 100.4),
        candle(100.4, 101.0, 100.0, 100.6),
        candle(100.6, 101.2, 100.2, 100.8),
        candle(100.8, 101.0, 99.3, 99.5),
        candle(99.5, 107.5, 99.4, 107, vol_ratio=1.0),  # no volume anomaly
        candle(107, 107.5, 106.0, 106.5),
        candle(106.5, 107.0, 105.8, 106.2),
    ]
    df = build_df(candles)
    assert [ob for ob in find_order_blocks(df, "15m", config)
            if ob.side == "bullish"] == []


def test_mitigated_order_block_dropped(config):
    # Price later closes below the OB low -> OB invalidated.
    candles = [
        candle(100, 101.5, 99.5, 101),
        candle(101, 102.5, 100.5, 102),
        candle(102, 103.5, 101.5, 103),
        candle(103, 104.5, 102.5, 104),
        candle(104, 105.5, 103.5, 105),
        candle(105, 106.0, 104.5, 105.5),
        candle(105.5, 105.6, 103.9, 104),
        candle(104, 104.2, 102.9, 103),
        candle(103, 103.2, 101.9, 102),
        candle(102, 102.2, 100.9, 101),
        candle(101, 101.2, 99.9, 100),
        candle(100, 100.6, 99.6, 100.2),
        candle(100.2, 100.8, 99.8, 100.4),
        candle(100.4, 101.0, 100.0, 100.6),
        candle(100.6, 101.2, 100.2, 100.8),
        candle(100.8, 101.0, 99.3, 99.5),
        candle(99.5, 107.5, 99.4, 107, vol_ratio=3.0),
        candle(107, 107.5, 106.0, 106.5),
        candle(106.5, 106.6, 98.0, 98.5),   # closes below OB low 99.3
    ]
    df = build_df(candles)
    assert [ob for ob in find_order_blocks(df, "15m", config)
            if ob.side == "bullish"] == []


# ----------------------------------------------------------------------
# Sweeps & breakouts
# ----------------------------------------------------------------------
def _flat_df(closes: list[float], highs: list[float], lows: list[float]):
    rows = []
    for c, h, l in zip(closes, highs, lows):
        rows.append(candle(c, h, l, c))
    return build_df(rows)


def test_wick_above_level_closing_back_inside_is_whale_sweep():
    level = SwingLevel(105.0, "high", "1h", pd.Timestamp("2026-01-01", tz="UTC"))
    df = _flat_df(
        closes=[104, 104, 104, 104, 104.0],
        highs=[104.5, 104.5, 104.5, 104.5, 105.8],  # last candle wicks past 105
        lows=[103.5] * 5,
    )
    sweeps, breakouts = find_sweeps_and_breakouts(df, [level])
    assert len(sweeps) == 1
    assert sweeps[0].side == "high"
    assert sweeps[0].level == pytest.approx(105.0)
    assert breakouts == []


def test_close_beyond_level_is_raw_breakout_not_sweep():
    level = SwingLevel(105.0, "high", "1h", pd.Timestamp("2026-01-01", tz="UTC"))
    df = _flat_df(
        closes=[104, 104, 104, 104, 106.0],   # closes THROUGH the level
        highs=[104.5, 104.5, 104.5, 104.5, 106.5],
        lows=[103.5] * 5,
    )
    sweeps, breakouts = find_sweeps_and_breakouts(df, [level])
    assert sweeps == []
    assert len(breakouts) == 1
    assert breakouts[0].side == "up"


def test_sweep_of_low_flags_long_reversal():
    level = SwingLevel(95.0, "low", "4h", pd.Timestamp("2026-01-01", tz="UTC"))
    df = _flat_df(
        closes=[96, 96, 96, 96, 96.2],
        highs=[96.5] * 5,
        lows=[95.5, 95.5, 95.5, 95.5, 94.4],  # stop-hunt below 95
    )
    sweeps, _ = find_sweeps_and_breakouts(df, [level])
    assert len(sweeps) == 1
    assert sweeps[0].side == "low"


def test_old_breakout_is_not_fresh():
    # Price crossed the level long ago; within the lookback window every
    # candle is already above -> no fresh breakout, no penalty fodder.
    level = SwingLevel(100.0, "high", "1h", pd.Timestamp("2026-01-01", tz="UTC"))
    df = _flat_df(
        closes=[106, 106, 106, 106, 106],
        highs=[106.5] * 5,
        lows=[105.5] * 5,
    )
    sweeps, breakouts = find_sweeps_and_breakouts(df, [level])
    assert sweeps == [] and breakouts == []


# ----------------------------------------------------------------------
# Major swing levels & squeeze
# ----------------------------------------------------------------------
def test_major_swing_levels_from_wave(config):
    df = make_ohlcv(zigzag_closes(400, base=100, amp=20, period=80), "1h")
    out = IndicatorCalculator(config).enrich(df)
    levels = major_swing_levels(out, "1h", order=8, count=3)
    highs = [lv for lv in levels if lv.side == "high"]
    lows = [lv for lv in levels if lv.side == "low"]
    assert 1 <= len(highs) <= 3 and 1 <= len(lows) <= 3
    assert all(lv.price > 110 for lv in highs)
    assert all(lv.price < 90 for lv in lows)


def test_squeeze_fires_on_high_oi_consolidation_at_poi(config):
    df = _flat_df(
        closes=[100.0] * 10,
        highs=[100.4] * 10,
        lows=[99.6] * 10,   # 8-bar range 0.8 < 2x ATR(1.0)
    )
    alert = detect_squeeze(df, oi_z=2.5, poi_centers=[100.2],
                           liquidation_note="short flush", config=config)
    assert alert.active
    assert any("OI 2.5" in r for r in alert.reasons)
    assert any("liquidations" in r for r in alert.reasons)


def test_no_squeeze_when_oi_normal_or_far_from_poi(config):
    df = _flat_df(closes=[100.0] * 10, highs=[100.4] * 10, lows=[99.6] * 10)
    assert not detect_squeeze(df, oi_z=1.0, poi_centers=[100.2],
                              liquidation_note=None, config=config).active
    assert not detect_squeeze(df, oi_z=2.5, poi_centers=[110.0],
                              liquidation_note=None, config=config).active
    assert not detect_squeeze(df, oi_z=None, poi_centers=[100.2],
                              liquidation_note=None, config=config).active
