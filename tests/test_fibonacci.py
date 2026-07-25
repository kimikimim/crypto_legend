"""Swing detection, fib arithmetic, and confluence zones."""

from __future__ import annotations

import numpy as np
import pytest

from app.fibonacci import (
    FibLevel,
    SwingLeg,
    fib_levels,
    find_confluence_zones,
    find_swing_leg,
)
from app.indicators import IndicatorCalculator
from tests.conftest import make_ohlcv, zigzag_closes


@pytest.fixture
def enriched_wave(config):
    df = make_ohlcv(zigzag_closes(400, base=100, amp=20, period=80), "1h")
    return IndicatorCalculator(config).enrich(df)


def test_find_swing_leg_detects_wave_extremes(enriched_wave):
    leg = find_swing_leg(enriched_wave, order=8)
    assert leg is not None
    # Wave oscillates 80..120 (+/- spread); detected swings must be near those.
    assert leg.high == pytest.approx(120.0, abs=2.0)
    assert leg.low == pytest.approx(80.0, abs=2.0)


def test_swing_leg_ignores_outlier_wick(config):
    df = make_ohlcv(zigzag_closes(400, base=100, amp=20, period=80), "1h")
    # Stop-hunt spike in the middle of the series, far above any real high.
    i = df.index[200]
    df.at[i, "high"] = 1000.0
    out = IndicatorCalculator(config).enrich(df)
    leg = find_swing_leg(out, order=8)
    assert leg is not None
    assert leg.high < 130.0  # the fake 1000.0 wick did not anchor the leg


def test_fib_levels_up_leg_arithmetic(config):
    leg = SwingLeg(low=100.0, high=200.0, direction="up",
                   low_time=None, high_time=None)
    levels = {lv.ratio: lv.price for lv in fib_levels(leg, "4h", config)}
    assert levels[0.0] == pytest.approx(200.0)
    assert levels[0.5] == pytest.approx(150.0)
    assert levels[0.618] == pytest.approx(138.2)
    assert levels[1.0] == pytest.approx(100.0)


def test_fib_levels_down_leg_arithmetic(config):
    leg = SwingLeg(low=100.0, high=200.0, direction="down",
                   low_time=None, high_time=None)
    levels = {lv.ratio: lv.price for lv in fib_levels(leg, "1h", config)}
    assert levels[0.0] == pytest.approx(100.0)
    assert levels[0.618] == pytest.approx(161.8)
    assert levels[1.0] == pytest.approx(200.0)


def test_confluence_zone_within_tolerance():
    a = [FibLevel(price=150.0, ratio=0.5, timeframe="4h")]
    b = [FibLevel(price=150.4, ratio=0.618, timeframe="1h")]
    zones = find_confluence_zones(a, b, tolerance=0.5)
    assert len(zones) == 1
    z = zones[0]
    assert z.low == pytest.approx(150.0)
    assert z.high == pytest.approx(150.4)
    assert set(z.ratios) == {0.5, 0.618}


def test_no_confluence_outside_tolerance():
    a = [FibLevel(price=150.0, ratio=0.5, timeframe="4h")]
    b = [FibLevel(price=151.0, ratio=0.618, timeframe="1h")]
    assert find_confluence_zones(a, b, tolerance=0.5) == []


def test_overlapping_pairs_merge_into_one_zone():
    a = [
        FibLevel(price=150.0, ratio=0.5, timeframe="4h"),
        FibLevel(price=150.3, ratio=0.618, timeframe="4h"),
    ]
    b = [FibLevel(price=150.2, ratio=0.5, timeframe="1h")]
    zones = find_confluence_zones(a, b, tolerance=0.5)
    assert len(zones) == 1
    assert zones[0].low == pytest.approx(150.0)
    assert zones[0].high == pytest.approx(150.3)


def test_empty_inputs_produce_no_zones():
    assert find_confluence_zones([], [], tolerance=1.0) == []
    assert fib_levels(None, "4h") == []
    flat = make_ohlcv(np.full(50, 100.0), "1h")
    flat["f_high"], flat["f_low"] = flat["high"], flat["low"]
    assert find_swing_leg(flat, order=8) is None
