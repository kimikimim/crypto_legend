"""Liquidation tracker (measured spikes), warm gating, and cold-start proxy."""

from __future__ import annotations

import pandas as pd
import pytest

from app.liquidations import (
    NO_SIGNAL,
    LiquidationTracker,
    proxy_liquidation_signal,
)

NOW = pd.Timestamp("2026-06-01 12:07", tz="UTC")
SYMBOL = "BTC/USDT:USDT"


def warmed_tracker(config) -> LiquidationTracker:
    tracker = LiquidationTracker(config)
    tracker.mark_started(NOW - pd.Timedelta(hours=23))
    # Steady background noise: 1000 USDT per side every 15 minutes for 20h.
    t = NOW - pd.Timedelta(hours=20)
    while t < NOW - pd.Timedelta(minutes=45):
        tracker.record(SYMBOL, t, "long", 1000.0)
        tracker.record(SYMBOL, t, "short", 1000.0)
        t += pd.Timedelta(minutes=15)
    return tracker


def test_not_warm_returns_none(config):
    tracker = LiquidationTracker(config)
    tracker.mark_started(NOW - pd.Timedelta(minutes=30))  # too recent
    tracker.record(SYMBOL, NOW, "long", 1_000_000.0)
    assert tracker.signal(SYMBOL, now=NOW) is None


def test_never_started_is_not_warm(config):
    assert LiquidationTracker(config).signal(SYMBOL, now=NOW) is None


def test_long_flush_spike_detected(config):
    tracker = warmed_tracker(config)
    tracker.record(SYMBOL, NOW - pd.Timedelta(minutes=5), "long", 500_000.0)
    signal = tracker.signal(SYMBOL, now=NOW)
    assert signal is not None and signal.source == "measured"
    assert signal.long_flush is True
    assert signal.short_flush is False
    assert any("long-side" in d for d in signal.detail)


def test_quiet_market_has_no_spike(config):
    tracker = warmed_tracker(config)
    signal = tracker.signal(SYMBOL, now=NOW)
    assert signal is not None
    assert not signal.long_flush and not signal.short_flush


def test_non_whitelisted_symbol_ignored(config):
    tracker = warmed_tracker(config)
    tracker.record("XRP/USDT:USDT", NOW, "long", 9e9)
    assert tracker.signal("XRP/USDT:USDT", now=NOW) is None


# ----------------------------------------------------------------------
# Cold-start proxy
# ----------------------------------------------------------------------
def test_proxy_red_cascade_flags_long_flush(config):
    row = {"open": 101.0, "close": 99.0, "vol_ratio": 2.5}
    signal = proxy_liquidation_signal(row, oi_change_z=-2.5, config=config)
    assert signal.long_flush and not signal.short_flush
    assert signal.source == "proxy"


def test_proxy_green_cascade_flags_short_flush(config):
    row = {"open": 99.0, "close": 101.0, "vol_ratio": 3.0}
    signal = proxy_liquidation_signal(row, oi_change_z=-3.0, config=config)
    assert signal.short_flush and not signal.long_flush


@pytest.mark.parametrize(
    "row,oi_z",
    [
        ({"open": 101.0, "close": 99.0, "vol_ratio": 2.5}, -1.0),   # OI drop mild
        ({"open": 101.0, "close": 99.0, "vol_ratio": 1.2}, -3.0),   # no volume
        ({"open": 101.0, "close": 99.0, "vol_ratio": 2.5}, None),   # no OI data
        ({"open": float("nan"), "close": 99.0, "vol_ratio": 2.5}, -3.0),
    ],
)
def test_proxy_requires_full_cascade_evidence(config, row, oi_z):
    assert proxy_liquidation_signal(row, oi_change_z=oi_z, config=config) == NO_SIGNAL
