"""Indicator math and wick-filter behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators import IndicatorCalculator, ema, rsi
from tests.conftest import make_ohlcv, zigzag_closes


def test_ema_matches_pandas_ewm():
    s = pd.Series(np.linspace(100, 200, 300))
    expected = s.ewm(span=120, adjust=False).mean()
    pd.testing.assert_series_equal(ema(s, 120), expected)


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 100))
    down = pd.Series(np.linspace(200, 100, 100))
    rsi_up = rsi(up, 14).dropna()
    rsi_down = rsi(down, 14).dropna()
    assert ((rsi_up >= 0) & (rsi_up <= 100)).all()
    assert rsi_up.iloc[-1] > 90  # monotonic rise -> pegged high
    assert rsi_down.iloc[-1] < 10


def test_rsi_flat_series_is_neutral():
    flat = pd.Series(np.full(60, 100.0))
    assert rsi(flat, 14).dropna().iloc[-1] == 50.0


def test_enrich_adds_all_scoring_columns(config):
    df = make_ohlcv(zigzag_closes(300), "15min")
    out = IndicatorCalculator(config).enrich(df)
    expected = {
        "ema120", "ema200", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "atr", "vol_ma", "vol_ratio",
        "f_high", "f_low", "rsi_rising", "rsi_falling",
        "rsi_cross_up_recent", "rsi_cross_dn_recent",
        "macd_cross_up_recent", "macd_cross_dn_recent",
        "hist_rising", "hist_falling",
    }
    assert expected.issubset(out.columns)
    assert not out["atr"].iloc[-1] != out["atr"].iloc[-1]  # not NaN
    assert (out["atr"].dropna() > 0).all()


def test_wick_filter_replaces_outlier_wick_with_body_extreme(config):
    df = make_ohlcv(zigzag_closes(100, amp=2.0, period=25), "15min")
    # Inject a stop-hunt candle: huge upper and lower wicks on the last bar.
    i = df.index[-1]
    body_top = max(df.at[i, "open"], df.at[i, "close"])
    body_bot = min(df.at[i, "open"], df.at[i, "close"])
    df.at[i, "high"] = body_top + 500.0   # vastly beyond 3x ATR
    df.at[i, "low"] = body_bot - 500.0

    out = IndicatorCalculator(config).enrich(df)
    assert out.at[i, "f_high"] == body_top
    assert out.at[i, "f_low"] == body_bot
    # A normal candle keeps its raw extremes.
    j = df.index[-2]
    assert out.at[j, "f_high"] == df.at[j, "high"]
    assert out.at[j, "f_low"] == df.at[j, "low"]


def test_macd_cross_flag_stays_recent_for_lookback_bars(config):
    # Down leg then up leg -> a bullish MACD cross happens on the way up.
    closes = np.concatenate([np.linspace(120, 80, 150), np.linspace(80, 130, 150)])
    df = make_ohlcv(closes, "15min")
    out = IndicatorCalculator(config).enrich(df)
    cross_rows = out.index[
        (out["macd"] > out["macd_signal"])
        & (out["macd"].shift(1) <= out["macd_signal"].shift(1))
    ]
    assert len(cross_rows) > 0
    pos = out.index.get_loc(cross_rows[-1])
    window = out.iloc[pos : pos + config.cross_lookback]
    assert window["macd_cross_up_recent"].all()
