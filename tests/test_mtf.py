"""Timeframe merging (look-ahead safety) and open-candle exclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data_fetcher import drop_open_candle
from app.mtf import merge_mtf
from tests.conftest import make_ohlcv


def _frame_with_marker(freq: str, periods: int) -> pd.DataFrame:
    df = make_ohlcv(np.linspace(100, 110, periods), freq)
    df["marker"] = np.arange(periods, dtype=float)
    return df


def test_merge_uses_only_closed_higher_tf_candles():
    m15 = _frame_with_marker("15min", 96)   # 2026-01-01 00:00 .. 23:45
    h1 = _frame_with_marker("1h", 24)
    h4 = _frame_with_marker("4h", 6)
    merged = merge_mtf(m15, h1, h4)

    # 15m candle opening 10:30 closes 10:45.
    ts = pd.Timestamp("2026-01-01 10:30", tz="UTC")
    row = merged.loc[ts]
    # Latest 1h candle closed by 10:45 is the one opening 09:00 (marker 9).
    assert row["marker_1h"] == 9.0
    # Latest 4h candle closed by 10:45 is the one opening 04:00 (marker 1);
    # the 08:00 candle is still forming at 10:45 and must NOT leak in.
    assert row["marker_4h"] == 1.0


def test_merge_includes_candle_closing_exactly_at_15m_close():
    m15 = _frame_with_marker("15min", 96)
    h1 = _frame_with_marker("1h", 24)
    h4 = _frame_with_marker("4h", 6)
    merged = merge_mtf(m15, h1, h4)

    # 15m candle opening 11:45 closes 12:00 — exactly when the 11:00 1h
    # candle and the 08:00 4h candle close. Both are usable at that instant.
    row = merged.loc[pd.Timestamp("2026-01-01 11:45", tz="UTC")]
    assert row["marker_1h"] == 11.0
    assert row["marker_4h"] == 2.0


def test_merge_keeps_15m_columns_unsuffixed():
    merged = merge_mtf(
        _frame_with_marker("15min", 96),
        _frame_with_marker("1h", 24),
        _frame_with_marker("4h", 6),
    )
    for col in ("open", "high", "low", "close", "volume"):
        assert col in merged.columns
        assert f"{col}_1h" in merged.columns
        assert f"{col}_4h" in merged.columns


def test_drop_open_candle_removes_forming_candle():
    now = pd.Timestamp("2026-01-01 10:07", tz="UTC")
    df = make_ohlcv(np.linspace(100, 101, 10), "15min",
                    start="2026-01-01 07:45")
    # Last candle opens 10:00 and would close 10:15 > now: it is open.
    assert df.index[-1] == pd.Timestamp("2026-01-01 10:00", tz="UTC")
    out = drop_open_candle(df, "15m", now=now)
    assert len(out) == len(df) - 1
    assert out.index[-1] == pd.Timestamp("2026-01-01 09:45", tz="UTC")


def test_drop_open_candle_keeps_fully_closed_history():
    now = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    df = make_ohlcv(np.linspace(100, 101, 10), "15min",
                    start="2026-01-01 07:45")
    out = drop_open_candle(df, "15m", now=now)
    assert len(out) == len(df)


def test_drop_open_candle_empty_frame_is_noop():
    df = make_ohlcv(np.array([100.0]), "15min").iloc[:0]
    assert drop_open_candle(df, "15m").empty
