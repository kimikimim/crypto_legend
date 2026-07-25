"""Shared synthetic OHLCV fixtures. Everything here is deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import EngineConfig


def make_ohlcv(
    closes: np.ndarray,
    freq: str,
    start: str = "2026-01-01",
    spread: float = 0.5,
    volume: float = 1000.0,
) -> pd.DataFrame:
    """Build a clean OHLCV frame around a close series (no outlier wicks)."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    index = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(len(closes), volume),
        },
        index=index,
    )


def zigzag_closes(
    n: int = 400, base: float = 100.0, amp: float = 20.0, period: int = 60
) -> np.ndarray:
    """Smooth deterministic wave: clear swing highs/lows, no randomness."""
    t = np.arange(n)
    return base + amp * np.sin(2 * np.pi * t / period)


@pytest.fixture(autouse=True)
def _no_liquidation_websocket(monkeypatch):
    """Tests must never open the live @forceOrder stream."""
    monkeypatch.setenv("MTF_LIQ_WS", "0")


@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


@pytest.fixture
def base_row() -> dict:
    """A neutral merged-snapshot row; tests override fields per scenario.

    Neutral means: no trend alignment either way is impossible (a value is
    either above or below), so we pick values that favor NEITHER side fully
    and zero out momentum/volume flags.
    """
    return {
        # 15m candle
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
        "volume": 1000.0, "vol_ma": 1000.0, "vol_ratio": 1.0,
        "atr": 1.0, "ema120": 100.0, "ema200": 100.0,
        "rsi": 50.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
        "rsi_rising": False, "rsi_falling": False,
        "rsi_cross_up_recent": False, "rsi_cross_dn_recent": False,
        "macd_cross_up_recent": False, "macd_cross_dn_recent": False,
        "hist_rising": False, "hist_falling": False,
        # 1h context
        "close_1h": 100.0, "ema120_1h": 100.0, "ema200_1h": 100.0,
        "atr_1h": 2.0,
        # 4h context
        "close_4h": 100.0, "ema120_4h": 100.0, "ema200_4h": 100.0,
        "atr_4h": 4.0,
    }
