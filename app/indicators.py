"""Indicator computation: EMA, RSI, MACD, Bollinger, ATR, volume, wick filter.

All indicators are implemented directly with pandas/numpy (Wilder smoothing
where conventional) so results are deterministic and dependency-light.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.config import DEFAULT_CONFIG, EngineConfig

logger = logging.getLogger(__name__)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Zero average loss -> RSI pegged at 100; totally flat -> neutral 50.
    out = out.mask(avg_loss.eq(0.0) & avg_gain.gt(0.0), 100.0)
    out = out.mask(avg_loss.eq(0.0) & avg_gain.eq(0.0), 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


class IndicatorCalculator:
    """Enriches a raw OHLCV frame with every column the scorer needs."""

    def __init__(self, config: EngineConfig = DEFAULT_CONFIG) -> None:
        self.cfg = config

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        if len(df) < cfg.ema_slow:
            logger.warning(
                "Only %d candles (< ema_slow=%d): slow EMA will be unstable",
                len(df), cfg.ema_slow,
            )
        out = df.copy()
        close = out["close"]

        out["ema120"] = ema(close, cfg.ema_fast)
        out["ema200"] = ema(close, cfg.ema_slow)
        out["rsi"] = rsi(close, cfg.rsi_period)
        out["macd"], out["macd_signal"], out["macd_hist"] = macd(
            close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(
            close, cfg.bb_period, cfg.bb_std
        )
        out["atr"] = atr(out, cfg.atr_period)
        out["vol_ma"] = out["volume"].rolling(cfg.vol_ma_period).mean()
        out["vol_ratio"] = out["volume"] / out["vol_ma"]

        self._add_wick_filter(out)
        self._add_momentum_flags(out)
        return out

    def _add_wick_filter(self, out: pd.DataFrame) -> None:
        """Replace outlier wicks (> wick_atr_mult * ATR) with body extremes.

        NaN ATR rows (warm-up) keep the raw high/low: `wick > NaN` is False.
        """
        body_top = out[["open", "close"]].max(axis=1)
        body_bot = out[["open", "close"]].min(axis=1)
        max_wick = self.cfg.wick_atr_mult * out["atr"]
        upper_wick = out["high"] - body_top
        lower_wick = body_bot - out["low"]
        out["f_high"] = np.where(upper_wick > max_wick, body_top, out["high"])
        out["f_low"] = np.where(lower_wick > max_wick, body_bot, out["low"])

    def _add_momentum_flags(self, out: pd.DataFrame) -> None:
        """Precompute cross/slope flags so scoring is a pure row function."""
        cfg = self.cfg
        r, prev_r = out["rsi"], out["rsi"].shift(1)
        out["rsi_rising"] = (r > prev_r).fillna(False)
        out["rsi_falling"] = (r < prev_r).fillna(False)

        cross_up = (r > cfg.rsi_oversold) & (prev_r <= cfg.rsi_oversold)
        cross_dn = (r < cfg.rsi_overbought) & (prev_r >= cfg.rsi_overbought)
        lb = cfg.cross_lookback
        out["rsi_cross_up_recent"] = (
            cross_up.rolling(lb, min_periods=1).max().fillna(0).astype(bool)
        )
        out["rsi_cross_dn_recent"] = (
            cross_dn.rolling(lb, min_periods=1).max().fillna(0).astype(bool)
        )

        m, s = out["macd"], out["macd_signal"]
        macd_up = (m > s) & (m.shift(1) <= s.shift(1))
        macd_dn = (m < s) & (m.shift(1) >= s.shift(1))
        out["macd_cross_up_recent"] = (
            macd_up.rolling(lb, min_periods=1).max().fillna(0).astype(bool)
        )
        out["macd_cross_dn_recent"] = (
            macd_dn.rolling(lb, min_periods=1).max().fillna(0).astype(bool)
        )
        h, prev_h = out["macd_hist"], out["macd_hist"].shift(1)
        out["hist_rising"] = (h > prev_h).fillna(False)
        out["hist_falling"] = (h < prev_h).fillna(False)
