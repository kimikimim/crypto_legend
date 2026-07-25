"""Smart Money Concepts: order blocks, fair value gaps, liquidity sweeps,
raw (retail) breakouts, and squeeze detection.

Conventions:
- Swing/structure LEVELS come from wick-filtered pivots (`f_high`/`f_low`)
  so stop-hunt spikes cannot define structure.
- Sweep DETECTION uses raw high/low — the outlier wick IS the sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import DEFAULT_CONFIG, EngineConfig
from app.fibonacci import pivot_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderBlock:
    """Last opposite-colored candle before an impulsive BOS move."""

    low: float
    high: float
    side: str                 # "bullish" (demand) or "bearish" (supply)
    timeframe: str
    time: pd.Timestamp

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class FairValueGap:
    """3-candle imbalance; unmitigated gaps act as magnetic zones."""

    low: float
    high: float
    side: str                 # "bullish" (gap up, support below) or "bearish"
    timeframe: str
    time: pd.Timestamp

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class SwingLevel:
    price: float
    side: str                 # "high" or "low"
    timeframe: str
    time: pd.Timestamp


@dataclass(frozen=True)
class LiquiditySweep:
    """15m wick beyond a major swing level with a close back inside —
    classified strictly as a Whale Sweep (stop-hunt). Mean-reversion signal."""

    level: float
    level_timeframe: str
    side: str                 # "high" (sweep above -> short bias) or "low"
    time: pd.Timestamp


@dataclass(frozen=True)
class RawBreakout:
    """Fresh close beyond a major swing level with NO wick-back: the retail
    breakout pattern the scorer penalizes."""

    level: float
    level_timeframe: str
    side: str                 # "up" or "down"
    time: pd.Timestamp


@dataclass(frozen=True)
class SqueezeAlert:
    active: bool
    oi_z: float | None
    reasons: tuple[str, ...]


# ----------------------------------------------------------------------
# Major swing levels (liquidity pools)
# ----------------------------------------------------------------------
def major_swing_levels(
    df: pd.DataFrame,
    timeframe: str,
    order: int,
    count: int,
) -> list[SwingLevel]:
    """Most recent `count` confirmed swing highs and lows of one timeframe."""
    if len(df) < 2 * order + 1:
        return []
    fh = df["f_high"].to_numpy(dtype=float)
    fl = df["f_low"].to_numpy(dtype=float)
    levels: list[SwingLevel] = []
    for i in pivot_indices(fh, order, "high")[-count:]:
        levels.append(SwingLevel(float(fh[i]), "high", timeframe, df.index[i]))
    for i in pivot_indices(fl, order, "low")[-count:]:
        levels.append(SwingLevel(float(fl[i]), "low", timeframe, df.index[i]))
    return levels


# ----------------------------------------------------------------------
# Fair Value Gaps
# ----------------------------------------------------------------------
def find_fvgs(
    df: pd.DataFrame, timeframe: str, config: EngineConfig = DEFAULT_CONFIG
) -> list[FairValueGap]:
    """Unmitigated 3-candle imbalances within the scan window.

    Bullish FVG at i: low[i] > high[i-2] -> gap (high[i-2], low[i]).
    Mitigated once a later candle fully fills the gap.
    """
    cfg = config
    n = len(df)
    if n < 3:
        return []
    start = max(2, n - cfg.smc_scan_bars)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)

    bullish: list[FairValueGap] = []
    bearish: list[FairValueGap] = []
    for i in range(start, n):
        min_size = cfg.fvg_min_atr_mult * atr[i]
        if not np.isfinite(min_size):
            continue
        if low[i] - high[i - 2] >= min_size:
            if i + 1 >= n or low[i + 1 :].min() > high[i - 2]:
                bullish.append(
                    FairValueGap(high[i - 2], low[i], "bullish", timeframe, df.index[i])
                )
        if low[i - 2] - high[i] >= min_size:
            if i + 1 >= n or high[i + 1 :].max() < low[i - 2]:
                bearish.append(
                    FairValueGap(high[i], low[i - 2], "bearish", timeframe, df.index[i])
                )
    return bullish[-cfg.fvg_max_count :] + bearish[-cfg.fvg_max_count :]


# ----------------------------------------------------------------------
# Order Blocks
# ----------------------------------------------------------------------
def find_order_blocks(
    df: pd.DataFrame, timeframe: str, config: EngineConfig = DEFAULT_CONFIG
) -> list[OrderBlock]:
    """Institutional POIs: the last opposite-colored candle before a
    high-volume impulsive move that broke local structure (BOS).

    Bullish OB: first close above the latest confirmed swing high, where the
    breaking leg contains a candle with range >= ob_impulse_atr_mult * ATR
    AND volume >= ob_vol_mult * volume MA; the OB is the last bearish candle
    within ob_lookback bars before the break. Mitigated (close beyond the far
    edge) OBs are discarded. Mirrored for bearish OBs.
    """
    cfg = config
    order = cfg.structure_order
    n = len(df)
    if n < 2 * order + 2:
        return []

    fh = df["f_high"].to_numpy(dtype=float)
    fl = df["f_low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    vol_ratio = df["vol_ratio"].to_numpy(dtype=float)

    hi_piv = pivot_indices(fh, order, "high")
    lo_piv = pivot_indices(fl, order, "low")

    def structure_level(pivots: np.ndarray, values: np.ndarray, i: int) -> float | None:
        """Latest pivot already confirmed at bar i (needs `order` bars after it)."""
        usable = pivots[pivots + order <= i]
        return float(values[usable[-1]]) if usable.size else None

    def leg_is_impulsive(start: int, end: int) -> bool:
        rng = slice(max(start, 0), end + 1)
        ranges = h[rng] - lo[rng]
        atrs = atr[rng]
        with np.errstate(invalid="ignore"):
            big_range = np.any(ranges >= cfg.ob_impulse_atr_mult * atrs)
            big_volume = np.any(vol_ratio[rng] >= cfg.ob_vol_mult)
        return bool(big_range and big_volume)

    bullish: list[OrderBlock] = []
    bearish: list[OrderBlock] = []
    start = max(2 * order + 1, n - cfg.smc_scan_bars)
    for i in range(start, n):
        swing_high = structure_level(hi_piv, fh, i)
        swing_low = structure_level(lo_piv, fl, i)

        # Bullish BOS: first close above the confirmed swing high.
        if (
            swing_high is not None
            and c[i] > swing_high
            and c[i - 1] <= swing_high
            and leg_is_impulsive(i - cfg.ob_lookback, i)
        ):
            for j in range(i - 1, max(i - cfg.ob_lookback, 0) - 1, -1):
                if c[j] < o[j]:  # last bearish candle before the impulse
                    if c[i + 1 :].min(initial=np.inf) >= lo[j]:  # unmitigated
                        bullish.append(
                            OrderBlock(lo[j], h[j], "bullish", timeframe, df.index[j])
                        )
                    break

        # Bearish BOS: first close below the confirmed swing low.
        if (
            swing_low is not None
            and c[i] < swing_low
            and c[i - 1] >= swing_low
            and leg_is_impulsive(i - cfg.ob_lookback, i)
        ):
            for j in range(i - 1, max(i - cfg.ob_lookback, 0) - 1, -1):
                if c[j] > o[j]:  # last bullish candle before the impulse
                    if c[i + 1 :].max(initial=-np.inf) <= h[j]:
                        bearish.append(
                            OrderBlock(lo[j], h[j], "bearish", timeframe, df.index[j])
                        )
                    break

    return bullish[-cfg.ob_max_count :] + bearish[-cfg.ob_max_count :]


# ----------------------------------------------------------------------
# Liquidity sweeps & raw breakouts
# ----------------------------------------------------------------------
def find_sweeps_and_breakouts(
    m15: pd.DataFrame,
    levels: list[SwingLevel],
    config: EngineConfig = DEFAULT_CONFIG,
) -> tuple[list[LiquiditySweep], list[RawBreakout]]:
    """Classify the last `sweep_lookback` CLOSED 15m candles against major
    4h/1h swing levels.

    Sweep (stop-hunt): raw wick beyond the level, close back inside, and the
    latest close still inside — invalidates breakout logic, mean-reversion.
    Raw breakout: a fresh close beyond the level (first candle to do so) with
    price still beyond — the retail pattern the scorer penalizes.
    """
    cfg = config
    if m15.empty or not levels:
        return [], []
    tail = m15.tail(cfg.sweep_lookback)
    cur_close = float(m15["close"].iloc[-1])

    sweeps: list[LiquiditySweep] = []
    breakouts: list[RawBreakout] = []
    for lv in levels:
        for ts, candle in tail.iterrows():
            pos = m15.index.get_loc(ts)
            prev_close = float(m15["close"].iloc[pos - 1]) if pos > 0 else None
            hi, lo_, cl = float(candle["high"]), float(candle["low"]), float(candle["close"])

            if lv.side == "high":
                if hi > lv.price and cl < lv.price and cur_close < lv.price:
                    sweeps.append(
                        LiquiditySweep(lv.price, lv.timeframe, "high", ts)
                    )
                elif (
                    cl > lv.price
                    and prev_close is not None
                    and prev_close <= lv.price
                    and cur_close > lv.price
                ):
                    breakouts.append(RawBreakout(lv.price, lv.timeframe, "up", ts))
            else:
                if lo_ < lv.price and cl > lv.price and cur_close > lv.price:
                    sweeps.append(
                        LiquiditySweep(lv.price, lv.timeframe, "low", ts)
                    )
                elif (
                    cl < lv.price
                    and prev_close is not None
                    and prev_close >= lv.price
                    and cur_close < lv.price
                ):
                    breakouts.append(RawBreakout(lv.price, lv.timeframe, "down", ts))

    if sweeps:
        logger.info("Whale sweep(s) detected: %s", sweeps)
    return sweeps, breakouts


# ----------------------------------------------------------------------
# Squeeze detection
# ----------------------------------------------------------------------
def detect_squeeze(
    m15: pd.DataFrame,
    oi_z: float | None,
    poi_centers: list[float],
    liquidation_note: str | None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> SqueezeAlert:
    """High-probability squeeze warning: OI unusually high (z >= threshold)
    while price consolidates near a fib/OB confluence POI."""
    cfg = config
    if oi_z is None or len(m15) < cfg.squeeze_consolidation_bars:
        return SqueezeAlert(active=False, oi_z=oi_z, reasons=())

    tail = m15.tail(cfg.squeeze_consolidation_bars)
    atr15 = float(m15["atr"].iloc[-1])
    close = float(m15["close"].iloc[-1])
    if not np.isfinite(atr15) or atr15 <= 0:
        return SqueezeAlert(active=False, oi_z=oi_z, reasons=())

    range_width = float(tail["high"].max() - tail["low"].min())
    consolidating = range_width < cfg.squeeze_range_atr_mult * atr15
    near_poi = any(
        abs(close - center) <= cfg.squeeze_poi_atr_mult * atr15
        for center in poi_centers
    )
    oi_extreme = oi_z >= cfg.oi_z_threshold

    if not (oi_extreme and consolidating and near_poi):
        return SqueezeAlert(active=False, oi_z=oi_z, reasons=())

    reasons = [
        f"OI {oi_z:.1f} std devs above 24h mean",
        f"price consolidating ({cfg.squeeze_consolidation_bars}-bar range "
        f"{range_width:.6g} < {cfg.squeeze_range_atr_mult}x ATR)",
        "price parked at fib/OB confluence POI",
    ]
    if liquidation_note:
        reasons.append(f"validated by liquidations: {liquidation_note}")
    logger.warning("SQUEEZE ALERT: %s", "; ".join(reasons))
    return SqueezeAlert(active=True, oi_z=oi_z, reasons=tuple(reasons))
