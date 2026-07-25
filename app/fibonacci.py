"""Wick-filtered swing detection, Fibonacci levels, and confluence zones."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from app.config import DEFAULT_CONFIG, EngineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SwingLeg:
    """The most recent completed swing (low -> high or high -> low)."""

    low: float
    high: float
    direction: str  # "up" (retracement acts as support) or "down"
    low_time: pd.Timestamp
    high_time: pd.Timestamp


@dataclass(frozen=True)
class FibLevel:
    price: float
    ratio: float
    timeframe: str


@dataclass(frozen=True)
class ConfluenceZone:
    """Price band where a 4h and a 1h fib level overlap within tolerance."""

    low: float
    high: float
    ratios: tuple[float, ...]       # every fib ratio contributing to the zone
    sources: tuple[str, ...]        # e.g. ("4h:0.618", "1h:0.5")

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2.0


def pivot_indices(arr: np.ndarray, order: int, kind: str) -> np.ndarray:
    """Pivot detection tolerant of equal-value plateaus (double tops etc.).

    argrelextrema with >= / <= accepts plateaus, but on a perfectly flat
    stretch it would mark every bar; require strict dominance over at least
    one adjacent bar to reject those.
    """
    if kind == "high":
        candidates = argrelextrema(arr, np.greater_equal, order=order)[0]
        strict = np.greater
    else:
        candidates = argrelextrema(arr, np.less_equal, order=order)[0]
        strict = np.less
    n = len(arr)
    keep = [
        i
        for i in candidates
        # argrelextrema pads edges (mode='clip'), which would confirm pivots
        # that lack `order` real bars on both sides — exactly the repainting
        # pivots we must reject.
        if order <= i < n - order
        and (strict(arr[i], arr[i - 1]) or strict(arr[i], arr[i + 1]))
    ]
    return np.asarray(keep, dtype=int)


def find_swing_leg(df: pd.DataFrame, order: int) -> SwingLeg | None:
    """Locate the latest confirmed swing leg on wick-filtered extremes.

    Uses `f_high`/`f_low` (outlier wicks already replaced by body extremes)
    so a single stop-hunt spike cannot anchor the Fibonacci grid. A pivot
    must dominate `order` bars on each side, so the last `order` bars can
    never form an unconfirmed (repainting) pivot.
    """
    if len(df) < 2 * order + 1:
        return None
    fh = df["f_high"].to_numpy(dtype=float)
    fl = df["f_low"].to_numpy(dtype=float)
    hi_idx = pivot_indices(fh, order, "high")
    lo_idx = pivot_indices(fl, order, "low")
    if hi_idx.size == 0 or lo_idx.size == 0:
        return None

    last_hi, last_lo = int(hi_idx[-1]), int(lo_idx[-1])
    high, low = float(fh[last_hi]), float(fl[last_lo])
    if high <= low:
        return None
    direction = "up" if last_hi > last_lo else "down"
    return SwingLeg(
        low=low,
        high=high,
        direction=direction,
        low_time=df.index[last_lo],
        high_time=df.index[last_hi],
    )


def fib_levels(
    leg: SwingLeg | None, timeframe: str, config: EngineConfig = DEFAULT_CONFIG
) -> list[FibLevel]:
    """Retracement levels for a swing leg.

    Up leg: measured down from the high (potential supports).
    Down leg: measured up from the low (potential resistances).
    """
    if leg is None:
        return []
    span = leg.high - leg.low
    levels = []
    for ratio in config.fib_ratios:
        if leg.direction == "up":
            price = leg.high - span * ratio
        else:
            price = leg.low + span * ratio
        levels.append(FibLevel(price=price, ratio=ratio, timeframe=timeframe))
    return levels


def find_confluence_zones(
    levels_a: list[FibLevel],
    levels_b: list[FibLevel],
    tolerance: float,
) -> list[ConfluenceZone]:
    """Pair up levels from two timeframes that sit within `tolerance` of
    each other, then merge overlapping pairs into distinct zones."""
    if tolerance <= 0 or not levels_a or not levels_b:
        return []

    raw: list[ConfluenceZone] = []
    for a in levels_a:
        for b in levels_b:
            if abs(a.price - b.price) <= tolerance:
                lo, hi = sorted((a.price, b.price))
                raw.append(
                    ConfluenceZone(
                        low=lo,
                        high=hi,
                        ratios=(a.ratio, b.ratio),
                        sources=(
                            f"{a.timeframe}:{a.ratio}",
                            f"{b.timeframe}:{b.ratio}",
                        ),
                    )
                )
    if not raw:
        return []

    raw.sort(key=lambda z: z.low)
    merged = [raw[0]]
    for zone in raw[1:]:
        cur = merged[-1]
        if zone.low <= cur.high:
            merged[-1] = ConfluenceZone(
                low=cur.low,
                high=max(cur.high, zone.high),
                ratios=tuple(sorted(set(cur.ratios + zone.ratios))),
                sources=tuple(sorted(set(cur.sources + zone.sources))),
            )
        else:
            merged.append(zone)
    logger.debug("Found %d confluence zones (tol=%.6g)", len(merged), tolerance)
    return merged
