"""Deterministic 0-100 MTF scoring for long and short entries.

Revised smart-money weights:
    Trend & MTF alignment        20  (4h: 10, 1h: 7, 15m: 3)
    Location & Confluence        25  (fib zone 10 | single fib 5, OB 7, FVG 5, reaction 3; capped)
    Whale & Liquidity Validation 25  (confirmed sweep 15, liquidation flush 10)
    Momentum                     15  (RSI 8/4, MACD 7/3)
    Volatility & Volume          15  (spike at S/R 15, soft 8, away 5)

Fresh retail breakouts (close beyond a major swing level with no wick-back)
subtract `breakout_penalty` from the breakout direction's total (floored 0).

Scoring is a pure function of one merged snapshot row plus a pre-computed
ScoringContext — same inputs always yield the same score.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.config import DEFAULT_CONFIG, EngineConfig
from app.fibonacci import ConfluenceZone, FibLevel
from app.liquidations import NO_SIGNAL, LiquidationSignal
from app.smc import FairValueGap, LiquiditySweep, OrderBlock, RawBreakout

logger = logging.getLogger(__name__)

TREND_MAX = 20.0
LOCATION_MAX = 25.0
WHALE_MAX = 25.0
MOMENTUM_MAX = 15.0
VOLATILITY_MAX = 15.0


@dataclass(frozen=True)
class ScoringContext:
    """Everything beyond the snapshot row that scoring depends on."""

    zones: list[ConfluenceZone] = field(default_factory=list)
    levels_4h: list[FibLevel] = field(default_factory=list)
    levels_1h: list[FibLevel] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fvgs: list[FairValueGap] = field(default_factory=list)
    sweeps: list[LiquiditySweep] = field(default_factory=list)
    breakouts: list[RawBreakout] = field(default_factory=list)
    liquidations: LiquidationSignal = NO_SIGNAL
    regime: str = "chop"          # 1D macro regime: "bull" / "bear" / "chop"


@dataclass(frozen=True)
class CategoryScore:
    points: float
    max_points: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DirectionScore:
    total: float
    trend: CategoryScore
    location: CategoryScore
    whale: CategoryScore
    momentum: CategoryScore
    volatility: CategoryScore
    penalty: float
    penalty_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScoreResult:
    long: DirectionScore
    short: DirectionScore


def _num(row: Mapping[str, Any], key: str) -> float:
    """Fetch a numeric field; missing/NaN becomes NaN (all comparisons False)."""
    val = row.get(key) if hasattr(row, "get") else row[key]
    try:
        val = float(val)
    except (TypeError, ValueError):
        return math.nan
    return val


def _flag(row: Mapping[str, Any], key: str) -> bool:
    val = row.get(key) if hasattr(row, "get") else row[key]
    return bool(val) and val == val  # NaN-safe


class ScoringEngine:
    """Evaluates the current 15m candle against 1h/4h + smart-money context."""

    def __init__(self, config: EngineConfig = DEFAULT_CONFIG) -> None:
        self.cfg = config

    def score(self, row: Mapping[str, Any], ctx: ScoringContext) -> ScoreResult:
        """`row` is one merged snapshot: 15m columns unsuffixed, higher
        timeframe columns suffixed `_1h` / `_4h`."""
        long_score = self._score_direction(row, ctx, "long")
        short_score = self._score_direction(row, ctx, "short")
        logger.info(
            "Scored long=%.1f short=%.1f", long_score.total, short_score.total
        )
        return ScoreResult(long=long_score, short=short_score)

    def _score_direction(
        self, row: Mapping[str, Any], ctx: ScoringContext, side: str
    ) -> DirectionScore:
        trend = self._score_trend(row, side)
        location, at_sr = self._score_location(row, ctx, side)
        whale = self._score_whale(ctx, side)
        momentum = self._score_momentum(row, side)
        volatility = self._score_volatility(row, at_sr)
        penalty, penalty_reasons = self._breakout_penalty(ctx, side)
        raw_total = (
            trend.points + location.points + whale.points
            + momentum.points + volatility.points - penalty
        )
        return DirectionScore(
            total=round(min(max(raw_total, 0.0), 100.0), 1),
            trend=trend,
            location=location,
            whale=whale,
            momentum=momentum,
            volatility=volatility,
            penalty=penalty,
            penalty_reasons=penalty_reasons,
        )

    # ------------------------------------------------------------------
    # Trend (20): 4h alignment 10, 1h alignment 7, 15m bias 3
    # ------------------------------------------------------------------
    def _score_trend(self, row: Mapping[str, Any], side: str) -> CategoryScore:
        is_long = side == "long"
        pts = 0.0
        reasons: list[str] = []

        def aligned(a: float, b: float) -> bool:
            return a > b if is_long else a < b

        if aligned(_num(row, "ema120_4h"), _num(row, "ema200_4h")):
            pts += 6.0
            reasons.append("4h EMA120/200 aligned")
        if aligned(_num(row, "close_4h"), _num(row, "ema120_4h")):
            pts += 4.0
            reasons.append("4h price beyond EMA120")
        if aligned(_num(row, "ema120_1h"), _num(row, "ema200_1h")):
            pts += 4.0
            reasons.append("1h EMA120/200 aligned")
        if aligned(_num(row, "close_1h"), _num(row, "ema120_1h")):
            pts += 3.0
            reasons.append("1h price beyond EMA120")
        if aligned(_num(row, "close"), _num(row, "ema120")):
            pts += 3.0
            reasons.append("15m price beyond EMA120")
        return CategoryScore(pts, TREND_MAX, tuple(reasons))

    # ------------------------------------------------------------------
    # Location & Confluence (25): fib confluence zone 10 (or single fib 5),
    # order block 7, FVG 5, reaction candle 3 — capped at 25.
    # ------------------------------------------------------------------
    def _score_location(
        self, row: Mapping[str, Any], ctx: ScoringContext, side: str
    ) -> tuple[CategoryScore, bool]:
        is_long = side == "long"
        close = _num(row, "close")
        low = _num(row, "low")
        high = _num(row, "high")
        atr15 = _num(row, "atr")
        pts = 0.0
        reasons: list[str] = []

        if math.isnan(close) or math.isnan(atr15) or atr15 <= 0:
            return CategoryScore(0.0, LOCATION_MAX, ()), False

        tol = self.cfg.touch_atr_mult * atr15

        def band_touched(b_low: float, b_high: float, b_center: float) -> bool:
            if is_long:
                # Demand: band at/below price, candle low reached into it.
                return b_center <= close and low <= b_high + tol and close >= b_low
            return b_center >= close and high >= b_low - tol and close <= b_high

        def band_reacted(b_low: float, b_high: float) -> bool:
            if is_long:
                return low <= b_high and close > b_high
            return high >= b_low and close < b_low

        touched_bands: list[tuple[float, float]] = []

        # Fib confluence zone (or single-TF fib level fallback).
        touched_zones = [
            z for z in ctx.zones if band_touched(z.low, z.high, z.center)
        ]
        if touched_zones:
            pts += 10.0
            best = touched_zones[0]
            touched_bands.extend((z.low, z.high) for z in touched_zones)
            reasons.append(
                f"price at 4h/1h fib confluence zone "
                f"[{best.low:.6g}, {best.high:.6g}] ({', '.join(best.sources)})"
            )
            if any(
                r in self.cfg.golden_ratios for z in touched_zones for r in z.ratios
            ):
                pts += 2.0
                reasons.append("zone includes golden-pocket ratio (0.5/0.618)")
        else:
            levels = ctx.levels_4h + ctx.levels_1h
            touched_levels = [
                lv
                for lv in levels
                if band_touched(lv.price, lv.price, lv.price)
            ]
            if touched_levels:
                lv = touched_levels[0]
                pts += 5.0
                touched_bands.extend((lv.price, lv.price) for lv in touched_levels)
                reasons.append(
                    f"price at single-TF fib {lv.timeframe}:{lv.ratio} ({lv.price:.6g})"
                )

        # Order block POI.
        ob_side = "bullish" if is_long else "bearish"
        touched_obs = [
            ob
            for ob in ctx.order_blocks
            if ob.side == ob_side and band_touched(ob.low, ob.high, ob.center)
        ]
        if touched_obs:
            ob = touched_obs[0]
            pts += 7.0
            touched_bands.extend((ob.low, ob.high) for ob in touched_obs)
            reasons.append(
                f"price at {ob.side} {ob.timeframe} order block "
                f"[{ob.low:.6g}, {ob.high:.6g}]"
            )

        # Fair value gap POI.
        fvg_side = "bullish" if is_long else "bearish"
        touched_fvgs = [
            g
            for g in ctx.fvgs
            if g.side == fvg_side and band_touched(g.low, g.high, g.center)
        ]
        if touched_fvgs:
            g = touched_fvgs[0]
            pts += 5.0
            touched_bands.extend((g.low, g.high) for g in touched_fvgs)
            reasons.append(
                f"price rebalancing {g.side} {g.timeframe} FVG "
                f"[{g.low:.6g}, {g.high:.6g}]"
            )

        if any(band_reacted(b_low, b_high) for b_low, b_high in touched_bands):
            pts += 3.0
            reasons.append("reaction candle closed beyond the POI")

        at_sr = bool(touched_bands)
        return CategoryScore(min(pts, LOCATION_MAX), LOCATION_MAX, tuple(reasons)), at_sr

    # ------------------------------------------------------------------
    # Whale & Liquidity Validation (25): confirmed sweep 15, liq flush 10
    # ------------------------------------------------------------------
    def _score_whale(self, ctx: ScoringContext, side: str) -> CategoryScore:
        is_long = side == "long"
        pts = 0.0
        reasons: list[str] = []

        # A sweep of the LOWS is a stop-hunt below support -> long reversal;
        # a sweep of the HIGHS -> short reversal.
        wanted = "low" if is_long else "high"
        sweeps = [s for s in ctx.sweeps if s.side == wanted]
        if sweeps:
            s = sweeps[-1]
            pts += 15.0
            reasons.append(
                f"whale sweep of {s.level_timeframe} swing {s.side} "
                f"({s.level:.6g}) with close back inside — mean-reversion "
                f"{'long' if is_long else 'short'}"
            )

        liq = ctx.liquidations
        flushed = liq.long_flush if is_long else liq.short_flush
        if flushed:
            pts += 10.0
            flushed_side = "long" if is_long else "short"
            reasons.append(
                f"{flushed_side} liquidation flush confirms entry "
                f"[{liq.source}] " + "; ".join(liq.detail)
            )
        return CategoryScore(pts, WHALE_MAX, tuple(reasons))

    # ------------------------------------------------------------------
    # Momentum (15): RSI 8 (cross) / 4 (slope), MACD 7 (cross) / 3 (slope)
    # ------------------------------------------------------------------
    def _score_momentum(self, row: Mapping[str, Any], side: str) -> CategoryScore:
        is_long = side == "long"
        pts = 0.0
        reasons: list[str] = []
        rsi_val = _num(row, "rsi")

        if is_long:
            if _flag(row, "rsi_cross_up_recent"):
                pts += 8.0
                reasons.append("RSI reclaimed oversold (30) recently")
            elif 40.0 <= rsi_val <= 70.0 and _flag(row, "rsi_rising"):
                pts += 4.0
                reasons.append("RSI rising in bullish range")
            if _flag(row, "macd_cross_up_recent"):
                pts += 7.0
                reasons.append("MACD bullish cross recently")
            elif _num(row, "macd_hist") > 0 and _flag(row, "hist_rising"):
                pts += 3.0
                reasons.append("MACD histogram positive and rising")
        else:
            if _flag(row, "rsi_cross_dn_recent"):
                pts += 8.0
                reasons.append("RSI rejected overbought (70) recently")
            elif 30.0 <= rsi_val <= 60.0 and _flag(row, "rsi_falling"):
                pts += 4.0
                reasons.append("RSI falling in bearish range")
            if _flag(row, "macd_cross_dn_recent"):
                pts += 7.0
                reasons.append("MACD bearish cross recently")
            elif _num(row, "macd_hist") < 0 and _flag(row, "hist_falling"):
                pts += 3.0
                reasons.append("MACD histogram negative and falling")
        return CategoryScore(pts, MOMENTUM_MAX, tuple(reasons))

    # ------------------------------------------------------------------
    # Volatility & Volume Anomalies (15)
    # ------------------------------------------------------------------
    def _score_volatility(
        self, row: Mapping[str, Any], at_sr: bool
    ) -> CategoryScore:
        ratio = _num(row, "vol_ratio")
        if math.isnan(ratio):
            return CategoryScore(0.0, VOLATILITY_MAX, ())
        if ratio >= self.cfg.vol_spike_mult and at_sr:
            return CategoryScore(
                15.0, VOLATILITY_MAX,
                (f"volume spike {ratio:.2f}x MA at S/R zone",),
            )
        if ratio >= self.cfg.vol_soft_mult and at_sr:
            return CategoryScore(
                8.0, VOLATILITY_MAX,
                (f"elevated volume {ratio:.2f}x MA at S/R zone",),
            )
        if ratio >= self.cfg.vol_spike_mult:
            return CategoryScore(
                5.0, VOLATILITY_MAX,
                (f"volume spike {ratio:.2f}x MA away from S/R",),
            )
        return CategoryScore(0.0, VOLATILITY_MAX, ())

    # ------------------------------------------------------------------
    # Penalties: retail breakout chase + counter-macro-regime entries
    # ------------------------------------------------------------------
    def _breakout_penalty(
        self, ctx: ScoringContext, side: str
    ) -> tuple[float, tuple[str, ...]]:
        penalty = 0.0
        reasons: list[str] = []

        wanted = "up" if side == "long" else "down"
        hits = [b for b in ctx.breakouts if b.side == wanted]
        if hits:
            b = hits[-1]
            penalty += self.cfg.breakout_penalty
            reasons.append(
                f"fresh raw breakout {b.side} through {b.level_timeframe} swing "
                f"({b.level:.6g}) without sweep confirmation — retail chase penalized"
            )

        counter_regime = (
            (side == "long" and ctx.regime == "bear")
            or (side == "short" and ctx.regime == "bull")
        )
        if counter_regime:
            penalty += self.cfg.regime_penalty
            reasons.append(
                f"{side} signal against the 1D {ctx.regime} macro regime "
                f"— bull/bear trap risk penalized"
            )
        return penalty, tuple(reasons)
