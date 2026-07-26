"""Market regime filter (1D) and structural trade planning (SL/TP/sizing).

SL placement is structural invalidation, never a percentage:
    sweep wick extreme -> POI band edge -> nearest major swing -> ATR fallback,
always buffered by sl_atr_buffer * ATR(15m).

TP1 targets the closest opposing Order Block or fib level; TP2 targets the
fib extension of the active 1h swing leg or the next major structural swing.

Position sizing keeps account risk constant: the wider the stop, the smaller
the suggested position.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from app.config import DEFAULT_CONFIG, TF_DELTA, EngineConfig
from app.fibonacci import ConfluenceZone, FibLevel, SwingLeg
from app.indicators import ema
from app.smc import FairValueGap, LiquiditySweep, OrderBlock, SwingLevel

logger = logging.getLogger(__name__)

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_CHOP = "chop"


@dataclass(frozen=True)
class MarketState:
    """Execution conditions at the moment of entry.

    Slippage depends on these, so they travel with the plan: the backtest
    must charge the same cost the live engine quoted, or the two diverge.
    """

    price: float
    atr: float
    vol_ratio: float = 1.0
    stressed: bool = False   # sweep in progress / volume spike: thin book


def slippage_pct(state: MarketState, config: EngineConfig = DEFAULT_CONFIG) -> float:
    """Per-side slippage estimate, in percent of notional.

    A flat assumption is wrong in both directions: too harsh on a quiet
    range, far too kind during the liquidation cascades this engine is built
    to trade. The volatility term scales with ATR as a share of price, and
    the stress multiplier fires when a sweep or volume spike says the book is
    being cleared out.
    """
    if state.price <= 0 or not math.isfinite(state.atr) or state.atr <= 0:
        return config.slippage_base_pct
    atr_pct = state.atr / state.price * 100.0
    slip = config.slippage_base_pct + config.slippage_atr_coef * atr_pct
    if state.stressed:
        slip *= config.slippage_stress_mult
    return min(slip, config.slippage_cap_pct)


def round_trip_cost_pct(
    state: MarketState, config: EngineConfig = DEFAULT_CONFIG
) -> float:
    """Entry plus exit: venue fees (fixed) and slippage (conditions-dependent)."""
    return 2.0 * (config.taker_fee_pct + slippage_pct(state, config))


@dataclass(frozen=True)
class TradePlan:
    side: str                     # "long" or "short"
    entry_zone_low: float
    entry_zone_high: float
    suggested_sl: float
    suggested_tp1: float
    suggested_tp2: float
    risk_weight: float            # 0-1: fraction of the max allowed size
    suggested_leverage: float     # position notional / equity, capped
    rr_tp1: float                 # gross reward:risk at TP1
    rr_tp1_net: float             # after round-trip fees + slippage
    sl_basis: str                 # which structure anchored the stop
    tradeable: bool               # cleared the net-RR gate
    cost_pct: float = 0.0         # modelled round-trip cost at entry
    stressed_entry: bool = False  # thin-book conditions were priced in
    time_stop_at: pd.Timestamp | None = None  # forced market exit deadline
    reject_reasons: tuple[str, ...] = ()


def decide_verdict(
    long_score: float,
    long_plan: TradePlan | None,
    short_score: float,
    short_plan: TradePlan | None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> tuple[str, tuple[str, ...]]:
    """Final tradeable direction: "LONG", "SHORT", or "NEUTRAL".

    Owned by the engine (not the UI) so a backtest and the live dashboard
    reach the same decision through the same code path. A side qualifies only
    if its score clears `min_score` AND its plan clears the net-RR gate; the
    higher-scoring qualifying side wins.
    """
    qualified: list[tuple[float, str]] = []
    notes: list[str] = []
    for side, score, plan in (
        ("LONG", long_score, long_plan),
        ("SHORT", short_score, short_plan),
    ):
        if score < config.min_score:
            notes.append(
                f"{side} score {score:.1f} below {config.min_score:.0f} threshold"
            )
            continue
        if plan is None:
            notes.append(f"{side} has no computable trade plan")
            continue
        if not plan.tradeable:
            notes.extend(f"{side}: {reason}" for reason in plan.reject_reasons)
            continue
        qualified.append((score, side))

    if not qualified:
        return "NEUTRAL", tuple(notes)
    qualified.sort(reverse=True)
    return qualified[0][1], ()


def determine_regime(df_1d: pd.DataFrame, config: EngineConfig = DEFAULT_CONFIG) -> str:
    """Macro regime from 1D EMA 50/200 alignment plus price location.

    bull: EMA50 > EMA200 and close > EMA50
    bear: EMA50 < EMA200 and close < EMA50
    chop: anything else (mixed/transition), or insufficient history.
    """
    if len(df_1d) < config.regime_min_candles:
        logger.warning(
            "Only %d daily candles (< %d): regime defaults to chop",
            len(df_1d), config.regime_min_candles,
        )
        return REGIME_CHOP
    close = df_1d["close"]
    fast = float(ema(close, config.regime_ema_fast).iloc[-1])
    slow = float(ema(close, config.regime_ema_slow).iloc[-1])
    last = float(close.iloc[-1])
    if fast > slow and last > fast:
        return REGIME_BULL
    if fast < slow and last < fast:
        return REGIME_BEAR
    return REGIME_CHOP


class TradePlanner:
    """Derives entry zone, structural SL, TP1/TP2, and sizing per direction."""

    def __init__(self, config: EngineConfig = DEFAULT_CONFIG) -> None:
        self.cfg = config

    def plan(
        self,
        side: str,
        m15: pd.DataFrame,
        zones: list[ConfluenceZone],
        levels: list[FibLevel],
        order_blocks: list[OrderBlock],
        fvgs: list[FairValueGap],
        sweeps: list[LiquiditySweep],
        swing_levels: list[SwingLevel],
        leg_1h: SwingLeg | None,
    ) -> TradePlan | None:
        is_long = side == "long"
        close = float(m15["close"].iloc[-1])
        atr = float(m15["atr"].iloc[-1])
        if not math.isfinite(close) or not math.isfinite(atr) or atr <= 0:
            return None

        # A sweep or a volume spike means the book is being cleared: price
        # the thin-book slippage rather than the quiet-market figure.
        vol_ratio = float(m15["vol_ratio"].iloc[-1]) if "vol_ratio" in m15 else 1.0
        if not math.isfinite(vol_ratio):
            vol_ratio = 1.0
        wanted = "low" if is_long else "high"
        stressed = bool(
            any(s.side == wanted for s in sweeps)
            or vol_ratio >= self.cfg.vol_spike_mult
        )
        state = MarketState(
            price=close, atr=atr, vol_ratio=vol_ratio, stressed=stressed
        )

        entry_low, entry_high, entry_is_poi = self._entry_zone(
            is_long, close, atr, zones, order_blocks, fvgs
        )
        sl, sl_basis = self._stop_loss(
            is_long, close, atr, m15, sweeps,
            entry_low, entry_high, entry_is_poi, swing_levels,
        )
        tp1, tp2 = self._take_profits(
            is_long, close, atr, levels, order_blocks, swing_levels, leg_1h
        )
        sl, tp1, tp2 = self._enforce_ordering(is_long, close, atr, sl, tp1, tp2)

        # Round-trip fees + slippage, in price terms. Every reward/risk figure
        # below is net of this: a stop 0.3% wide against a 0.14% cost is
        # nearly half cost, and gross RR would badly misrepresent the trade.
        cost_pct = round_trip_cost_pct(state, self.cfg)
        cost = close * cost_pct / 100.0
        risk_abs = abs(close - sl)
        reward_abs = abs(tp1 - close)

        risk_frac = (risk_abs + cost) / close
        account_risk = self.cfg.account_risk_pct / 100.0
        leverage = min(self.cfg.max_leverage, account_risk / risk_frac)
        risk_weight = round(leverage / self.cfg.max_leverage, 3)

        rr_tp1 = reward_abs / risk_abs
        rr_tp1_net = (reward_abs - cost) / (risk_abs + cost)

        reject_reasons: list[str] = []
        if rr_tp1_net < self.cfg.min_rr_tp1:
            reject_reasons.append(
                f"net RR at TP1 {rr_tp1_net:.2f} < required {self.cfg.min_rr_tp1:.2f} "
                f"(gross {rr_tp1:.2f}, round-trip cost {cost_pct:.2f}% of notional"
                + (", thin-book pricing applied)" if stressed else ")")
            )

        # Third barrier, enforced live and not only in the labeler: a position
        # that has reached neither target nor stop is closed at market, so the
        # backtest is not assuming an exit the trader was never told about.
        time_stop_at = m15.index[-1] + TF_DELTA["15m"] * (1 + self.cfg.max_hold_bars)

        return TradePlan(
            side=side,
            entry_zone_low=round(entry_low, 8),
            entry_zone_high=round(entry_high, 8),
            suggested_sl=round(sl, 8),
            suggested_tp1=round(tp1, 8),
            suggested_tp2=round(tp2, 8),
            risk_weight=risk_weight,
            # 3 decimals: rounding to 2 skews the risk budget by several
            # percent when a wide stop drives leverage below ~0.2x.
            suggested_leverage=round(leverage, 3),
            rr_tp1=round(rr_tp1, 2),
            rr_tp1_net=round(rr_tp1_net, 2),
            sl_basis=sl_basis,
            tradeable=not reject_reasons,
            cost_pct=round(cost_pct, 4),
            stressed_entry=stressed,
            time_stop_at=time_stop_at,
            reject_reasons=tuple(reject_reasons),
        )

    # ------------------------------------------------------------------
    def _entry_zone(
        self,
        is_long: bool,
        close: float,
        atr: float,
        zones: list[ConfluenceZone],
        order_blocks: list[OrderBlock],
        fvgs: list[FairValueGap],
    ) -> tuple[float, float, bool]:
        """Nearest aligned POI band within 1 ATR of price (is_poi=True), else
        a tight band just behind the current close (is_poi=False)."""
        bands: list[tuple[float, float]] = []
        bands += [(z.low, z.high) for z in zones]
        ob_side = "bullish" if is_long else "bearish"
        bands += [(ob.low, ob.high) for ob in order_blocks if ob.side == ob_side]
        bands += [(g.low, g.high) for g in fvgs if g.side == ob_side]

        if is_long:
            candidates = [
                b for b in bands
                if b[1] <= close + atr * 0.1 and close - b[0] <= atr
            ]
            if candidates:
                best = max(candidates, key=lambda b: b[1])
                return best[0], best[1], True
            return close - 0.5 * atr, close, False
        candidates = [
            b for b in bands
            if b[0] >= close - atr * 0.1 and b[1] - close <= atr
        ]
        if candidates:
            best = min(candidates, key=lambda b: b[0])
            return best[0], best[1], True
        return close, close + 0.5 * atr, False

    # ------------------------------------------------------------------
    def _stop_loss(
        self,
        is_long: bool,
        close: float,
        atr: float,
        m15: pd.DataFrame,
        sweeps: list[LiquiditySweep],
        entry_low: float,
        entry_high: float,
        entry_is_poi: bool,
        swing_levels: list[SwingLevel],
    ) -> tuple[float, str]:
        buffer = self.cfg.sl_atr_buffer * atr

        # 1) Structural invalidation of the sweep: just beyond the wick that
        #    ran the stops.
        wanted = "low" if is_long else "high"
        sweep_candles = [s.time for s in sweeps if s.side == wanted]
        if sweep_candles:
            rows = m15.loc[m15.index.isin(sweep_candles)]
            if not rows.empty:
                if is_long:
                    return float(rows["low"].min()) - buffer, "sweep_wick"
                return float(rows["high"].max()) + buffer, "sweep_wick"

        # 2) Far edge of the entry POI band (only when it is real structure).
        if entry_is_poi:
            if is_long and entry_low < close:
                return entry_low - buffer, "poi_band"
            if not is_long and entry_high > close:
                return entry_high + buffer, "poi_band"

        # 3) Nearest major structural swing behind the trade.
        if is_long:
            below = [lv.price for lv in swing_levels
                     if lv.side == "low" and lv.price < close]
            if below:
                return max(below) - buffer, "swing_level"
        else:
            above = [lv.price for lv in swing_levels
                     if lv.side == "high" and lv.price > close]
            if above:
                return min(above) + buffer, "swing_level"

        # 4) Pure volatility fallback.
        fallback = self.cfg.sl_fallback_atr * atr
        return (close - fallback if is_long else close + fallback), "atr_fallback"

    # ------------------------------------------------------------------
    def _take_profits(
        self,
        is_long: bool,
        close: float,
        atr: float,
        levels: list[FibLevel],
        order_blocks: list[OrderBlock],
        swing_levels: list[SwingLevel],
        leg_1h: SwingLeg | None,
    ) -> tuple[float, float]:
        min_dist = self.cfg.tp_min_atr_dist * atr

        # TP1: closest opposing OB edge or fib level in profit direction.
        tp1_candidates: list[float] = []
        opposing = "bearish" if is_long else "bullish"
        for ob in order_blocks:
            if ob.side == opposing:
                tp1_candidates.append(ob.low if is_long else ob.high)
        tp1_candidates += [lv.price for lv in levels]

        if is_long:
            valid1 = [p for p in tp1_candidates if p >= close + min_dist]
            tp1 = min(valid1) if valid1 else close + self.cfg.tp1_fallback_atr * atr
        else:
            valid1 = [p for p in tp1_candidates if p <= close - min_dist]
            tp1 = max(valid1) if valid1 else close - self.cfg.tp1_fallback_atr * atr

        # TP2: fib extension of the active 1h leg, else next structural swing.
        tp2_candidates: list[float] = []
        if leg_1h is not None:
            span = leg_1h.high - leg_1h.low
            if is_long:
                tp2_candidates.append(leg_1h.low + self.cfg.fib_extension * span)
            else:
                tp2_candidates.append(leg_1h.high - self.cfg.fib_extension * span)
        for lv in swing_levels:
            if is_long and lv.side == "high" and lv.price > tp1 + min_dist:
                tp2_candidates.append(lv.price)
            elif not is_long and lv.side == "low" and lv.price < tp1 - min_dist:
                tp2_candidates.append(lv.price)

        if is_long:
            valid2 = [p for p in tp2_candidates if p > tp1 + min_dist]
            tp2 = min(valid2) if valid2 else close + self.cfg.tp2_fallback_atr * atr
        else:
            valid2 = [p for p in tp2_candidates if p < tp1 - min_dist]
            tp2 = max(valid2) if valid2 else close - self.cfg.tp2_fallback_atr * atr
        return tp1, tp2

    # ------------------------------------------------------------------
    def _enforce_ordering(
        self,
        is_long: bool,
        close: float,
        atr: float,
        sl: float,
        tp1: float,
        tp2: float,
    ) -> tuple[float, float, float]:
        """Guarantee sl < entry < tp1 < tp2 (mirrored for shorts)."""
        if is_long:
            if sl >= close:
                sl = close - self.cfg.sl_fallback_atr * atr
            if tp1 <= close:
                tp1 = close + self.cfg.tp1_fallback_atr * atr
            if tp2 <= tp1:
                tp2 = tp1 + (self.cfg.tp2_fallback_atr - self.cfg.tp1_fallback_atr) * atr
        else:
            if sl <= close:
                sl = close + self.cfg.sl_fallback_atr * atr
            if tp1 >= close:
                tp1 = close - self.cfg.tp1_fallback_atr * atr
            if tp2 >= tp1:
                tp2 = tp1 - (self.cfg.tp2_fallback_atr - self.cfg.tp1_fallback_atr) * atr
        return sl, tp1, tp2
