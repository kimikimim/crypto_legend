"""Triple-barrier outcome labeling on 1-minute price paths.

Resolving a trade on 15m bars alone forces a guess whenever one bar spans
both the stop and the target, and guessing favourably is the standard way a
backtest ends up overstated. Every outcome here is resolved by walking the
1m series, and when a single 1m bar still straddles both levels the stop is
assumed to have been hit first. Results are therefore a lower bound.

Execution assumptions, stated explicitly because they drive the numbers:
- entry is a market fill at the close of the signal bar (the first moment
  the signal exists), never inside the entry zone at a better price;
- fees and slippage are charged once as a round trip, on top of the stop
  distance, matching the live sizing model;
- a position that reaches neither barrier within `max_hold_bars` is closed
  at the market price then, counted as a timeout rather than discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import DEFAULT_CONFIG, TF_DELTA, EngineConfig

logger = logging.getLogger(__name__)

BARRIER_TP1 = "tp1"
BARRIER_SL = "sl"
BARRIER_TIMEOUT = "timeout"


@dataclass(frozen=True)
class BacktestReport:
    trades: pd.DataFrame
    signals_in: int
    resolved: int
    unresolved_missing_1m: int

    @property
    def coverage(self) -> float:
        return self.resolved / self.signals_in if self.signals_in else 0.0


class TripleBarrierLabeler:
    """Turns signals into realized outcomes using the 1m path."""

    def __init__(self, config: EngineConfig = DEFAULT_CONFIG) -> None:
        self.cfg = config

    def resolve_one(
        self,
        side: str,
        entry_price: float,
        sl: float,
        tp1: float,
        tp2: float,
        path: pd.DataFrame,
    ) -> dict | None:
        """Resolve a single trade against its forward 1m path.

        `path` must contain only bars strictly after the entry, already
        limited to the maximum holding period.
        """
        if path.empty:
            return None
        is_long = side == "long"
        high = path["high"].to_numpy(dtype=float)
        low = path["low"].to_numpy(dtype=float)

        hit_sl = low <= sl if is_long else high >= sl
        hit_tp1 = high >= tp1 if is_long else low <= tp1
        hit_tp2 = high >= tp2 if is_long else low <= tp2

        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
        i_tp1 = int(np.argmax(hit_tp1)) if hit_tp1.any() else None
        i_tp2 = int(np.argmax(hit_tp2)) if hit_tp2.any() else None

        if i_sl is not None and (i_tp1 is None or i_sl <= i_tp1):
            # Ties go to the stop: within one 1m bar the true order is
            # unknown, so assume the unfavourable one.
            barrier, idx, exit_price = BARRIER_SL, i_sl, sl
        elif i_tp1 is not None:
            barrier, idx, exit_price = BARRIER_TP1, i_tp1, tp1
        else:
            barrier, idx = BARRIER_TIMEOUT, len(path) - 1
            exit_price = float(path["close"].iloc[-1])

        # TP2 only counts if it was reached before the stop.
        tp2_reached = bool(
            i_tp2 is not None and (i_sl is None or i_tp2 < i_sl)
        )

        direction = 1.0 if is_long else -1.0
        gross_pct = direction * (exit_price - entry_price) / entry_price * 100.0
        cost_pct = self.cfg.round_trip_cost_pct
        net_pct = gross_pct - cost_pct

        risk_pct = (abs(entry_price - sl) / entry_price) * 100.0 + cost_pct
        return {
            "barrier": barrier,
            "exit_at": path.index[idx],
            "exit_price": exit_price,
            "gross_pnl_pct": gross_pct,
            "net_pnl_pct": net_pct,
            "r_multiple": net_pct / risk_pct if risk_pct > 0 else 0.0,
            "minutes_held": int(
                (path.index[idx] - path.index[0]).total_seconds() // 60
            )
            + 1,
            "tp2_reached": tp2_reached,
        }

    # ------------------------------------------------------------------
    def run(
        self,
        signals: pd.DataFrame,
        m1: pd.DataFrame,
        max_hold_bars: int | None = None,
    ) -> BacktestReport:
        """Label every actionable signal. `signals` is a Replayer.run frame."""
        actionable = signals[signals["verdict"] != "NEUTRAL"]
        max_hold = pd.Timedelta(
            minutes=15 * (max_hold_bars or self.cfg.max_hold_bars)
        )

        rows: list[dict] = []
        missing = 0
        for evaluated_at, sig in actionable.iterrows():
            entry_time = evaluated_at + TF_DELTA["15m"]  # signal bar's close
            window = m1.loc[
                (m1.index >= entry_time) & (m1.index < entry_time + max_hold)
            ]
            outcome = self.resolve_one(
                side=str(sig["side"]),
                entry_price=float(sig["price"]),
                sl=float(sig["sl"]),
                tp1=float(sig["tp1"]),
                tp2=float(sig["tp2"]),
                path=window,
            )
            if outcome is None:
                missing += 1
                continue
            rows.append(
                {
                    "signal_at": evaluated_at,
                    "entry_at": entry_time,
                    "symbol": sig.get("symbol"),
                    "side": sig["side"],
                    "score": (
                        sig["long_score"] if sig["side"] == "long" else sig["short_score"]
                    ),
                    "regime": sig.get("regime"),
                    "entry_price": float(sig["price"]),
                    "sl": float(sig["sl"]),
                    "tp1": float(sig["tp1"]),
                    "tp2": float(sig["tp2"]),
                    "rr_tp1_net": sig.get("rr_tp1_net"),
                    "sl_basis": sig.get("sl_basis"),
                    "risk_weight": sig.get("risk_weight"),
                    # Carry the winning side's category points so each can be
                    # tested for information on its own, without a re-replay.
                    **{
                        f"cat_{cat}": sig.get(f"{sig['side']}_{cat}")
                        for cat in (
                            "trend", "location", "whale",
                            "momentum", "volatility", "penalty",
                        )
                    },
                    **outcome,
                }
            )

        trades = (
            pd.DataFrame(rows).set_index("signal_at").sort_index()
            if rows
            else pd.DataFrame()
        )
        if missing:
            logger.warning(
                "%d signals unresolved: no 1m coverage for their holding window",
                missing,
            )
        return BacktestReport(
            trades=trades,
            signals_in=len(actionable),
            resolved=len(rows),
            unresolved_missing_1m=missing,
        )
