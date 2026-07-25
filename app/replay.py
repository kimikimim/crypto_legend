"""Historical replay: score any past bar exactly as the live engine would.

The whole point is parity. Rather than reimplementing the analysis for
backtesting — which would validate a parallel system nobody trades — replay
feeds the *same* `analyze_frames` the live path uses, with every frame
truncated at the bar being scored. Truncation is what makes the
forward-looking mitigation scans in find_fvgs / find_order_blocks safe: at
bar i the data simply ends at bar i, so there is no future to leak.

Higher timeframes are derived from the stored 15m series and cut by candle
CLOSE time, mirroring live behaviour where only closed bars are visible.

Note on coverage: Open Interest, CVD and liquidations are not replayable —
Binance serves ~30 days of OI history and no public liquidation history — so
replay scores the price-structure subset and leaves the Whale category to
the forward-test journal. Signals here are therefore a lower bound on the
full model's score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from app.config import DEFAULT_CONFIG, TF_DELTA, EngineConfig
from app.engine import AnalysisResult, MTFAnalysisEngine
from app.exceptions import EngineError
from app.store import resample_ohlcv

logger = logging.getLogger(__name__)

HTF_TIMEFRAMES = ("1h", "4h", "1d")


@dataclass(frozen=True)
class ReplayWindow:
    """How much history each timeframe gets, in bars."""

    m15: int = 500
    h1: int = 500
    h4: int = 400
    d1: int = 400


class Replayer:
    """Walks a stored 15m series and scores each bar as of that moment."""

    def __init__(
        self,
        engine: MTFAnalysisEngine | None = None,
        config: EngineConfig = DEFAULT_CONFIG,
        window: ReplayWindow = ReplayWindow(),
    ) -> None:
        self.cfg = config
        self.engine = engine or MTFAnalysisEngine(config=config)
        self.window = window

    # ------------------------------------------------------------------
    def prepare(self, m15: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Derive the higher timeframes once, up front."""
        if m15.empty:
            raise ValueError("15m series is empty")
        if not m15.index.is_monotonic_increasing:
            raise ValueError("15m series must be sorted by time")
        return {tf: resample_ohlcv(m15, tf) for tf in HTF_TIMEFRAMES}

    def frames_at(
        self,
        m15: pd.DataFrame,
        htf: dict[str, pd.DataFrame],
        i: int,
    ) -> dict[str, pd.DataFrame]:
        """Everything visible at the close of 15m bar `i`, and nothing more."""
        close_time = m15.index[i] + TF_DELTA["15m"]
        frames = {"15m": m15.iloc[max(0, i - self.window.m15 + 1) : i + 1]}
        sizes = {"1h": self.window.h1, "4h": self.window.h4, "1d": self.window.d1}
        for tf in HTF_TIMEFRAMES:
            full = htf[tf]
            # index + duration <= close_time  <=>  index <= close_time - duration
            cut = int(full.index.searchsorted(close_time - TF_DELTA[tf], side="right"))
            frames[tf] = full.iloc[max(0, cut - sizes[tf]) : cut]
        return frames

    # ------------------------------------------------------------------
    def iter_results(
        self,
        symbol: str,
        m15: pd.DataFrame,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        stride: int = 1,
    ) -> Iterator[AnalysisResult]:
        """Yield one AnalysisResult per scored bar."""
        htf = self.prepare(m15)
        first = 0 if start is None else int(m15.index.searchsorted(start, side="left"))
        last = len(m15) if end is None else int(m15.index.searchsorted(end, side="right"))

        skipped = 0
        for i in range(first, last, stride):
            try:
                yield self.engine.analyze_frames(
                    symbol, self.frames_at(m15, htf, i)
                )
            except EngineError:
                skipped += 1  # warm-up bars without enough history
                continue
        if skipped:
            logger.info("Replay skipped %d bars for insufficient history", skipped)

    def run(
        self,
        symbol: str,
        m15: pd.DataFrame,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        stride: int = 1,
        progress_every: int = 2000,
    ) -> pd.DataFrame:
        """Replay a range and return one row per scored bar."""
        rows: list[dict] = []
        for result in self.iter_results(symbol, m15, start, end, stride):
            rows.append(result_to_row(result))
            if progress_every and len(rows) % progress_every == 0:
                logger.info("  replayed %d bars (at %s)", len(rows), result.evaluated_at)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows).set_index("evaluated_at").sort_index()
        logger.info(
            "Replayed %s: %d bars, %d actionable",
            symbol, len(out), int((out["verdict"] != "NEUTRAL").sum()),
        )
        return out


def result_to_row(result: AnalysisResult) -> dict:
    """Flatten one analysis into a ledger row, keeping the per-category
    points that the ablation study needs."""
    plan = result.primary_plan
    row = {
        "evaluated_at": result.evaluated_at,
        "symbol": result.symbol,
        "price": result.price,
        "atr_15m": result.atr_15m,
        "regime": result.regime,
        "verdict": result.verdict,
        "long_score": result.scores.long.total,
        "short_score": result.scores.short.total,
        "sweep_count": len(result.smart_money.sweeps),
        "breakout_count": len(result.smart_money.breakouts),
        "ob_count": len(result.smart_money.order_blocks),
        "fvg_count": len(result.smart_money.fvgs),
        "fib_zone_count": len(result.zones),
    }
    for side, d in (("long", result.scores.long), ("short", result.scores.short)):
        row[f"{side}_trend"] = d.trend.points
        row[f"{side}_location"] = d.location.points
        row[f"{side}_whale"] = d.whale.points
        row[f"{side}_momentum"] = d.momentum.points
        row[f"{side}_volatility"] = d.volatility.points
        row[f"{side}_penalty"] = d.penalty
    row.update(
        {
            "side": plan.side if plan else None,
            "entry_low": plan.entry_zone_low if plan else None,
            "entry_high": plan.entry_zone_high if plan else None,
            "sl": plan.suggested_sl if plan else None,
            "tp1": plan.suggested_tp1 if plan else None,
            "tp2": plan.suggested_tp2 if plan else None,
            "leverage": plan.suggested_leverage if plan else None,
            "risk_weight": plan.risk_weight if plan else None,
            "rr_tp1_net": plan.rr_tp1_net if plan else None,
            "sl_basis": plan.sl_basis if plan else None,
        }
    )
    return row
