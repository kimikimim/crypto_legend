"""Orchestrator: fetch (OHLCV + OI + CVD) -> drop open candles -> indicators
-> fib/SMC structures -> liquidation signal -> merge -> score."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from app.config import DEFAULT_CONFIG, TIMEFRAMES, EngineConfig
from app.data_fetcher import DataFetcher, drop_open_candle, validate_symbol
from app.exceptions import InsufficientDataError
from app.fibonacci import (
    ConfluenceZone,
    FibLevel,
    fib_levels,
    find_confluence_zones,
    find_swing_leg,
)
from app.indicators import IndicatorCalculator
from app.liquidations import (
    NO_SIGNAL,
    LiquidationSignal,
    LiquidationTracker,
    proxy_liquidation_signal,
)
from app.mtf import merge_mtf
from app.risk import REGIME_CHOP, TradePlan, TradePlanner, determine_regime
from app.scoring import ScoreResult, ScoringContext, ScoringEngine
from app.smc import (
    FairValueGap,
    LiquiditySweep,
    OrderBlock,
    RawBreakout,
    SqueezeAlert,
    detect_squeeze,
    find_fvgs,
    find_order_blocks,
    find_sweeps_and_breakouts,
    major_swing_levels,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenInterestStats:
    value: float | None = None       # latest OI (USDT notional)
    z: float | None = None           # vs 24h rolling window
    change_z: float | None = None    # last delta vs 24h deltas


@dataclass(frozen=True)
class CvdStats:
    last: float | None = None        # latest cumulative delta
    delta: float | None = None       # last 15m delta
    slope3: float | None = None      # sum of last three 15m deltas
    divergence: str | None = None    # "bullish" / "bearish" / None


@dataclass(frozen=True)
class SmartMoneyReport:
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fvgs: list[FairValueGap] = field(default_factory=list)
    sweeps: list[LiquiditySweep] = field(default_factory=list)
    breakouts: list[RawBreakout] = field(default_factory=list)
    liquidations: LiquidationSignal = NO_SIGNAL
    squeeze: SqueezeAlert = SqueezeAlert(active=False, oi_z=None, reasons=())
    open_interest: OpenInterestStats = OpenInterestStats()
    cvd: CvdStats = CvdStats()


@dataclass(frozen=True)
class AnalysisResult:
    symbol: str
    evaluated_at: pd.Timestamp        # open time of the scored 15m candle
    price: float                      # close of the scored 15m candle
    use_closed_candle: bool
    scores: ScoreResult
    zones: list[ConfluenceZone]
    levels_4h: list[FibLevel]
    levels_1h: list[FibLevel]
    smart_money: SmartMoneyReport
    regime: str                       # 1D macro regime: bull / bear / chop
    long_plan: TradePlan | None
    short_plan: TradePlan | None
    klines: list[dict]                # recent 15m candles for charting

    @property
    def primary_direction(self) -> str:
        return "long" if self.scores.long.total >= self.scores.short.total else "short"

    @property
    def primary_plan(self) -> TradePlan | None:
        return self.long_plan if self.primary_direction == "long" else self.short_plan


class MTFAnalysisEngine:
    """End-to-end multi-timeframe + smart-money analysis for one symbol.

    Only the hardcoded BTC/ETH/SOL USDT-M pairs are accepted.
    """

    def __init__(
        self,
        fetcher: DataFetcher | None = None,
        config: EngineConfig = DEFAULT_CONFIG,
        liquidation_tracker: LiquidationTracker | None = None,
    ) -> None:
        self.cfg = config
        self.fetcher = fetcher or DataFetcher()
        self.indicators = IndicatorCalculator(config)
        self.scorer = ScoringEngine(config)
        self.planner = TradePlanner(config)
        self.liq_tracker = liquidation_tracker

    def analyze(self, symbol: str, use_closed_candle: bool = True) -> AnalysisResult:
        unified = validate_symbol(symbol)
        logger.info(
            "Analyzing %s (use_closed_candle=%s)", unified, use_closed_candle
        )

        frames = self.fetcher.fetch_mtf(unified, TIMEFRAMES, limit=self.cfg.fetch_limit)
        if use_closed_candle:
            frames = {tf: drop_open_candle(df, tf) for tf, df in frames.items()}
        for tf, df in frames.items():
            if len(df) < self.cfg.min_candles:
                raise InsufficientDataError(
                    f"{unified} {tf}: only {len(df)} candles "
                    f"(need >= {self.cfg.min_candles})"
                )

        enriched = {tf: self.indicators.enrich(df) for tf, df in frames.items()}
        merged = merge_mtf(enriched["15m"], enriched["1h"], enriched["4h"])
        row = merged.iloc[-1]
        eval_close = merged.index[-1] + pd.Timedelta(minutes=15)

        # --- fibonacci confluence ---
        leg_4h = find_swing_leg(enriched["4h"], self.cfg.extrema_order["4h"])
        leg_1h = find_swing_leg(enriched["1h"], self.cfg.extrema_order["1h"])
        levels_4h = fib_levels(leg_4h, "4h", self.cfg)
        levels_1h = fib_levels(leg_1h, "1h", self.cfg)
        atr_1h = float(enriched["1h"]["atr"].iloc[-1])
        zones = find_confluence_zones(
            levels_4h, levels_1h, self.cfg.overlap_atr_mult * atr_1h
        )

        # --- smart money structures ---
        order_blocks = find_order_blocks(enriched["1h"], "1h", self.cfg) + \
            find_order_blocks(enriched["15m"], "15m", self.cfg)
        fvgs = find_fvgs(enriched["1h"], "1h", self.cfg) + \
            find_fvgs(enriched["15m"], "15m", self.cfg)
        swing_levels = major_swing_levels(
            enriched["4h"], "4h", self.cfg.extrema_order["4h"],
            self.cfg.major_pivot_count,
        ) + major_swing_levels(
            enriched["1h"], "1h", self.cfg.extrema_order["1h"],
            self.cfg.major_pivot_count,
        )
        sweeps, breakouts = find_sweeps_and_breakouts(
            enriched["15m"], swing_levels, self.cfg
        )

        # --- 1D macro regime filter (fail-soft to "chop" = no penalty) ---
        regime = self._macro_regime(unified, use_closed_candle)

        # --- derivatives data (fail-soft: engine still scores without it) ---
        oi_stats = self._open_interest_stats(unified, eval_close)
        cvd_stats = self._cvd_stats(unified, eval_close, enriched["15m"])

        # --- liquidation signal: measured stream if warm, else OI proxy ---
        liq_signal = self._liquidation_signal(unified, row, oi_stats, eval_close)

        # --- squeeze warning ---
        poi_centers = [z.center for z in zones] + [ob.center for ob in order_blocks]
        squeeze = detect_squeeze(
            enriched["15m"],
            oi_stats.z,
            poi_centers,
            "; ".join(liq_signal.detail) if liq_signal.detail else None,
            self.cfg,
        )

        ctx = ScoringContext(
            zones=zones,
            levels_4h=levels_4h,
            levels_1h=levels_1h,
            order_blocks=order_blocks,
            fvgs=fvgs,
            sweeps=sweeps,
            breakouts=breakouts,
            liquidations=liq_signal,
            regime=regime,
        )
        scores = self.scorer.score(row, ctx)

        # --- structural trade plans (entry zone / SL / TP1 / TP2 / sizing) ---
        plan_args = dict(
            m15=enriched["15m"],
            zones=zones,
            levels=levels_4h + levels_1h,
            order_blocks=order_blocks,
            fvgs=fvgs,
            sweeps=sweeps,
            swing_levels=swing_levels,
            leg_1h=leg_1h,
        )
        long_plan = self.planner.plan("long", **plan_args)
        short_plan = self.planner.plan("short", **plan_args)

        return AnalysisResult(
            symbol=unified,
            evaluated_at=merged.index[-1],
            price=float(row["close"]),
            use_closed_candle=use_closed_candle,
            scores=scores,
            zones=zones,
            levels_4h=levels_4h,
            levels_1h=levels_1h,
            smart_money=SmartMoneyReport(
                order_blocks=order_blocks,
                fvgs=fvgs,
                sweeps=sweeps,
                breakouts=breakouts,
                liquidations=liq_signal,
                squeeze=squeeze,
                open_interest=oi_stats,
                cvd=cvd_stats,
            ),
            regime=regime,
            long_plan=long_plan,
            short_plan=short_plan,
            klines=self._chart_klines(enriched["15m"]),
        )

    @staticmethod
    def _chart_klines(m15: pd.DataFrame, count: int = 300) -> list[dict]:
        tail = m15.tail(count)
        return [
            {
                "time": ts.isoformat(),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
            for ts, r in tail.iterrows()
        ]

    def _macro_regime(self, symbol: str, use_closed_candle: bool) -> str:
        try:
            df_1d = self.fetcher.fetch_ohlcv(symbol, "1d", limit=self.cfg.fetch_limit)
            if use_closed_candle:
                df_1d = drop_open_candle(df_1d, "1d")
            regime = determine_regime(df_1d, self.cfg)
            logger.info("1D macro regime for %s: %s", symbol, regime)
            return regime
        except Exception as exc:  # noqa: BLE001 — fail-soft, no regime penalty
            logger.warning("1D regime unavailable for %s: %s", symbol, exc)
            return REGIME_CHOP

    # ------------------------------------------------------------------
    # derivatives helpers
    # ------------------------------------------------------------------
    def _open_interest_stats(
        self, symbol: str, eval_close: pd.Timestamp
    ) -> OpenInterestStats:
        try:
            oi_df = self.fetcher.fetch_open_interest(
                symbol, period=self.cfg.oi_period
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft, engine still scores
            logger.warning("OI unavailable for %s: %s", symbol, exc)
            return OpenInterestStats()

        oi = oi_df.loc[oi_df.index <= eval_close, "oi"]
        if len(oi) < self.cfg.oi_window // 2:
            logger.warning("OI history too short for %s (%d points)", symbol, len(oi))
            return OpenInterestStats()

        window = oi.tail(self.cfg.oi_window)
        mean, std = float(window.mean()), float(window.std(ddof=0))
        z = (float(oi.iloc[-1]) - mean) / std if std > 0 else 0.0

        deltas = oi.diff().dropna().tail(self.cfg.oi_window)
        dmean, dstd = float(deltas.mean()), float(deltas.std(ddof=0))
        change_z = (
            (float(deltas.iloc[-1]) - dmean) / dstd if dstd > 0 and len(deltas) else None
        )
        return OpenInterestStats(value=float(oi.iloc[-1]), z=z, change_z=change_z)

    def _cvd_stats(
        self, symbol: str, eval_close: pd.Timestamp, m15: pd.DataFrame
    ) -> CvdStats:
        try:
            cvd_df = self.fetcher.fetch_cvd(
                symbol, interval=self.cfg.cvd_interval, limit=self.cfg.cvd_limit
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft, engine still scores
            logger.warning("CVD unavailable for %s: %s", symbol, exc)
            return CvdStats()

        # Only 15m buckets fully closed by the evaluation time.
        closed = cvd_df.loc[cvd_df.index + pd.Timedelta(minutes=15) <= eval_close]
        if len(closed) < self.cfg.cvd_div_lookback + 1:
            return CvdStats()

        cvd = closed["cvd"]
        delta = closed["delta"]
        lb = self.cfg.cvd_div_lookback
        cvd_chg = float(cvd.iloc[-1] - cvd.iloc[-1 - lb])

        divergence = None
        if len(m15) > lb:
            price_chg = float(m15["close"].iloc[-1] - m15["close"].iloc[-1 - lb])
            if price_chg < 0 < cvd_chg:
                divergence = "bullish"    # price down, aggressive buyers absorb
            elif price_chg > 0 > cvd_chg:
                divergence = "bearish"    # price up, aggressive sellers distribute
        return CvdStats(
            last=float(cvd.iloc[-1]),
            delta=float(delta.iloc[-1]),
            slope3=float(delta.tail(3).sum()),
            divergence=divergence,
        )

    def _liquidation_signal(
        self,
        symbol: str,
        row: pd.Series,
        oi_stats: OpenInterestStats,
        eval_close: pd.Timestamp,
    ) -> LiquidationSignal:
        if self.liq_tracker is not None:
            measured = self.liq_tracker.signal(symbol, now=eval_close)
            if measured is not None:
                return measured
            logger.info(
                "Liquidation stream not warm for %s — using OI/volume proxy",
                symbol,
            )
        return proxy_liquidation_signal(row, oi_stats.change_z, self.cfg)
