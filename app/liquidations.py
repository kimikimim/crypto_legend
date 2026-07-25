"""Liquidation tracking: live @forceOrder stream + cold-start proxy.

Binance provides no public REST endpoint for historical liquidations, so:
- `LiquidationCollector` (optional, started by the API lifespan) watches the
  real-time liquidation stream via ccxt.pro and feeds `LiquidationTracker`.
- Until the tracker is warm, `proxy_liquidation_signal` infers flush events
  deterministically from OI drops + volume spikes on the closed 15m candle.

Side convention (documented, flip via interpretation if desired):
- A SELL force order liquidates a LONG position -> "long flush". A long-flush
  spike at the lows is capitulation that validates a mean-reversion LONG.
- A BUY force order liquidates a SHORT -> "short flush", validating a SHORT
  entry after a sweep of the highs.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from app.config import ALLOWED_SYMBOLS, DEFAULT_CONFIG, EngineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiquidationEvent:
    timestamp: pd.Timestamp
    side: str        # "long" (longs liquidated) or "short"
    notional: float  # USDT value


@dataclass(frozen=True)
class LiquidationSignal:
    long_flush: bool     # long-capitulation spike -> validates LONG entries
    short_flush: bool    # short-squeeze spike -> validates SHORT entries
    source: str          # "measured" | "proxy" | "none"
    detail: tuple[str, ...] = ()


NO_SIGNAL = LiquidationSignal(False, False, "none")


class LiquidationTracker:
    """Rolling in-memory window of liquidation events per symbol."""

    def __init__(self, config: EngineConfig = DEFAULT_CONFIG) -> None:
        self.cfg = config
        self._events: dict[str, deque[LiquidationEvent]] = {
            s: deque() for s in ALLOWED_SYMBOLS
        }
        self._started_at: pd.Timestamp | None = None

    def mark_started(self, now: pd.Timestamp | None = None) -> None:
        if self._started_at is None:
            self._started_at = now or pd.Timestamp.now(tz="UTC")

    def record(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        side: str,
        notional: float,
    ) -> None:
        if symbol not in self._events:
            return  # whitelist only
        if side not in ("long", "short") or notional <= 0:
            return
        self._events[symbol].append(LiquidationEvent(timestamp, side, notional))
        self._prune(symbol, timestamp)

    def _prune(self, symbol: str, now: pd.Timestamp) -> None:
        cutoff = now - pd.Timedelta(hours=self.cfg.liq_window_hours)
        events = self._events[symbol]
        while events and events[0].timestamp < cutoff:
            events.popleft()

    def is_warm(self, now: pd.Timestamp | None = None) -> bool:
        """Warm once the collector has streamed for liq_warm_min_buckets."""
        if self._started_at is None:
            return False
        now = now or pd.Timestamp.now(tz="UTC")
        bucket = pd.Timedelta(self.cfg.liq_bucket)
        return now - self._started_at >= self.cfg.liq_warm_min_buckets * bucket

    def signal(
        self, symbol: str, now: pd.Timestamp | None = None
    ) -> LiquidationSignal | None:
        """Measured spike signal, or None when the stream is not warm yet.

        A side spikes when the max bucket notional of the last
        `sweep_lookback` buckets exceeds baseline mean + z_threshold * std.
        """
        if not self.is_warm(now) or symbol not in self._events:
            return None
        now = now or pd.Timestamp.now(tz="UTC")
        self._prune(symbol, now)
        events = list(self._events[symbol])

        bucket = pd.Timedelta(self.cfg.liq_bucket)
        window = pd.Timedelta(hours=self.cfg.liq_window_hours)
        edges = pd.date_range(
            end=now.ceil(self.cfg.liq_bucket), freq=bucket,
            periods=int(window / bucket) + 1,
        )
        per_side: dict[str, pd.Series] = {}
        for side in ("long", "short"):
            series = pd.Series(0.0, index=edges[:-1])
            for ev in events:
                if ev.side != side:
                    continue
                slot = ev.timestamp.floor(self.cfg.liq_bucket)
                if slot in series.index:
                    series[slot] += ev.notional
            per_side[side] = series

        recent_n = self.cfg.sweep_lookback
        detail: list[str] = []
        flags: dict[str, bool] = {}
        for side, series in per_side.items():
            baseline = series.iloc[:-recent_n]
            recent = series.iloc[-recent_n:]
            mean, std = float(baseline.mean()), float(baseline.std(ddof=0))
            peak = float(recent.max())
            if std > 0:
                spiked = peak > mean + self.cfg.liq_z_threshold * std
            else:
                spiked = peak > 0 and mean == 0
            flags[side] = spiked
            if spiked:
                detail.append(
                    f"{side}-side liquidations {peak:,.0f} USDT vs "
                    f"24h mean {mean:,.0f}"
                )
        return LiquidationSignal(
            long_flush=flags["long"],
            short_flush=flags["short"],
            source="measured",
            detail=tuple(detail),
        )


def proxy_liquidation_signal(
    row: Mapping[str, Any],
    oi_change_z: float | None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> LiquidationSignal:
    """Cold-start proxy: a liquidation cascade forcibly closes positions, so
    OI drops sharply while volume spikes. Candle color infers the flushed
    side (red flush = longs, green spike = shorts)."""
    if oi_change_z is None or math.isnan(oi_change_z):
        return NO_SIGNAL
    vol_ratio = float(row.get("vol_ratio", math.nan))
    close = float(row.get("close", math.nan))
    open_ = float(row.get("open", math.nan))
    if math.isnan(vol_ratio) or math.isnan(close) or math.isnan(open_):
        return NO_SIGNAL

    cascade = (
        oi_change_z <= -config.liq_z_threshold
        and vol_ratio >= config.vol_spike_mult
    )
    if not cascade:
        return NO_SIGNAL
    detail = (
        f"OI drop z={oi_change_z:.1f} with {vol_ratio:.1f}x volume "
        f"(proxy inference)",
    )
    if close < open_:
        return LiquidationSignal(True, False, "proxy", detail)
    if close > open_:
        return LiquidationSignal(False, True, "proxy", detail)
    return NO_SIGNAL


class LiquidationCollector:
    """Background task streaming Binance @forceOrder events via ccxt.pro."""

    def __init__(
        self,
        tracker: LiquidationTracker,
        symbols: tuple[str, ...] = ALLOWED_SYMBOLS,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.tracker = tracker
        self.symbols = symbols
        self.reconnect_delay = reconnect_delay
        self._exchange: Any | None = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        try:
            import ccxt.pro as ccxtpro  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "ccxt.pro unavailable — liquidation stream disabled, "
                "falling back to OI/volume proxy"
            )
            return
        self._exchange = ccxtpro.binanceusdm({"enableRateLimit": True})
        self.tracker.mark_started()
        self._tasks = [
            asyncio.create_task(self._watch(symbol), name=f"liq-{symbol}")
            for symbol in self.symbols
        ]
        logger.info("Liquidation collector started for %s", ", ".join(self.symbols))

    async def _watch(self, symbol: str) -> None:
        while True:
            try:
                events = await self._exchange.watch_liquidations(symbol)
                for event in events or []:
                    self._ingest(symbol, event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # reconnect on any stream error
                logger.warning("Liquidation stream %s error: %s", symbol, exc)
                await asyncio.sleep(self.reconnect_delay)

    def _ingest(self, symbol: str, event: dict) -> None:
        try:
            info_order = (event.get("info") or {}).get("o") or {}
            raw_side = (event.get("side") or info_order.get("S") or "").lower()
            # A SELL force order closes a long; BUY closes a short.
            side = {"sell": "long", "buy": "short"}.get(raw_side)
            notional = event.get("quoteValue")
            if notional is None:
                price = float(event.get("price") or 0)
                amount = float(event.get("amount") or 0)
                notional = price * amount
            ts = pd.Timestamp(event.get("timestamp"), unit="ms", tz="UTC")
            if side:
                self.tracker.record(symbol, ts, side, float(notional))
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("Skipping malformed liquidation event: %s (%s)", event, exc)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._exchange is not None:
            await self._exchange.close()
        logger.info("Liquidation collector stopped")
