"""Binance USDT-M Futures market data via ccxt, with retry/backoff.

Fetches OHLCV, Open Interest history, and taker-buy klines (for CVD).
All fetching is hard-restricted to the BTC/ETH/SOL USDT whitelist.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

import ccxt
import pandas as pd

from app.config import ALLOWED_SYMBOLS, TF_DELTA
from app.exceptions import DataFetchError, UnsupportedSymbolError

logger = logging.getLogger(__name__)

T = TypeVar("T")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_symbol(symbol: str) -> str:
    """Normalize user input to a ccxt unified USDT-M futures symbol.

    "BTCUSDT" / "btc/usdt" / "BTC/USDT:USDT" -> "BTC/USDT:USDT"
    """
    s = symbol.upper().replace("-", "").strip()
    if s.endswith(":USDT"):
        return s
    if "/" in s:
        return f"{s}:USDT"
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT:USDT"
    raise ValueError(f"Cannot normalize symbol: {symbol!r} (expected a USDT pair)")


def validate_symbol(symbol: str) -> str:
    """Normalize and enforce the hardcoded three-symbol whitelist."""
    unified = normalize_symbol(symbol)
    if unified not in ALLOWED_SYMBOLS:
        raise UnsupportedSymbolError(
            f"{unified} is not supported; the engine is restricted to "
            f"{', '.join(ALLOWED_SYMBOLS)}"
        )
    return unified


def market_id(unified: str) -> str:
    """"BTC/USDT:USDT" -> Binance raw id "BTCUSDT"."""
    return unified.split("/")[0] + "USDT"


class DataFetcher:
    """Market data access for Binance USDT-M Futures (whitelisted pairs only).

    Handles rate limits and transient network errors with exponential
    backoff. An injected `exchange` (any object with the used methods)
    makes the class fully testable offline.
    """

    def __init__(
        self,
        exchange: Any | None = None,
        max_retries: int = 4,
        backoff_base: float = 1.5,
    ) -> None:
        self._exchange = exchange or ccxt.binanceusdm({"enableRateLimit": True})
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch candles as a UTC-indexed DataFrame (index = candle open time)."""
        if timeframe not in TF_DELTA:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        unified = validate_symbol(symbol)
        raw = self._with_retries(
            f"OHLCV {unified} {timeframe}",
            lambda: self._exchange.fetch_ohlcv(unified, timeframe=timeframe, limit=limit),
        )
        return self._to_frame(raw, unified, timeframe)

    def fetch_mtf(
        self, symbol: str, timeframes: tuple[str, ...], limit: int = 500
    ) -> dict[str, pd.DataFrame]:
        """Fetch all requested timeframes for one symbol."""
        return {tf: self.fetch_ohlcv(symbol, tf, limit=limit) for tf in timeframes}

    def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        since: pd.Timestamp,
        until: pd.Timestamp | None = None,
        page_limit: int = 1500,
        on_page: Callable[[pd.Timestamp, int], None] | None = None,
    ) -> pd.DataFrame:
        """Paginate history from `since` to `until` (default: now).

        Used to build the local backtest store; the exchange caps each
        response at ~1500 candles, so long ranges need many round trips.
        """
        if timeframe not in TF_DELTA:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        unified = validate_symbol(symbol)
        until = until or pd.Timestamp.now(tz="UTC")
        step = TF_DELTA[timeframe]

        pages: list[pd.DataFrame] = []
        cursor = since
        while cursor < until:
            since_ms = int(cursor.timestamp() * 1000)
            raw = self._with_retries(
                f"OHLCV {unified} {timeframe} from {cursor}",
                lambda ms=since_ms: self._exchange.fetch_ohlcv(
                    unified, timeframe=timeframe, since=ms, limit=page_limit
                ),
            )
            if not raw:
                break
            page = self._to_frame(raw, unified, timeframe)
            page = page[page.index <= until]
            if page.empty:
                break
            pages.append(page)
            if on_page is not None:
                on_page(page.index[-1], len(page))
            next_cursor = page.index[-1] + step
            if next_cursor <= cursor:  # exchange returned no forward progress
                break
            cursor = next_cursor

        if not pages:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        out = pd.concat(pages)
        return out[~out.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------------
    # Open Interest
    # ------------------------------------------------------------------
    def fetch_open_interest(
        self, symbol: str, period: str = "15m", limit: int = 500
    ) -> pd.DataFrame:
        """Open Interest history snapshots -> frame with an `oi` column
        (USDT notional; falls back to base amount when notional is absent).

        Binance serves at most ~30 days of history for this endpoint.
        """
        unified = validate_symbol(symbol)
        raw = self._with_retries(
            f"OI {unified} {period}",
            lambda: self._exchange.fetch_open_interest_history(
                unified, timeframe=period, limit=limit
            ),
        )
        if not raw:
            raise DataFetchError(f"Empty OI response for {unified}")
        rows = [
            {
                "timestamp": entry["timestamp"],
                "oi": entry.get("openInterestValue")
                or entry.get("openInterestAmount"),
            }
            for entry in raw
            if entry.get("timestamp") is not None
        ]
        df = pd.DataFrame(rows).dropna()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype(float).sort_index()
        return df[~df.index.duplicated(keep="last")]

    # ------------------------------------------------------------------
    # CVD (Cumulative Volume Delta)
    # ------------------------------------------------------------------
    def fetch_cvd(self, symbol: str, interval: str = "5m", limit: int = 1500) -> pd.DataFrame:
        """Approximate CVD from Binance futures klines' taker-buy volume.

        Per candle: delta = taker_buy - taker_sell = 2*taker_buy - volume.
        5m deltas are resampled to 15m and cumulatively summed. This is the
        best REST-only approximation of aggressive buy/sell flow.
        """
        unified = validate_symbol(symbol)
        raw = self._with_retries(
            f"CVD klines {unified} {interval}",
            lambda: self._exchange.fapiPublicGetKlines(
                {"symbol": market_id(unified), "interval": interval, "limit": limit}
            ),
        )
        if not raw:
            raise DataFetchError(f"Empty kline response for {unified}")
        # Kline row: [openTime, o, h, l, c, volume, closeTime, quoteVol,
        #             trades, takerBuyBase, takerBuyQuote, ignore]
        ts = pd.to_datetime([int(r[0]) for r in raw], unit="ms", utc=True)
        volume = pd.Series([float(r[5]) for r in raw], index=ts)
        taker_buy = pd.Series([float(r[9]) for r in raw], index=ts)
        delta = 2.0 * taker_buy - volume

        delta_15m = delta.resample("15min", label="left", closed="left").sum()
        out = pd.DataFrame({"delta": delta_15m, "cvd": delta_15m.cumsum()})
        out.index.name = "timestamp"
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _with_retries(self, what: str, call: Callable[[], T]) -> T:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return call()
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as exc:
                last_exc = exc
                wait = self._backoff_base * (2**attempt)
                logger.warning(
                    "Rate limited fetching %s (attempt %d/%d), sleeping %.1fs",
                    what, attempt + 1, self._max_retries, wait,
                )
                time.sleep(wait)
            except ccxt.NetworkError as exc:
                last_exc = exc
                wait = self._backoff_base * (2**attempt)
                logger.warning(
                    "Network error fetching %s (attempt %d/%d): %s",
                    what, attempt + 1, self._max_retries, exc,
                )
                time.sleep(wait)
        raise DataFetchError(
            f"Failed to fetch {what} after {self._max_retries} attempts"
        ) from last_exc

    @staticmethod
    def _to_frame(raw: list[list[float]], symbol: str, timeframe: str) -> pd.DataFrame:
        if not raw:
            raise DataFetchError(f"Empty OHLCV response for {symbol} {timeframe}")
        df = pd.DataFrame(raw, columns=["timestamp", *OHLCV_COLUMNS])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype(float).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        logger.debug("Fetched %d %s candles for %s", len(df), timeframe, symbol)
        return df


def drop_open_candle(
    df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Remove the still-forming candle so indicator values never repaint.

    A candle is open when its close time (open time + duration) is in the
    future. Only the last row can be open.
    """
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    if df.index[-1] + TF_DELTA[timeframe] > now:
        return df.iloc[:-1]
    return df
