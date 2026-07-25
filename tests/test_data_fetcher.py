"""DataFetcher: symbol normalization, frame shaping, retry/backoff."""

from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from app.data_fetcher import DataFetcher, normalize_symbol, validate_symbol
from app.exceptions import DataFetchError, UnsupportedSymbolError


def _raw_candles(n: int = 5) -> list[list[float]]:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    out = []
    for i in range(n):
        ts = int((start + pd.Timedelta(minutes=15 * i)).timestamp() * 1000)
        out.append([ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0])
    return out


class FlakyExchange:
    """Raises rate-limit errors N times, then succeeds."""

    def __init__(self, failures: int, exc: type[Exception] = ccxt.RateLimitExceeded):
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe=None, limit=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("simulated")
        return _raw_candles()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.data_fetcher.time.sleep", lambda _s: None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTCUSDT", "BTC/USDT:USDT"),
        ("btcusdt", "BTC/USDT:USDT"),
        ("BTC/USDT", "BTC/USDT:USDT"),
        ("BTC/USDT:USDT", "BTC/USDT:USDT"),
        ("eth-usdt", "ETH/USDT:USDT"),
    ],
)
def test_normalize_symbol(raw, expected):
    assert normalize_symbol(raw) == expected


def test_normalize_symbol_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_symbol("USDT")


def test_fetch_returns_utc_indexed_float_frame():
    fetcher = DataFetcher(exchange=FlakyExchange(failures=0))
    df = fetcher.fetch_ohlcv("BTCUSDT", "15m", limit=5)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert len(df) == 5
    assert df["close"].dtype == float


def test_fetch_retries_through_rate_limits():
    exchange = FlakyExchange(failures=2)
    fetcher = DataFetcher(exchange=exchange, max_retries=4)
    df = fetcher.fetch_ohlcv("BTCUSDT", "15m")
    assert exchange.calls == 3
    assert len(df) == 5


def test_fetch_retries_through_network_errors():
    exchange = FlakyExchange(failures=1, exc=ccxt.NetworkError)
    fetcher = DataFetcher(exchange=exchange, max_retries=3)
    assert len(fetcher.fetch_ohlcv("ETHUSDT", "1h")) == 5


def test_fetch_raises_after_retries_exhausted():
    exchange = FlakyExchange(failures=99)
    fetcher = DataFetcher(exchange=exchange, max_retries=3)
    with pytest.raises(DataFetchError):
        fetcher.fetch_ohlcv("BTCUSDT", "15m")
    assert exchange.calls == 3


def test_fetch_rejects_unknown_timeframe():
    fetcher = DataFetcher(exchange=FlakyExchange(failures=0))
    with pytest.raises(ValueError):
        fetcher.fetch_ohlcv("BTCUSDT", "5m")


# ----------------------------------------------------------------------
# Symbol whitelist (BTC / ETH / SOL only)
# ----------------------------------------------------------------------
def test_whitelist_accepts_exactly_the_three_targets():
    assert validate_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert validate_symbol("eth/usdt") == "ETH/USDT:USDT"
    assert validate_symbol("SOL/USDT:USDT") == "SOL/USDT:USDT"


@pytest.mark.parametrize("symbol", ["XRPUSDT", "DOGEUSDT", "BNB/USDT:USDT"])
def test_whitelist_rejects_everything_else(symbol):
    with pytest.raises(UnsupportedSymbolError):
        validate_symbol(symbol)


def test_fetch_refuses_non_whitelisted_symbol_without_calling_exchange():
    exchange = FlakyExchange(failures=0)
    fetcher = DataFetcher(exchange=exchange)
    with pytest.raises(UnsupportedSymbolError):
        fetcher.fetch_ohlcv("XRPUSDT", "15m")
    assert exchange.calls == 0


# ----------------------------------------------------------------------
# Open Interest
# ----------------------------------------------------------------------
class OiExchange:
    def fetch_open_interest_history(self, symbol, timeframe=None, limit=None):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        return [
            {
                "timestamp": int((start + pd.Timedelta(minutes=15 * i)).timestamp() * 1000),
                "openInterestAmount": 100.0 + i,
                "openInterestValue": 1_000_000.0 + i * 1000,
            }
            for i in range(10)
        ]


def test_fetch_open_interest_prefers_notional_value():
    df = DataFetcher(exchange=OiExchange()).fetch_open_interest("BTCUSDT")
    assert list(df.columns) == ["oi"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["oi"].iloc[0] == 1_000_000.0
    assert df["oi"].iloc[-1] == 1_009_000.0


# ----------------------------------------------------------------------
# CVD from taker-buy klines
# ----------------------------------------------------------------------
class KlineExchange:
    """Six 5m klines: first 15m bucket net buying, second net selling."""

    def fapiPublicGetKlines(self, params):
        assert params["symbol"] == "BTCUSDT"
        start = pd.Timestamp("2026-01-01", tz="UTC")
        rows = []
        for i in range(6):
            ts = int((start + pd.Timedelta(minutes=5 * i)).timestamp() * 1000)
            taker_buy = "7.0" if i < 3 else "3.0"   # vol 10 -> delta +4 / -4
            rows.append([ts, "100", "101", "99", "100.5", "10.0",
                         ts + 299_999, "1000", 50, taker_buy, "700", "0"])
        return rows


def test_fetch_cvd_delta_and_cumsum_arithmetic():
    df = DataFetcher(exchange=KlineExchange()).fetch_cvd("BTCUSDT")
    # delta = 2*taker_buy - volume: 3 x +4 then 3 x -4, in two 15m buckets.
    assert len(df) == 2
    assert df["delta"].iloc[0] == pytest.approx(12.0)
    assert df["delta"].iloc[1] == pytest.approx(-12.0)
    assert df["cvd"].iloc[0] == pytest.approx(12.0)
    assert df["cvd"].iloc[1] == pytest.approx(0.0)
