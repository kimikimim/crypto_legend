"""Local OHLCV store: Parquet files plus integrity checks.

A backtest is only as trustworthy as its data. Fetching from the exchange on
every run makes results irreproducible and silently tolerates gaps, so
history is downloaded once, validated, and pinned to disk. 1m candles are
stored alongside 15m because resolving whether a stop or a target was hit
first inside a 15m bar requires the finer series — assuming the favourable
order is the classic way backtests end up overstated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.config import TF_DELTA
from app.data_fetcher import OHLCV_COLUMNS, DataFetcher, validate_symbol

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path("data/ohlcv")


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a fine series into a coarser one (15m -> 1h/4h/1d).

    Replay derives every higher timeframe from the stored 15m candles rather
    than downloading each separately: the series are then guaranteed mutually
    consistent, and one archive covers all of them. Binance's 1h/4h/1d bars
    are anchored to UTC midnight, which is also pandas' default origin here.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        df.resample(TF_DELTA[timeframe], label="left", closed="left")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    out.index.name = "timestamp"
    return out


@dataclass(frozen=True)
class IntegrityReport:
    """What is wrong with a stored series, if anything."""

    symbol: str
    timeframe: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    missing_bars: int
    missing_samples: tuple[str, ...] = ()
    duplicate_timestamps: int = 0
    non_monotonic: bool = False
    nan_rows: int = 0
    ohlc_violations: int = 0
    nonpositive_prices: int = 0
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        span = f"{self.first} .. {self.last}" if self.rows else "empty"
        head = f"{self.symbol} {self.timeframe}: {self.rows} rows, {span}"
        return head if self.ok else f"{head}\n  - " + "\n  - ".join(self.issues)


def validate_ohlcv(
    df: pd.DataFrame, symbol: str, timeframe: str, max_samples: int = 5
) -> IntegrityReport:
    """Check a series for the defects that quietly corrupt backtests."""
    issues: list[str] = []
    if df.empty:
        return IntegrityReport(
            symbol=symbol, timeframe=timeframe, rows=0, first=None, last=None,
            missing_bars=0, issues=("series is empty",),
        )

    step = TF_DELTA[timeframe]
    duplicates = int(df.index.duplicated().sum())
    non_monotonic = not df.index.is_monotonic_increasing

    expected = pd.date_range(df.index[0], df.index[-1], freq=step, tz="UTC")
    missing = expected.difference(df.index)
    nan_rows = int(df[OHLCV_COLUMNS].isna().any(axis=1).sum())

    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    violations = int(
        (
            (df["high"] < body_high - 1e-9)
            | (df["low"] > body_low + 1e-9)
            | (df["high"] < df["low"])
        ).sum()
    )
    nonpositive = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())

    if duplicates:
        issues.append(f"{duplicates} duplicate timestamps")
    if non_monotonic:
        issues.append("index is not sorted")
    if len(missing):
        issues.append(f"{len(missing)} missing bars")
    if nan_rows:
        issues.append(f"{nan_rows} rows contain NaN")
    if violations:
        issues.append(f"{violations} rows violate OHLC bounds")
    if nonpositive:
        issues.append(f"{nonpositive} rows have non-positive prices")

    return IntegrityReport(
        symbol=symbol,
        timeframe=timeframe,
        rows=len(df),
        first=df.index[0],
        last=df.index[-1],
        missing_bars=len(missing),
        missing_samples=tuple(str(ts) for ts in missing[:max_samples]),
        duplicate_timestamps=duplicates,
        non_monotonic=non_monotonic,
        nan_rows=nan_rows,
        ohlc_violations=violations,
        nonpositive_prices=nonpositive,
        issues=tuple(issues),
    )


class OHLCVStore:
    """Parquet-backed candle archive, one file per symbol and timeframe."""

    def __init__(
        self,
        root: Path | str = DEFAULT_STORE_DIR,
        fetcher: DataFetcher | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher

    @property
    def fetcher(self) -> DataFetcher:
        if self._fetcher is None:
            self._fetcher = DataFetcher()
        return self._fetcher

    def path_for(self, symbol: str, timeframe: str) -> Path:
        slug = validate_symbol(symbol).split("/")[0].lower()
        return self.root / f"{slug}_{timeframe}.parquet"

    # ------------------------------------------------------------------
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        """Merge into the existing file; later rows win on conflict."""
        path = self.path_for(symbol, timeframe)
        merged = df
        if path.exists():
            merged = pd.concat([pd.read_parquet(path), df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "timestamp"
        merged.to_parquet(path)
        logger.info("Stored %d %s %s candles -> %s", len(merged), symbol, timeframe, path)
        return path

    # ------------------------------------------------------------------
    def sync(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
        progress: bool = False,
    ) -> IntegrityReport:
        """Download anything missing between `start` and now, then validate.

        Resumable: an existing file means only the tail is fetched.
        """
        unified = validate_symbol(symbol)
        end = end or pd.Timestamp.now(tz="UTC")
        existing = self.load(unified, timeframe)
        cursor = start
        if not existing.empty:
            cursor = max(start, existing.index[-1] + TF_DELTA[timeframe])

        if cursor < end:
            def _log(last: pd.Timestamp, n: int) -> None:
                if progress:
                    logger.info("  %s %s: +%d bars through %s", unified, timeframe, n, last)

            fresh = self.fetcher.fetch_ohlcv_range(
                unified, timeframe, since=cursor, until=end,
                on_page=_log if progress else None,
            )
            if not fresh.empty:
                self.save(unified, timeframe, fresh)
        else:
            logger.info("%s %s already current", unified, timeframe)

        return self.validate(unified, timeframe)

    def validate(self, symbol: str, timeframe: str) -> IntegrityReport:
        return validate_ohlcv(self.load(symbol, timeframe), validate_symbol(symbol), timeframe)
