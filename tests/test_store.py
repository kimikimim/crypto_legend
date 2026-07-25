"""OHLCV store: integrity checks, persistence, and resumable sync."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.store import OHLCVStore, validate_ohlcv
from tests.conftest import make_ohlcv


@pytest.fixture
def clean() -> pd.DataFrame:
    return make_ohlcv(np.linspace(100, 110, 200), "15min")


# ----------------------------------------------------------------------
# Integrity checks
# ----------------------------------------------------------------------
def test_clean_series_passes(clean):
    report = validate_ohlcv(clean, "BTC/USDT:USDT", "15m")
    assert report.ok
    assert report.rows == 200
    assert report.missing_bars == 0


def test_missing_bars_detected_and_sampled(clean):
    gapped = clean.drop(clean.index[50:53])
    report = validate_ohlcv(gapped, "BTC/USDT:USDT", "15m")
    assert not report.ok
    assert report.missing_bars == 3
    assert len(report.missing_samples) == 3
    assert any("3 missing bars" in i for i in report.issues)


def test_duplicate_and_unsorted_index_detected(clean):
    dupes = pd.concat([clean, clean.iloc[[10]]])
    report = validate_ohlcv(dupes, "BTC/USDT:USDT", "15m")
    assert report.duplicate_timestamps == 1
    assert report.non_monotonic is True
    assert not report.ok


def test_ohlc_bound_violations_detected(clean):
    broken = clean.copy()
    broken.iloc[5, broken.columns.get_loc("high")] = broken["low"].iloc[5] - 1
    report = validate_ohlcv(broken, "BTC/USDT:USDT", "15m")
    assert report.ohlc_violations >= 1
    assert not report.ok


def test_nan_and_nonpositive_prices_detected(clean):
    dirty = clean.copy()
    dirty.iloc[3, dirty.columns.get_loc("close")] = np.nan
    dirty.iloc[7, dirty.columns.get_loc("open")] = 0.0
    report = validate_ohlcv(dirty, "BTC/USDT:USDT", "15m")
    assert report.nan_rows == 1
    assert report.nonpositive_prices == 1


def test_empty_series_reports_empty():
    report = validate_ohlcv(pd.DataFrame(), "BTC/USDT:USDT", "1m")
    assert not report.ok
    assert report.rows == 0
    assert "empty" in report.issues[0]


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_save_and_load_roundtrip(tmp_path, clean):
    store = OHLCVStore(tmp_path)
    store.save("BTCUSDT", "15m", clean)
    loaded = store.load("BTCUSDT", "15m")
    pd.testing.assert_frame_equal(loaded, clean, check_freq=False)


def test_save_merges_and_dedupes(tmp_path, clean):
    store = OHLCVStore(tmp_path)
    store.save("BTCUSDT", "15m", clean.iloc[:100])
    store.save("BTCUSDT", "15m", clean.iloc[80:])  # overlapping window
    loaded = store.load("BTCUSDT", "15m")
    assert len(loaded) == 200
    assert loaded.index.is_monotonic_increasing
    assert not loaded.index.duplicated().any()


def test_load_respects_time_window(tmp_path, clean):
    store = OHLCVStore(tmp_path)
    store.save("BTCUSDT", "15m", clean)
    window = store.load("BTCUSDT", "15m", start=clean.index[10], end=clean.index[20])
    assert len(window) == 11
    assert window.index[0] == clean.index[10]


def test_missing_file_loads_empty(tmp_path):
    assert OHLCVStore(tmp_path).load("ETHUSDT", "1m").empty


def test_paths_are_per_symbol_and_timeframe(tmp_path):
    store = OHLCVStore(tmp_path)
    assert store.path_for("BTCUSDT", "15m").name == "btc_15m.parquet"
    assert store.path_for("SOL/USDT:USDT", "1m").name == "sol_1m.parquet"


# ----------------------------------------------------------------------
# Sync (resumable download)
# ----------------------------------------------------------------------
class RangeFetcher:
    """Serves a fixed synthetic history and records what was requested."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.requests: list[pd.Timestamp] = []

    def fetch_ohlcv_range(self, symbol, timeframe, since, until=None, **kw):
        self.requests.append(since)
        out = self.df[self.df.index >= since]
        if until is not None:
            out = out[out.index <= until]
        return out


def test_sync_downloads_history_then_stays_quiet_when_current(tmp_path, clean):
    fetcher = RangeFetcher(clean)
    store = OHLCVStore(tmp_path, fetcher=fetcher)
    end = clean.index[-1]

    first = store.sync("BTCUSDT", "15m", start=clean.index[0], end=end)
    assert first.ok and first.rows == 200
    assert fetcher.requests == [clean.index[0]]

    # Nothing new to fetch -> no exchange round trip at all.
    store.sync("BTCUSDT", "15m", start=clean.index[0], end=end)
    assert len(fetcher.requests) == 1


def test_sync_resumes_from_the_stored_tail(tmp_path, clean):
    """A partially built store must fetch only the missing tail."""
    fetcher = RangeFetcher(clean)
    store = OHLCVStore(tmp_path, fetcher=fetcher)
    store.save("BTCUSDT", "15m", clean.iloc[:150])

    report = store.sync("BTCUSDT", "15m", start=clean.index[0], end=clean.index[-1])
    assert fetcher.requests == [clean.index[149] + pd.Timedelta(minutes=15)]
    assert report.ok and report.rows == 200


def test_sync_reports_gaps_in_downloaded_data(tmp_path, clean):
    gapped = clean.drop(clean.index[100:105])
    store = OHLCVStore(tmp_path, fetcher=RangeFetcher(gapped))
    report = store.sync("BTCUSDT", "15m", start=clean.index[0], end=clean.index[-1])
    assert not report.ok
    assert report.missing_bars == 5
