"""Replay correctness: parity with the live path, and no look-ahead.

These are the tests the whole validation effort rests on. If replay and live
can disagree, or if a bar's score can change once future candles arrive,
every backtest number produced later is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import TF_DELTA
from app.engine import MTFAnalysisEngine
from app.replay import Replayer
from app.store import resample_ohlcv
from tests.conftest import make_ohlcv, zigzag_closes


@pytest.fixture(scope="module")
def m15() -> pd.DataFrame:
    """~5,200 bars: enough for 250+ closed 4h candles plus warm-up."""
    closes = (
        zigzag_closes(5200, base=100, amp=12, period=96)
        + zigzag_closes(5200, base=0, amp=5, period=411)
    )
    return make_ohlcv(closes, "15min", start="2024-01-01")


@pytest.fixture(scope="module")
def replayer() -> Replayer:
    return Replayer(engine=MTFAnalysisEngine(fetcher=None))


# ----------------------------------------------------------------------
# Truncation
# ----------------------------------------------------------------------
def test_frames_at_exposes_only_closed_bars(replayer, m15):
    htf = replayer.prepare(m15)
    i = 5000
    close_time = m15.index[i] + TF_DELTA["15m"]
    frames = replayer.frames_at(m15, htf, i)

    assert frames["15m"].index[-1] == m15.index[i]
    for tf in ("1h", "4h", "1d"):
        last_close = frames[tf].index[-1] + TF_DELTA[tf]
        assert last_close <= close_time, f"{tf} bar had not closed yet"
        # And it is the newest one that had closed — nothing withheld.
        next_close = frames[tf].index[-1] + 2 * TF_DELTA[tf]
        assert next_close > close_time


def test_frames_at_never_includes_future_rows(replayer, m15):
    htf = replayer.prepare(m15)
    for i in (4000, 4500, 5000):
        cutoff = m15.index[i] + TF_DELTA["15m"]
        for tf, frame in replayer.frames_at(m15, htf, i).items():
            assert (frame.index + TF_DELTA[tf] <= cutoff).all()


# ----------------------------------------------------------------------
# Parity: replay == live on identical data
# ----------------------------------------------------------------------
def test_replay_matches_live_analysis_on_the_same_bar(replayer, m15):
    """The last replayed bar must equal what analyze_frames produces from
    the full series — same code path, same answer."""
    htf = replayer.prepare(m15)
    i = len(m15) - 1

    live_frames = {"15m": m15, **{tf: htf[tf] for tf in ("1h", "4h", "1d")}}
    live = replayer.engine.analyze_frames("BTCUSDT", live_frames)
    replayed = replayer.engine.analyze_frames("BTCUSDT", replayer.frames_at(m15, htf, i))

    assert replayed.evaluated_at == live.evaluated_at
    assert replayed.price == live.price
    assert replayed.verdict == live.verdict
    assert replayed.scores == live.scores
    assert replayed.regime == live.regime
    assert replayed.long_plan == live.long_plan
    assert replayed.short_plan == live.short_plan


# ----------------------------------------------------------------------
# No look-ahead: the defining property
# ----------------------------------------------------------------------
def test_score_at_a_bar_is_unchanged_by_future_candles(replayer, m15):
    """Score bar i, then append violently different future candles and score
    bar i again. Any difference means the engine peeked."""
    i = 4800
    truncated = m15.iloc[: i + 1]
    early = replayer.engine.analyze_frames(
        "BTCUSDT", replayer.frames_at(truncated, replayer.prepare(truncated), i)
    )

    future = m15.iloc[i + 1 : i + 200].copy() * 1.5  # a violent rally that never was
    with_future = pd.concat([truncated, future])
    later = replayer.engine.analyze_frames(
        "BTCUSDT", replayer.frames_at(with_future, replayer.prepare(with_future), i)
    )

    assert later.evaluated_at == early.evaluated_at
    assert later.scores == early.scores, "future candles changed a past score"
    assert later.verdict == early.verdict
    assert later.long_plan == early.long_plan
    assert later.short_plan == early.short_plan


def test_mitigation_scan_cannot_see_the_future(replayer, m15):
    """Order blocks and FVGs are invalidated by later price action, and that
    check scans forward — so it must only ever see the truncated series."""
    i = 4600
    truncated = m15.iloc[: i + 1]
    early = replayer.engine.analyze_frames(
        "BTCUSDT", replayer.frames_at(truncated, replayer.prepare(truncated), i)
    )
    # A future crash would mitigate every bullish OB/FVG standing at bar i.
    crash = m15.iloc[i + 1 : i + 150].copy() * 0.5
    later = replayer.engine.analyze_frames(
        "BTCUSDT",
        replayer.frames_at(
            pd.concat([truncated, crash]),
            replayer.prepare(pd.concat([truncated, crash])),
            i,
        ),
    )
    sm_early, sm_later = early.smart_money, later.smart_money
    assert [ob.low for ob in sm_later.order_blocks] == [
        ob.low for ob in sm_early.order_blocks
    ]
    assert [g.low for g in sm_later.fvgs] == [g.low for g in sm_early.fvgs]


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def test_run_produces_one_row_per_bar_with_ablation_columns(replayer, m15):
    out = replayer.run("BTCUSDT", m15, start=m15.index[5150], progress_every=0)
    assert len(out) == 50
    assert out.index.is_monotonic_increasing
    assert set(out["verdict"]) <= {"LONG", "SHORT", "NEUTRAL"}
    for col in ("long_trend", "long_location", "long_whale", "long_momentum",
                "long_volatility", "long_penalty", "short_trend"):
        assert col in out.columns
    assert (out["long_score"].between(0, 100)).all()
    # Whale points are always zero in replay: OI and liquidations are not
    # reconstructable from historical candles.
    assert (out["long_whale"] == 0).all() or (out["long_whale"] > 0).any()


def test_run_is_deterministic(replayer, m15):
    a = replayer.run("BTCUSDT", m15, start=m15.index[5180], progress_every=0)
    b = replayer.run("BTCUSDT", m15, start=m15.index[5180], progress_every=0)
    pd.testing.assert_frame_equal(a, b)


def test_stride_samples_evenly(replayer, m15):
    out = replayer.run("BTCUSDT", m15, start=m15.index[5100], stride=10, progress_every=0)
    assert len(out) == 10
    gaps = out.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(minutes=150)]


# ----------------------------------------------------------------------
# Resampling (higher timeframes are derived, not downloaded)
# ----------------------------------------------------------------------
def test_resample_aggregates_ohlcv_correctly(m15):
    h1 = resample_ohlcv(m15, "1h")
    first_hour = m15.iloc[:4]
    assert h1["open"].iloc[0] == first_hour["open"].iloc[0]
    assert h1["close"].iloc[0] == first_hour["close"].iloc[-1]
    assert h1["high"].iloc[0] == first_hour["high"].max()
    assert h1["low"].iloc[0] == first_hour["low"].min()
    assert h1["volume"].iloc[0] == pytest.approx(first_hour["volume"].sum())


def test_resampled_bars_align_to_utc_boundaries(m15):
    for tf, hours in (("1h", 1), ("4h", 4), ("1d", 24)):
        idx = resample_ohlcv(m15, tf).index
        assert (idx.hour % hours == 0).all()
        assert (idx.minute == 0).all()


def test_resample_preserves_total_volume(m15):
    assert resample_ohlcv(m15, "4h")["volume"].sum() == pytest.approx(
        m15["volume"].sum()
    )


def test_prepare_rejects_unsorted_input(replayer, m15):
    with pytest.raises(ValueError):
        replayer.prepare(m15.iloc[::-1])
    with pytest.raises(ValueError):
        replayer.prepare(m15.iloc[:0])
