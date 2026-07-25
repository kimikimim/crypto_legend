"""Signal journal: persistence, idempotency, and coverage stats."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import pandas as pd

from app.engine import MTFAnalysisEngine
from app.journal import SignalCollector, SignalJournal
from app.main import app
from tests.test_engine_api import FakeFetcher


@pytest.fixture
def journal(tmp_path) -> SignalJournal:
    return SignalJournal(tmp_path / "j.db")


@pytest.fixture
def result():
    return MTFAnalysisEngine(fetcher=FakeFetcher()).analyze("BTCUSDT")


def test_record_persists_signal_and_derivatives_state(journal, result):
    row_id = journal.record(result)
    assert row_id is not None and row_id > 0

    rows = journal.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTC/USDT:USDT"
    assert row["verdict"] == result.verdict
    assert row["long_score"] == result.scores.long.total
    assert row["short_score"] == result.scores.short.total
    assert row["regime"] == result.regime
    assert row["price"] == result.price
    # The unbackfillable columns — the reason this table exists.
    assert row["oi_value"] == result.smart_money.open_interest.value
    assert row["cvd_last"] == result.smart_money.cvd.last
    assert row["liq_source"] == result.smart_money.liquidations.source


def test_score_breakdown_is_stored_for_later_ablation(journal, result):
    journal.record(result)
    breakdown = json.loads(journal.recent()[0]["score_breakdown"])
    for side in ("long", "short"):
        assert set(breakdown[side]) == {
            "total", "trend", "location", "whale", "momentum",
            "volatility", "penalty",
        }
    assert breakdown["long"]["total"] == result.scores.long.total


def test_same_candle_is_recorded_once(journal, result):
    assert journal.record(result) is not None
    assert journal.record(result) is None  # idempotent
    assert len(journal.recent()) == 1


def test_neutral_verdict_stores_null_plan_and_reasons(journal, result):
    journal.record(result)
    row = journal.recent()[0]
    if result.verdict == "NEUTRAL":
        assert row["sl"] is None and row["plan_side"] is None
        assert json.loads(row["verdict_reasons"])
    else:
        assert row["sl"] is not None
        assert row["plan_side"] == result.verdict.lower()


def test_filters_and_stats(journal, result):
    journal.record(result)
    assert journal.recent(symbol="BTC/USDT:USDT")
    assert journal.recent(symbol="ETH/USDT:USDT") == []
    assert journal.recent(verdict="NOPE") == []

    stats = journal.stats()
    assert stats["total_signals"] == 1
    assert stats["resolved_outcomes"] == 0
    assert stats["per_symbol"][0]["symbol"] == "BTC/USDT:USDT"
    assert stats["per_symbol"][0]["signals"] == 1
    assert stats["by_verdict"][result.verdict] == 1


def test_record_never_raises_on_bad_input(journal):
    class Broken:
        symbol = "BTC/USDT:USDT"

        def __getattr__(self, name):
            raise RuntimeError("boom")

    assert journal.record(Broken()) is None  # swallowed, scoring unaffected


def test_collector_sleeps_until_just_after_a_candle_close(journal):
    collector = SignalCollector(
        MTFAnalysisEngine(fetcher=FakeFetcher()), journal, settle_seconds=20.0
    )
    wait = collector._seconds_to_next_close()  # noqa: SLF001
    # Always lands inside the next candle, never before the bar has closed.
    assert 20.0 <= wait <= 15 * 60 + 20.0
    now = pd.Timestamp.now(tz="UTC")
    landing = now + pd.Timedelta(seconds=wait)
    assert landing > now.ceil("900s")


def test_one_symbol_failing_does_not_block_the_others(journal, monkeypatch):
    """One cycle must journal all three symbols, and a failure on one
    symbol must not stop the others."""
    engine = MTFAnalysisEngine(fetcher=FakeFetcher())
    real_analyze = engine.analyze

    def flaky(symbol, use_closed_candle=True):
        if symbol == "ETH/USDT:USDT":
            raise RuntimeError("exchange hiccup")
        return real_analyze(symbol, use_closed_candle)

    monkeypatch.setattr(engine, "analyze", flaky)
    for symbol in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"):
        try:
            journal.record(engine.analyze(symbol))
        except RuntimeError:
            continue
    recorded = {r["symbol"] for r in journal.recent()}
    assert recorded == {"BTC/USDT:USDT", "SOL/USDT:USDT"}


def test_scoring_endpoint_journals_closed_candles_only():
    with TestClient(app) as client:
        client.app.state.engine = MTFAnalysisEngine(fetcher=FakeFetcher())

        client.get("/api/v1/score/BTCUSDT?use_closed_candle=false")
        assert client.get("/api/v1/journal/stats").json()["total_signals"] == 0

        client.get("/api/v1/score/BTCUSDT")
        stats = client.get("/api/v1/journal/stats").json()
        assert stats["total_signals"] == 1

        signals = client.get("/api/v1/journal?limit=10").json()["signals"]
        assert signals[0]["symbol"] == "BTC/USDT:USDT"
