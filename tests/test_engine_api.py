"""End-to-end engine runs on synthetic data + FastAPI endpoint contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.engine import MTFAnalysisEngine
from app.exceptions import InsufficientDataError, UnsupportedSymbolError
from app.main import app
from tests.conftest import make_ohlcv, zigzag_closes


class FakeFetcher:
    """Serves deterministic synthetic frames instead of hitting Binance."""

    def __init__(self, periods: int = 400):
        self.periods = periods

    def fetch_ohlcv(self, symbol, timeframe, limit=500):
        freq = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
                "1h": "1h", "4h": "4h", "1d": "1D"}[timeframe]
        period = {"1m": 60, "3m": 60, "5m": 60, "15m": 96,
                  "1h": 60, "4h": 40, "1d": 50}[timeframe]
        return make_ohlcv(
            zigzag_closes(self.periods, base=100, amp=15, period=period), freq
        )

    def fetch_mtf(self, symbol, timeframes, limit=500):
        return {tf: self.fetch_ohlcv(symbol, tf, limit) for tf in timeframes}

    def fetch_open_interest(self, symbol, period="15m", limit=500):
        idx = pd.date_range("2026-01-01", periods=self.periods,
                            freq="15min", tz="UTC")
        oi = 1e9 + 1e6 * np.sin(np.arange(self.periods) / 10.0)
        return pd.DataFrame({"oi": oi}, index=idx)

    def fetch_cvd(self, symbol, interval="5m", limit=1500):
        idx = pd.date_range("2026-01-01", periods=self.periods,
                            freq="15min", tz="UTC")
        delta = 100.0 * np.sin(np.arange(self.periods) / 7.0)
        return pd.DataFrame({"delta": delta, "cvd": delta.cumsum()}, index=idx)


def test_engine_end_to_end_on_synthetic_data():
    engine = MTFAnalysisEngine(fetcher=FakeFetcher())
    result = engine.analyze("BTCUSDT", use_closed_candle=True)
    assert result.symbol == "BTC/USDT:USDT"
    assert 0.0 <= result.scores.long.total <= 100.0
    assert 0.0 <= result.scores.short.total <= 100.0
    assert result.regime in ("bull", "bear", "chop")
    assert isinstance(result.price, float)
    assert isinstance(result.evaluated_at, pd.Timestamp)
    # Trade plans exist and are structurally valid.
    price = result.price
    lp, sp = result.long_plan, result.short_plan
    assert lp is not None and sp is not None
    assert lp.suggested_sl < price < lp.suggested_tp1 < lp.suggested_tp2
    assert sp.suggested_tp2 < sp.suggested_tp1 < price < sp.suggested_sl
    assert 0 < lp.risk_weight <= 1 and 0 < sp.risk_weight <= 1
    # Deterministic: a second run over identical data scores identically.
    again = MTFAnalysisEngine(fetcher=FakeFetcher()).analyze("BTCUSDT")
    assert again.scores == result.scores
    assert again.long_plan == result.long_plan


def test_engine_survives_missing_derivatives_data():
    """OI/CVD/1d fetch failures must degrade gracefully, not crash."""

    class OhlcvOnlyFetcher(FakeFetcher):
        def fetch_ohlcv(self, symbol, timeframe, limit=500):
            if timeframe == "1d":
                raise RuntimeError("daily endpoint down")
            return super().fetch_ohlcv(symbol, timeframe, limit)

        def fetch_open_interest(self, *a, **k):
            raise RuntimeError("OI endpoint down")

        def fetch_cvd(self, *a, **k):
            raise RuntimeError("klines endpoint down")

    result = MTFAnalysisEngine(fetcher=OhlcvOnlyFetcher()).analyze("ETHUSDT")
    assert result.regime == "chop"                      # fail-soft default
    assert result.smart_money.open_interest.value is None
    assert result.smart_money.cvd.last is None
    assert result.smart_money.liquidations.source == "none"
    assert 0.0 <= result.scores.long.total <= 100.0


def test_engine_refuses_insufficient_data():
    engine = MTFAnalysisEngine(fetcher=FakeFetcher(periods=100))
    with pytest.raises(InsufficientDataError):
        engine.analyze("BTCUSDT")


def test_engine_rejects_bad_symbol():
    engine = MTFAnalysisEngine(fetcher=FakeFetcher())
    with pytest.raises(ValueError):
        engine.analyze("USDT")


def test_engine_rejects_non_whitelisted_symbol():
    engine = MTFAnalysisEngine(fetcher=FakeFetcher())
    with pytest.raises(UnsupportedSymbolError):
        engine.analyze("XRPUSDT")


def test_api_score_endpoint_contract():
    with TestClient(app) as client:
        client.app.state.engine = MTFAnalysisEngine(fetcher=FakeFetcher())
        resp = client.get("/api/v1/score/BTCUSDT")
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "BTC/USDT:USDT"
        assert 0 <= body["long_score"] <= 100
        assert 0 <= body["short_score"] <= 100
        assert body["use_closed_candle"] is True

        # Section 6 headline fields.
        assert isinstance(body["is_squeeze_warning"], bool)
        assert body["regime"] in ("bull", "bear", "chop")
        assert body["primary_direction"] in ("long", "short")
        for key in ("entry_zone", "suggested_sl", "suggested_tp1",
                    "suggested_tp2", "risk_weight"):
            assert key in body and body[key] is not None
        assert 0 <= body["risk_weight"] <= 1

        # Smart-money weights: 20 / 25 / 25 / 15 / 15.
        for side in ("long_breakdown", "short_breakdown"):
            breakdown = body[side]
            assert breakdown["trend"]["max_points"] == 20
            assert breakdown["location"]["max_points"] == 25
            assert breakdown["whale"]["max_points"] == 25
            assert breakdown["momentum"]["max_points"] == 15
            assert breakdown["volatility"]["max_points"] == 15
            assert "penalty" in breakdown

        sm = body["smart_money"]
        for key in ("order_blocks", "fvgs", "sweeps", "breakouts",
                    "liquidations", "squeeze", "open_interest", "cvd"):
            assert key in sm
        assert body["long_plan"]["side"] == "long"
        assert body["short_plan"]["side"] == "short"

        # Chart data for the dashboard.
        assert len(body["klines"]) > 0
        kline = body["klines"][-1]
        assert set(kline) == {"time", "open", "high", "low", "close"}
        assert kline["close"] == body["price"]

        # 15m ATR for the dashboard's zone proximity filter.
        assert body["atr_15m"] is not None and body["atr_15m"] > 0

        # Unified structure zones for chart overlays.
        for zone in body["structure_zones"]:
            assert zone["type"] in ("fib", "ob", "fvg", "sr")
            assert zone["sentiment"] in ("bullish", "bearish", "neutral")
            assert zone["min_price"] <= zone["max_price"]
            assert zone["label"]


def test_api_rejects_non_whitelisted_symbol_with_400():
    with TestClient(app) as client:
        client.app.state.engine = MTFAnalysisEngine(fetcher=FakeFetcher())
        resp = client.get("/api/v1/score/XRPUSDT")
        assert resp.status_code == 400
        assert "restricted" in resp.json()["detail"]


def test_api_klines_endpoint_serves_chart_timeframes():
    with TestClient(app) as client:
        client.app.state.engine = MTFAnalysisEngine(fetcher=FakeFetcher())
        for tf in ("1m", "3m", "5m", "15m", "1h", "4h", "1d"):
            resp = client.get(f"/api/v1/klines/BTCUSDT?timeframe={tf}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["symbol"] == "BTC/USDT:USDT"
            assert body["timeframe"] == tf
            assert len(body["klines"]) > 0
            assert set(body["klines"][0]) == {"time", "open", "high", "low", "close"}


def test_api_klines_rejects_bad_timeframe_and_symbol():
    with TestClient(app) as client:
        client.app.state.engine = MTFAnalysisEngine(fetcher=FakeFetcher())
        assert client.get("/api/v1/klines/BTCUSDT?timeframe=2h").status_code == 400
        assert client.get("/api/v1/klines/XRPUSDT?timeframe=5m").status_code == 400


def test_api_ticker_rejects_non_whitelisted_symbol():
    with TestClient(app) as client:
        client.app.state.engine = MTFAnalysisEngine(fetcher=FakeFetcher())
        assert client.get("/api/v1/ticker/XRPUSDT").status_code == 400


def test_api_symbols_endpoint_lists_exactly_three():
    with TestClient(app) as client:
        resp = client.get("/api/v1/symbols")
        assert resp.json() == {
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
        }


def test_api_health():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_api_root_redirects_to_docs():
    with TestClient(app) as client:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/docs"
