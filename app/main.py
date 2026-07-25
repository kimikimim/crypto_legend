"""FastAPI service exposing the MTF + smart-money scoring engine.

Run:  MTF_LIQ_WS=1 uvicorn app.main:app --reload

MTF_LIQ_WS=1 (default) starts the background Binance @forceOrder websocket
collector; set MTF_LIQ_WS=0 to disable it (the engine then uses the OI/volume
proxy for liquidation validation).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import ccxt
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app import __version__
from app.config import ALLOWED_SYMBOLS, CHART_TIMEFRAMES
from app.data_fetcher import drop_open_candle, validate_symbol
from app.engine import MTFAnalysisEngine
from app.exceptions import (
    DataFetchError,
    InsufficientDataError,
    UnsupportedSymbolError,
)
from app.journal import SignalCollector, SignalJournal
from app.liquidations import LiquidationCollector, LiquidationTracker
from app.models import KlineModel, KlinesResponse, ScoreResponse, to_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracker = LiquidationTracker()
    collector: LiquidationCollector | None = None
    if os.environ.get("MTF_LIQ_WS", "1") != "0":
        collector = LiquidationCollector(tracker)
        await collector.start()
    else:
        logger.info("Liquidation websocket disabled (MTF_LIQ_WS=0) — proxy mode")

    engine = MTFAnalysisEngine(liquidation_tracker=tracker)
    journal = SignalJournal()
    app.state.engine = engine
    app.state.journal = journal

    signal_collector: SignalCollector | None = None
    if os.environ.get("MTF_COLLECTOR", "1") != "0":
        signal_collector = SignalCollector(engine, journal)
        await signal_collector.start()
    else:
        logger.info("Signal collector disabled (MTF_COLLECTOR=0)")

    logger.info("MTF analysis engine initialized for %s", ", ".join(ALLOWED_SYMBOLS))
    yield
    if signal_collector is not None:
        await signal_collector.stop()
    if collector is not None:
        await collector.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MTF Smart-Money Scoring Engine",
        description=(
            "Multi-timeframe (4h/1h/15m) technical + smart-money analysis "
            "and 0-100 entry scoring for Binance USDT-M Futures "
            "(BTC, ETH, SOL only)."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # Local dashboard dev servers only (any localhost port — Vite hops to
    # 5174+ when 5173 is busy). Never expose remote origins here.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/symbols")
    async def symbols() -> dict[str, list[str]]:
        return {"symbols": list(ALLOWED_SYMBOLS)}

    @app.get("/api/v1/score/{symbol}", response_model=ScoreResponse)
    async def score(
        request: Request,
        symbol: str,
        use_closed_candle: bool = Query(
            True,
            description=(
                "Exclude the still-forming candle so the score cannot repaint."
            ),
        ),
    ) -> ScoreResponse:
        engine: MTFAnalysisEngine = request.app.state.engine
        try:
            # ccxt calls are blocking; keep the event loop free.
            result = await asyncio.to_thread(
                engine.analyze, symbol, use_closed_candle
            )
        except (ValueError, UnsupportedSymbolError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ccxt.BadSymbol as exc:
            raise HTTPException(status_code=400, detail=f"Unknown symbol: {exc}") from exc
        except ccxt.RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail="Exchange rate limit hit") from exc
        except InsufficientDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (DataFetchError, ccxt.NetworkError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        # Forward-test ledger. Closed candles only: those are deterministic,
        # so a row is a permanent fact rather than a snapshot of a moving bar.
        journal: SignalJournal | None = getattr(request.app.state, "journal", None)
        if journal is not None and use_closed_candle:
            await asyncio.to_thread(journal.record, result)

        return to_response(result)

    @app.get("/api/v1/journal")
    async def journal_list(
        request: Request,
        symbol: str | None = None,
        verdict: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict:
        journal: SignalJournal | None = getattr(request.app.state, "journal", None)
        if journal is None:
            raise HTTPException(status_code=503, detail="Journal unavailable")
        return {"signals": await asyncio.to_thread(journal.recent, symbol, verdict, limit)}

    @app.get("/api/v1/journal/stats")
    async def journal_stats(request: Request) -> dict:
        journal: SignalJournal | None = getattr(request.app.state, "journal", None)
        if journal is None:
            raise HTTPException(status_code=503, detail="Journal unavailable")
        return await asyncio.to_thread(journal.stats)

    @app.get("/api/v1/ticker/{symbol}")
    async def ticker(request: Request, symbol: str) -> dict:
        """Live last price + 24h change (display only — scoring stays
        closed-candle based). Used as fallback when the browser cannot
        reach Binance's websocket directly."""
        try:
            unified = validate_symbol(symbol)
        except (ValueError, UnsupportedSymbolError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        engine: MTFAnalysisEngine = request.app.state.engine
        try:
            t = await asyncio.to_thread(
                engine.fetcher._exchange.fetch_ticker, unified  # noqa: SLF001
            )
        except ccxt.RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail="Exchange rate limit hit") from exc
        except (ccxt.NetworkError, Exception) as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "symbol": unified,
            "last": t.get("last"),
            "change_24h_pct": t.get("percentage"),
        }

    @app.get("/api/v1/klines/{symbol}", response_model=KlinesResponse)
    async def klines(
        request: Request,
        symbol: str,
        timeframe: str = Query("15m", description=f"One of: {', '.join(CHART_TIMEFRAMES)}"),
        limit: int = Query(400, ge=50, le=1000),
        use_closed_candle: bool = Query(True),
    ) -> KlinesResponse:
        """Chart-only candles for any supported timeframe (no scoring)."""
        if timeframe not in CHART_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"timeframe must be one of {', '.join(CHART_TIMEFRAMES)}",
            )
        try:
            unified = validate_symbol(symbol)
        except (ValueError, UnsupportedSymbolError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        engine: MTFAnalysisEngine = request.app.state.engine

        def _fetch():
            df = engine.fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
            return drop_open_candle(df, timeframe) if use_closed_candle else df

        try:
            df = await asyncio.to_thread(_fetch)
        except (ValueError, UnsupportedSymbolError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ccxt.RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail="Exchange rate limit hit") from exc
        except (DataFetchError, ccxt.NetworkError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return KlinesResponse(
            symbol=unified,
            timeframe=timeframe,
            klines=[
                KlineModel(
                    time=ts.isoformat(),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                )
                for ts, r in df.iterrows()
            ],
        )

    return app


app = create_app()
