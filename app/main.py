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
from app.config import ALLOWED_SYMBOLS
from app.engine import MTFAnalysisEngine
from app.exceptions import (
    DataFetchError,
    InsufficientDataError,
    UnsupportedSymbolError,
)
from app.liquidations import LiquidationCollector, LiquidationTracker
from app.models import ScoreResponse, to_response

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

    app.state.engine = MTFAnalysisEngine(liquidation_tracker=tracker)
    logger.info("MTF analysis engine initialized for %s", ", ".join(ALLOWED_SYMBOLS))
    yield
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
        return to_response(result)

    return app


app = create_app()
