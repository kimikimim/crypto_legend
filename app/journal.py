"""Signal journal: an append-only record of every scored evaluation.

Why this exists: the engine's own validation needs data that cannot be
backfilled. Binance serves roughly 30 days of Open Interest history and no
public liquidation history at all, so the Whale & Liquidity features (25 of
100 points) can never be reconstructed for a long backtest. Recording them
live, starting now, is the only way to ever validate them.

It also provides the forward-test ledger: signals captured in real time,
paired later with measured outcomes, which is the honest gate before real
capital. Storage is stdlib sqlite3 (no new dependency), WAL mode, one row
per (symbol, candle).

Only closed-candle evaluations are journaled. Those are deterministic and
non-repainting, so a row is a permanent fact about that candle; an open
candle would record a value that no longer exists a minute later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from app.config import ALLOWED_SYMBOLS
from app.engine import AnalysisResult, MTFAnalysisEngine

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/journal.db")


def resolve_db_path(path: Path | str | None = None) -> Path:
    """Explicit path, else $MTF_JOURNAL_DB, else the default location."""
    return Path(path or os.environ.get("MTF_JOURNAL_DB") or DEFAULT_DB_PATH)

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    evaluated_at    TEXT NOT NULL,      -- scored candle open time, ISO UTC
    recorded_at     TEXT NOT NULL,      -- when this row was written
    price           REAL NOT NULL,
    verdict         TEXT NOT NULL,      -- LONG | SHORT | NEUTRAL
    long_score      REAL NOT NULL,
    short_score     REAL NOT NULL,
    regime          TEXT NOT NULL,
    atr_15m         REAL,
    -- recommended plan (NULL when NEUTRAL)
    plan_side       TEXT,
    entry_low       REAL,
    entry_high      REAL,
    sl              REAL,
    tp1             REAL,
    tp2             REAL,
    leverage        REAL,
    risk_weight     REAL,
    rr_tp1          REAL,
    rr_tp1_net      REAL,
    sl_basis        TEXT,
    -- derivatives state (unreconstructable later — the reason this table exists)
    oi_value        REAL,
    oi_z            REAL,
    oi_change_z     REAL,
    cvd_last        REAL,
    cvd_delta       REAL,
    cvd_divergence  TEXT,
    liq_source      TEXT,
    liq_long_flush  INTEGER,
    liq_short_flush INTEGER,
    squeeze_active  INTEGER,
    -- structure counts
    sweep_count     INTEGER,
    breakout_count  INTEGER,
    ob_count        INTEGER,
    fvg_count       INTEGER,
    fib_zone_count  INTEGER,
    -- per-category points for later ablation, plus rejection reasons
    score_breakdown TEXT NOT NULL,
    verdict_reasons TEXT NOT NULL,
    UNIQUE(symbol, evaluated_at)
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
    ON signals(symbol, evaluated_at);
CREATE INDEX IF NOT EXISTS idx_signals_verdict ON signals(verdict);

-- Populated by the outcome resolver (triple-barrier labeling).
CREATE TABLE IF NOT EXISTS outcomes (
    signal_id   INTEGER PRIMARY KEY REFERENCES signals(id),
    resolved_at TEXT NOT NULL,
    barrier     TEXT NOT NULL,      -- tp1 | tp2 | sl | timeout
    exit_price  REAL NOT NULL,
    pnl_pct     REAL NOT NULL,      -- net of fees and slippage
    bars_held   INTEGER NOT NULL
);
"""


class SignalJournal:
    """Append-only signal store. Never raises into the scoring path."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = resolve_db_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.info("Signal journal ready at %s", self.path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def record(self, result: AnalysisResult) -> int | None:
        """Persist one evaluation. Returns the row id, or None if the candle
        was already journaled or the write failed."""
        try:
            row = self._to_row(result)
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO signals ({', '.join(row)}) "
                    f"VALUES ({', '.join(':' + k for k in row)})",
                    row,
                )
                if cur.rowcount == 0:
                    return None  # same candle already recorded
                logger.info(
                    "Journaled %s %s verdict=%s",
                    result.symbol, result.evaluated_at, result.verdict,
                )
                return int(cur.lastrowid or 0)
        except Exception as exc:  # noqa: BLE001 — journaling must never break scoring
            logger.warning("Failed to journal signal: %s", exc)
            return None

    @staticmethod
    def _to_row(result: AnalysisResult) -> dict[str, Any]:
        sm = result.smart_money
        plan = result.primary_plan  # None when NEUTRAL
        scores = result.scores
        breakdown = {
            side: {
                "total": d.total,
                "trend": d.trend.points,
                "location": d.location.points,
                "whale": d.whale.points,
                "momentum": d.momentum.points,
                "volatility": d.volatility.points,
                "penalty": d.penalty,
            }
            for side, d in (("long", scores.long), ("short", scores.short))
        }
        return {
            "symbol": result.symbol,
            "evaluated_at": result.evaluated_at.isoformat(),
            "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "price": result.price,
            "verdict": result.verdict,
            "long_score": scores.long.total,
            "short_score": scores.short.total,
            "regime": result.regime,
            "atr_15m": result.atr_15m,
            "plan_side": plan.side if plan else None,
            "entry_low": plan.entry_zone_low if plan else None,
            "entry_high": plan.entry_zone_high if plan else None,
            "sl": plan.suggested_sl if plan else None,
            "tp1": plan.suggested_tp1 if plan else None,
            "tp2": plan.suggested_tp2 if plan else None,
            "leverage": plan.suggested_leverage if plan else None,
            "risk_weight": plan.risk_weight if plan else None,
            "rr_tp1": plan.rr_tp1 if plan else None,
            "rr_tp1_net": plan.rr_tp1_net if plan else None,
            "sl_basis": plan.sl_basis if plan else None,
            "oi_value": sm.open_interest.value,
            "oi_z": sm.open_interest.z,
            "oi_change_z": sm.open_interest.change_z,
            "cvd_last": sm.cvd.last,
            "cvd_delta": sm.cvd.delta,
            "cvd_divergence": sm.cvd.divergence,
            "liq_source": sm.liquidations.source,
            "liq_long_flush": int(sm.liquidations.long_flush),
            "liq_short_flush": int(sm.liquidations.short_flush),
            "squeeze_active": int(sm.squeeze.active),
            "sweep_count": len(sm.sweeps),
            "breakout_count": len(sm.breakouts),
            "ob_count": len(sm.order_blocks),
            "fvg_count": len(sm.fvgs),
            "fib_zone_count": len(result.zones),
            "score_breakdown": json.dumps(breakdown),
            "verdict_reasons": json.dumps(list(result.verdict_reasons)),
        }

    # ------------------------------------------------------------------
    def recent(
        self,
        symbol: str | None = None,
        verdict: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses, params = [], {}
        if symbol:
            clauses.append("symbol = :symbol")
            params["symbol"] = symbol
        if verdict:
            clauses.append("verdict = :verdict")
            params["verdict"] = verdict
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params["limit"] = limit
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM signals {where} "
                f"ORDER BY evaluated_at DESC LIMIT :limit",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Coverage summary — how much forward-test data exists so far."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM signals").fetchone()["n"]
            by_verdict = {
                r["verdict"]: r["n"]
                for r in conn.execute(
                    "SELECT verdict, COUNT(*) AS n FROM signals GROUP BY verdict"
                )
            }
            per_symbol = [
                dict(r)
                for r in conn.execute(
                    "SELECT symbol, COUNT(*) AS signals, "
                    "MIN(evaluated_at) AS first_candle, "
                    "MAX(evaluated_at) AS last_candle "
                    "FROM signals GROUP BY symbol ORDER BY symbol"
                )
            ]
            measured_liq = conn.execute(
                "SELECT COUNT(*) AS n FROM signals WHERE liq_source = 'measured'"
            ).fetchone()["n"]
            resolved = conn.execute("SELECT COUNT(*) AS n FROM outcomes").fetchone()["n"]
        return {
            "total_signals": total,
            "by_verdict": by_verdict,
            "per_symbol": per_symbol,
            "measured_liquidation_rows": measured_liq,
            "resolved_outcomes": resolved,
        }


class SignalCollector:
    """Scores every whitelisted symbol shortly after each 15m candle closes.

    Without this the journal only holds candles that someone happened to
    request, leaving gaps in the forward-test record. Runs on the candle
    boundary so each closed bar is captured exactly once.
    """

    def __init__(
        self,
        engine: MTFAnalysisEngine,
        journal: SignalJournal,
        symbols: tuple[str, ...] = ALLOWED_SYMBOLS,
        interval: pd.Timedelta = pd.Timedelta(minutes=15),
        settle_seconds: float = 20.0,
    ) -> None:
        self.engine = engine
        self.journal = journal
        self.symbols = symbols
        self.interval = interval
        self.settle_seconds = settle_seconds
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="signal-collector")
        logger.info(
            "Signal collector started (%s candles, %s)",
            self.interval, ", ".join(self.symbols),
        )

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._seconds_to_next_close())
                for symbol in self.symbols:
                    try:
                        result = await asyncio.to_thread(
                            self.engine.analyze, symbol, True
                        )
                        await asyncio.to_thread(self.journal.record, result)
                    except Exception as exc:  # noqa: BLE001 — one symbol failing
                        logger.warning("Collector failed for %s: %s", symbol, exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never kill the loop
                logger.warning("Collector cycle error: %s", exc)
                await asyncio.sleep(60)

    def _seconds_to_next_close(self) -> float:
        """Sleep until just after the next candle boundary, so the exchange
        has published the closed bar."""
        now = pd.Timestamp.now(tz="UTC")
        freq = f"{int(self.interval.total_seconds())}s"
        next_close = now.ceil(freq)
        if (next_close - now).total_seconds() < self.settle_seconds:
            next_close = next_close + self.interval
        return (next_close - now).total_seconds() + self.settle_seconds

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        logger.info("Signal collector stopped")
