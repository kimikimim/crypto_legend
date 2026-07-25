# MTF Scoring Engine — Binance USDT-M Futures

Multi-timeframe (4h / 1h / 15m) technical analysis engine that outputs a
deterministic 0–100 entry score for both **long** and **short**, with a full
per-category breakdown.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run the tests (fully offline, synthetic data)
.venv/bin/python -m pytest

# Score a symbol against live Binance Futures data (public API, no keys)
.venv/bin/python scripts/run_analysis.py BTCUSDT

# Run the API
.venv/bin/uvicorn app.main:app --reload
# GET http://127.0.0.1:8000/api/v1/score/BTCUSDT?use_closed_candle=true
```

## Architecture

| Module | Responsibility |
|---|---|
| `app/data_fetcher.py` | `DataFetcher` — ccxt Binance USDT-M OHLCV with retry/backoff on rate limits & network errors; `drop_open_candle` for repaint prevention |
| `app/indicators.py` | `IndicatorCalculator` — EMA 120/200, Wilder RSI, MACD, Bollinger, Wilder ATR, volume MA, wick filter, momentum cross flags |
| `app/fibonacci.py` | Wick-filtered swing detection (scipy extrema), fib retracements, 4h×1h confluence zones |
| `app/mtf.py` | Look-ahead-safe `merge_asof` of 1h/4h context onto the 15m frame |
| `app/scoring.py` | `ScoringEngine` — pure, deterministic 0–100 scoring |
| `app/engine.py` | `MTFAnalysisEngine` — orchestration |
| `app/main.py` | FastAPI service |

## Scoring model (per direction)

| Category | Max | Graded sub-scores |
|---|---|---|
| Trend | 30 | 4h EMA120>200 (9) · 4h close>EMA120 (6) · 1h EMA120>200 (6) · 1h close>EMA120 (4) · 15m close>EMA120 (5) — mirrored for shorts |
| Location & Confluence | 35 | touch of 4h×1h confluence zone (20) *or* single-TF fib level (10) · reaction candle closing beyond the zone (10) · golden-pocket ratio 0.5/0.618 in zone (5) |
| Momentum (15m) | 20 | RSI oversold/overbought reclaim within 3 bars (10) else RSI slope in range (5) · MACD cross within 3 bars (10) else histogram slope (5) |
| Volatility | 15 | volume ≥2× MA20 **at** the S/R zone (15) · ≥1.5× at zone (8) · ≥2× away from zone (5) |

## Determinism & anti-repaint guarantees

- **`use_closed_candle=True`** (default) drops any still-forming candle on
  every timeframe before indicators are computed — scores cannot fluctuate
  intra-candle.
- **Wick filtering**: a wick longer than `3 × ATR` is treated as stop-hunt
  noise; the candle body extreme is used instead when hunting swing pivots,
  so fib grids can't be anchored to a liquidation spike.
- **Confirmed pivots only**: a swing needs `order` bars on *both* sides
  (edge-padded pseudo-pivots are rejected), so fib levels never repaint.
- **No HTF look-ahead**: 1h/4h values are keyed by candle *close* time before
  the as-of merge — a 15m candle can only see higher-timeframe candles that
  had fully closed by its own close.

## Tolerances (strict ATR mode)

- 4h and 1h fib levels form a confluence zone when within `0.3 × ATR(1h)`.
- Price "touches" a zone/level when within `0.25 × ATR(15m)`.

All parameters live in `app/config.py` (`EngineConfig`).
# crypto_legend
