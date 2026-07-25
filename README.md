# crypto_legend

Multi-timeframe (1D / 4h / 1h / 15m) **smart-money scoring engine** for
Binance USDT-M Futures. Outputs a deterministic 0–100 entry score for long
and short, a full evidence breakdown, and a structural trade plan
(entry zone / SL / TP1 / TP2 / position sizing).

**Hard whitelist: BTC/USDT, ETH/USDT, SOL/USDT only.** The engine never
scans the broader market.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Tests (fully offline, synthetic data)
.venv/bin/python -m pytest

# Score a symbol against live Binance data (public API, no keys)
.venv/bin/python scripts/run_analysis.py BTCUSDT

# API (MTF_LIQ_WS=0 disables the live liquidation websocket collector)
.venv/bin/uvicorn app.main:app --reload
# GET /api/v1/score/BTCUSDT?use_closed_candle=true
# GET /api/v1/symbols
```

## Architecture

| Module | Responsibility |
|---|---|
| `app/data_fetcher.py` | ccxt Binance USDT-M: OHLCV, Open Interest history, taker-buy klines (CVD); retry/backoff; symbol whitelist; open-candle exclusion |
| `app/indicators.py` | EMA 120/200, Wilder RSI, MACD, Bollinger, Wilder ATR, volume MA, wick filter, momentum cross flags |
| `app/fibonacci.py` | Wick-filtered swing detection, fib retracements, 4h×1h confluence zones |
| `app/smc.py` | Order blocks (BOS + impulse + volume), fair value gaps, liquidity sweeps vs raw breakouts, squeeze detection |
| `app/liquidations.py` | Live @forceOrder stream collector (ccxt.pro) + rolling spike stats + cold-start OI/volume proxy |
| `app/risk.py` | 1D macro regime filter, structural SL/TP planner, constant-account-risk sizing |
| `app/mtf.py` | Look-ahead-safe `merge_asof` of 1h/4h context onto the 15m frame |
| `app/scoring.py` | Pure, deterministic 0–100 scoring with penalties |
| `app/engine.py` | Orchestration (fail-soft on derivatives-data outages) |
| `app/main.py` | FastAPI service |

## Scoring model (per direction, 100 pts)

| Category | Max | Graded sub-scores |
|---|---|---|
| Trend & MTF alignment | 20 | 4h EMA120/200 (6) · 4h close>EMA120 (4) · 1h EMA120/200 (4) · 1h close>EMA120 (3) · 15m close>EMA120 (3) |
| Location & Confluence | 25 | 4h×1h fib confluence zone touch (10, +2 golden pocket) *or* single-TF fib (5) · aligned order block (7) · aligned FVG (5) · reaction candle (3) — capped |
| Whale & Liquidity Validation | 25 | confirmed liquidity sweep within last 3 closed 15m candles (15) · same-side liquidation flush spike (10) |
| Momentum (15m) | 15 | RSI oversold/overbought reclaim (8) else slope (4) · MACD cross (7) else histogram slope (3) |
| Volatility & Volume Anomalies | 15 | volume ≥2× MA20 at S/R (15) · ≥1.5× at S/R (8) · ≥2× away (5) |

**Penalties** (subtracted from the total, floored at 0):

- **Retail breakout −15**: a fresh close *through* a major 4h/1h swing level
  with no wick-back. Chasing breakouts is actively punished.
- **Counter-regime −20**: a long in a 1D bear regime (or short in bull).
  Regime = 1D EMA50/200 alignment + price location → `bull` / `bear` / `chop`.

## Whale manipulation logic

- **Liquidity sweep (stop-hunt)**: a 15m *raw wick* beyond a major 4h/1h swing
  with the close back inside is classified strictly as a Whale Sweep. It
  invalidates breakout logic and scores the **mean-reversion** direction.
- **Liquidations** (Binance has no public REST history for these):
  - warm path: background websocket collector (`@forceOrder` via ccxt.pro)
    with 24h rolling per-side bucket stats; spike = > mean + 2σ.
  - cold path: deterministic proxy — OI drop ≤ −2σ + volume ≥ 2× MA;
    candle color infers the flushed side.
  - Convention: a long-side capitulation flush validates a LONG reversal
    entry; a short squeeze flush validates a SHORT after a high sweep.
- **Squeeze warning** (flag, not points): OI > 2σ above its 24h mean while
  price consolidates (8-bar range < 2× ATR) at a fib/OB confluence POI;
  liquidation spikes are attached as validation evidence.
- **CVD**: approximated from 5m taker-buy kline volume
  (delta = 2×takerBuy − volume), resampled to 15m; reported with
  price/CVD divergence as accumulation/distribution evidence.

## Trade plan output

- `entry_zone` — nearest aligned POI band (zone/OB/FVG) within 1 ATR, else a
  tight band behind price.
- `suggested_sl` — structural invalidation, never a percentage:
  sweep wick extreme → POI band edge → nearest major swing → ATR fallback;
  always buffered by 0.5 × ATR(15m).
- `suggested_tp1` — closest opposing order block or fib level.
- `suggested_tp2` — 1.618 fib extension of the active 1h leg, or the next
  major structural swing.
- `risk_weight` / `suggested_leverage` — constant account risk
  (default 1% per trade, leverage capped at 5×): wider stop ⇒ smaller size.

Headline JSON fields: `long_score`, `short_score`, `is_squeeze_warning`,
`regime`, `primary_direction`, `entry_zone`, `suggested_sl`,
`suggested_tp1`, `suggested_tp2`, `risk_weight` (from the higher-scoring
side), plus full `long_plan` / `short_plan` and category breakdowns.

## Determinism & anti-repaint guarantees

- `use_closed_candle=True` (default) drops any still-forming candle on every
  timeframe (including 1D) before indicators are computed.
- Wicks longer than 3 × ATR are treated as stop-hunt noise when hunting swing
  pivots (body extremes used instead) — but sweep *detection* deliberately
  uses raw wicks, because the wick is the sweep.
- Pivots need `order` bars on both sides; edge-padded pseudo-pivots are
  rejected, so fib levels and structure never repaint.
- 1h/4h values are keyed by candle *close* time before the as-of merge —
  a 15m candle can only see higher-timeframe candles that had fully closed.
- Scoring is a pure function: same snapshot + context ⇒ same score.

All parameters live in `app/config.py` (`EngineConfig`). This software is
for research/education; it is not financial advice.
