"""Central configuration for the MTF scoring engine.

All tunable parameters live here so the scoring rules stay deterministic
and auditable in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Timeframe -> candle duration. Order matters: higher timeframes first.
TF_DELTA: dict[str, pd.Timedelta] = {
    "1d": pd.Timedelta(days=1),
    "4h": pd.Timedelta(hours=4),
    "1h": pd.Timedelta(hours=1),
    "15m": pd.Timedelta(minutes=15),
}

TIMEFRAMES: tuple[str, ...] = ("4h", "1h", "15m")

# Hard whitelist: the engine fetches and scores ONLY these three pairs.
# Never expand this to full-market scanning.
ALLOWED_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
)


@dataclass(frozen=True)
class EngineConfig:
    """Deterministic parameters for indicators, extrema, and scoring."""

    # --- data ---
    fetch_limit: int = 500          # candles per timeframe (>= ema_slow + buffer)
    min_candles: int = 250          # hard floor below which analysis is refused

    # --- indicators ---
    ema_fast: int = 120
    ema_slow: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    vol_ma_period: int = 20

    # --- wick filtering ---
    # A wick longer than wick_atr_mult * ATR is treated as stop-hunt noise:
    # the candle body extreme is used instead of the raw high/low.
    wick_atr_mult: float = 3.0

    # --- extrema / fibonacci ---
    # scipy argrelextrema `order` (bars required on each side of a pivot).
    extrema_order: dict[str, int] = field(
        default_factory=lambda: {"4h": 6, "1h": 8}
    )
    fib_ratios: tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
    # Ratios considered high-value ("golden pocket") for the confluence bonus.
    golden_ratios: tuple[float, ...] = (0.5, 0.618)

    # --- confluence tolerances (strict ATR mode) ---
    # 4h and 1h fib levels overlap when within overlap_atr_mult * ATR(1h).
    overlap_atr_mult: float = 0.3
    # Price "touches" a zone/level when within touch_atr_mult * ATR(15m).
    touch_atr_mult: float = 0.25

    # --- momentum ---
    cross_lookback: int = 3         # bars a RSI/MACD cross stays "recent"
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # --- volatility ---
    vol_spike_mult: float = 2.0     # volume > 2x MA -> full spike
    vol_soft_mult: float = 1.5      # volume > 1.5x MA -> soft spike

    # --- smart money concepts ---
    smc_scan_bars: int = 200        # how far back OBs/FVGs are hunted
    structure_order: int = 5        # pivot strictness for BOS structure
    ob_lookback: int = 10           # bars back from a BOS to find the OB candle
    ob_impulse_atr_mult: float = 1.2  # impulse candle range >= 1.2x ATR
    ob_vol_mult: float = 1.5        # impulse leg needs volume >= 1.5x MA
    ob_max_count: int = 3           # freshest unmitigated OBs kept per side
    fvg_min_atr_mult: float = 0.3   # ignore gaps smaller than 0.3x ATR
    fvg_max_count: int = 5          # freshest unmitigated FVGs kept per side

    # --- liquidity sweeps / breakouts ---
    sweep_lookback: int = 3         # closed 15m candles a sweep stays "fresh"
    major_pivot_count: int = 3      # recent 4h/1h swing highs+lows tracked
    breakout_penalty: float = 15.0  # subtracted from a fresh raw-breakout side

    # --- open interest / squeeze ---
    oi_period: str = "15m"
    oi_window: int = 96             # 24h of 15m OI snapshots
    oi_z_threshold: float = 2.0     # OI z-score for "unusually high"
    squeeze_consolidation_bars: int = 8
    squeeze_range_atr_mult: float = 2.0   # 8-bar range < 2x ATR = consolidating
    squeeze_poi_atr_mult: float = 1.0     # "near" a POI = within 1x ATR(15m)

    # --- liquidations ---
    liq_window_hours: int = 24
    liq_bucket: str = "15min"
    liq_warm_min_buckets: int = 24  # >= 6h of collected stream before "measured"
    liq_z_threshold: float = 2.0    # bucket notional z-score for a spike

    # --- CVD ---
    cvd_interval: str = "5m"        # kline granularity for taker-buy delta
    cvd_limit: int = 1500           # max Binance klines per request (~5.2 days)
    cvd_div_lookback: int = 12      # 15m buckets for divergence detection

    # --- macro regime filter (1D) ---
    regime_ema_fast: int = 50
    regime_ema_slow: int = 200
    regime_penalty: float = 20.0    # counter-macro-trend signals lose 20 pts
    regime_min_candles: int = 210   # need slow EMA warm-up on the daily

    # --- trade planning (SL/TP) ---
    sl_atr_buffer: float = 0.5      # SL sits 0.5x ATR(15m) beyond the wick
    sl_fallback_atr: float = 1.5    # no structure at all -> 1.5x ATR stop
    tp_min_atr_dist: float = 0.5    # targets closer than this are ignored
    tp1_fallback_atr: float = 2.0
    tp2_fallback_atr: float = 4.0
    fib_extension: float = 1.618    # TP2 extension of the active swing leg

    # --- position sizing ---
    account_risk_pct: float = 1.0   # % of equity risked per trade
    max_leverage: float = 5.0       # hard cap for suggested leverage


DEFAULT_CONFIG = EngineConfig()
