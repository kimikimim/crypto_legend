"""Look-ahead-safe merging of 15m / 1h / 4h indicator frames.

Higher-timeframe values are keyed by their candle CLOSE time before an
as-of merge onto the 15m candle close times, so a 15m row can only ever
see 4h/1h candles that had fully closed by the end of that 15m candle.
"""

from __future__ import annotations

import pandas as pd

from app.config import TF_DELTA


def merge_mtf(
    m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame
) -> pd.DataFrame:
    """Merge 1h/4h context onto the 15m frame.

    Returns a frame indexed by 15m candle open time; 15m columns keep their
    names, higher-timeframe columns get `_1h` / `_4h` suffixes.
    """
    left = m15.copy()
    left.index.name = "timestamp"
    left["close_time"] = left.index + TF_DELTA["15m"]
    merged = left.reset_index().sort_values("close_time")

    for tf, frame in (("1h", h1), ("4h", h4)):
        right = frame.add_suffix(f"_{tf}").copy()
        right["close_time"] = frame.index + TF_DELTA[tf]
        right = right.reset_index(drop=True).sort_values("close_time")
        merged = pd.merge_asof(
            merged,
            right,
            on="close_time",
            direction="backward",
            allow_exact_matches=True,
        )

    return merged.set_index("timestamp").sort_index()
