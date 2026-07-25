"""Backtest analytics — above all, is the score informative?

The headline question is not "did it make money" but "does a score of 80
outperform a score of 50". If expectancy is flat across score buckets the
weighting scheme carries no information, and no amount of parameter tuning
will fix that: the features are wrong. Everything else here is secondary.

Sample sizes are reported alongside every statistic because BTC, ETH and SOL
are strongly correlated on intraday horizons — simultaneous signals are much
closer to one observation than to three, so a raw trade count overstates how
much independent evidence exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_SCORE_BINS = (40, 50, 60, 70, 80, 100)


@dataclass(frozen=True)
class PerformanceStats:
    trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    total_r: float
    max_drawdown_r: float
    avg_win_r: float
    avg_loss_r: float
    t_stat: float
    p_value: float

    @property
    def significant(self) -> bool:
        """Expectancy distinguishable from zero at the 5% level. Note this
        ignores the multiple-testing burden of having tuned parameters by
        inspection, so treat it as necessary, not sufficient."""
        return self.p_value < 0.05 and self.expectancy_r > 0

    def as_dict(self) -> dict:
        return {
            "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "profit_factor": round(self.profit_factor, 3),
            "total_r": round(self.total_r, 2),
            "max_drawdown_r": round(self.max_drawdown_r, 2),
            "avg_win_r": round(self.avg_win_r, 3),
            "avg_loss_r": round(self.avg_loss_r, 3),
            "t_stat": round(self.t_stat, 3),
            "p_value": round(self.p_value, 4),
        }


EMPTY_STATS = PerformanceStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def performance(trades: pd.DataFrame) -> PerformanceStats:
    """Headline statistics for a set of labeled trades."""
    if trades.empty:
        return EMPTY_STATS
    r = trades["r_multiple"].to_numpy(dtype=float)
    wins, losses = r[r > 0], r[r <= 0]

    equity = np.cumsum(r)
    drawdown = float(np.max(np.maximum.accumulate(equity) - equity)) if len(r) else 0.0

    if len(r) > 1 and r.std(ddof=1) > 0:
        t_stat, p_value = stats.ttest_1samp(r, 0.0)
    else:
        t_stat, p_value = 0.0, 1.0

    return PerformanceStats(
        trades=len(r),
        win_rate=float(len(wins) / len(r)),
        expectancy_r=float(r.mean()),
        profit_factor=(
            float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
        ),
        total_r=float(r.sum()),
        max_drawdown_r=drawdown,
        avg_win_r=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_r=float(losses.mean()) if len(losses) else 0.0,
        t_stat=float(t_stat),
        p_value=float(p_value),
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Does a higher score actually earn more?"""

    table: pd.DataFrame
    spearman_rho: float
    spearman_p: float
    monotonic: bool
    verdict: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def score_calibration(
    trades: pd.DataFrame,
    bins: tuple[int, ...] = DEFAULT_SCORE_BINS,
    min_bucket: int = 20,
) -> CalibrationResult:
    """Bucket trades by signal score and compare their outcomes."""
    if trades.empty:
        return CalibrationResult(pd.DataFrame(), 0.0, 1.0, False, "NO DATA")

    df = trades.copy()
    df["bucket"] = pd.cut(df["score"], bins=list(bins), right=False)

    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        stats_ = performance(group)
        rows.append(
            {
                "bucket": str(bucket),
                "trades": stats_.trades,
                "win_rate": round(stats_.win_rate, 3),
                "expectancy_r": round(stats_.expectancy_r, 3),
                "total_r": round(stats_.total_r, 1),
                "profit_factor": round(stats_.profit_factor, 2),
            }
        )
    table = pd.DataFrame(rows)

    rho, p_value = (
        stats.spearmanr(df["score"], df["r_multiple"])
        if len(df) > 2
        else (0.0, 1.0)
    )
    rho = float(rho) if not np.isnan(rho) else 0.0
    p_value = float(p_value) if not np.isnan(p_value) else 1.0

    usable = table[table["trades"] >= min_bucket]
    monotonic = (
        len(usable) >= 2 and usable["expectancy_r"].is_monotonic_increasing
    )

    notes: list[str] = []
    thin = table[table["trades"] < min_bucket]
    if not thin.empty:
        notes.append(
            f"{len(thin)} bucket(s) below {min_bucket} trades — treat as noise"
        )
    if len(usable) < 2:
        verdict = "INCONCLUSIVE — not enough populated score buckets"
    elif rho > 0 and p_value < 0.05:
        verdict = "INFORMATIVE — higher scores earn more (rank correlation is significant)"
    elif p_value >= 0.05:
        verdict = "FLAT — score does not predict outcome; the weights carry no information"
    else:
        verdict = "INVERTED — higher scores earn less, which is worse than useless"

    return CalibrationResult(
        table=table,
        spearman_rho=rho,
        spearman_p=p_value,
        monotonic=monotonic,
        verdict=verdict,
        notes=tuple(notes),
    )


def breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """Performance grouped by a column (regime, symbol, side, sl_basis).

    Regime and symbol splits matter most: an aggregate that looks profitable
    can be one good regime carrying three bad ones.
    """
    if trades.empty or by not in trades.columns:
        return pd.DataFrame()
    rows = []
    for key, group in trades.groupby(by, observed=True):
        rows.append({by: key, **performance(group).as_dict()})
    return pd.DataFrame(rows).sort_values("trades", ascending=False)


def barrier_mix(trades: pd.DataFrame) -> pd.DataFrame:
    """How trades ended — a timeout-heavy mix means the targets are unreachable
    within the holding period, not that the signal was wrong."""
    if trades.empty:
        return pd.DataFrame()
    counts = trades["barrier"].value_counts()
    return pd.DataFrame(
        {
            "barrier": counts.index,
            "count": counts.to_numpy(),
            "share": (counts / counts.sum()).round(3).to_numpy(),
        }
    )


def baseline_comparison(
    trades: pd.DataFrame, random_trades: pd.DataFrame
) -> dict:
    """Signal edge versus random entries carrying identical SL/TP geometry.

    If the two are indistinguishable, any apparent edge belongs to the risk
    management, not to the entry logic.
    """
    signal, random_ = performance(trades), performance(random_trades)
    if signal.trades == 0 or random_.trades == 0:
        return {"comparable": False}
    t_stat, p_value = stats.ttest_ind(
        trades["r_multiple"], random_trades["r_multiple"], equal_var=False
    )
    return {
        "comparable": True,
        "signal": signal.as_dict(),
        "random": random_.as_dict(),
        "expectancy_gap_r": round(signal.expectancy_r - random_.expectancy_r, 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "beats_random": bool(p_value < 0.05 and signal.expectancy_r > random_.expectancy_r),
    }
