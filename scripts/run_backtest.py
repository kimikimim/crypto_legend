"""Replay history, label outcomes, and report whether the score is informative.

Usage:
    python scripts/run_backtest.py --symbols BTCUSDT --stride 4
    python scripts/run_backtest.py --start 2024-01-01 --end 2025-01-01

Reads only the local parquet store, so results are reproducible. Signals and
trades are written to data/backtest/ for further analysis.

Caveat carried through every number below: Open Interest, CVD and
liquidations cannot be reconstructed from historical candles, so replay
scores the price-structure subset (75 of 100 points) and the Whale category
is always zero. Validating those features requires the forward-test journal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis import (  # noqa: E402
    barrier_mix,
    breakdown,
    performance,
    score_calibration,
)
from app.backtest import TripleBarrierLabeler  # noqa: E402
from app.config import ALLOWED_SYMBOLS  # noqa: E402
from app.replay import Replayer  # noqa: E402
from app.store import OHLCVStore  # noqa: E402

OUT_DIR = Path("data/backtest")


def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(df.to_string(index=False) if not df.empty else "(no data)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the historical backtest")
    parser.add_argument("--symbols", nargs="+", default=list(ALLOWED_SYMBOLS))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="score every Nth 15m bar (use >1 for a fast first pass)",
    )
    parser.add_argument("--reuse-signals", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    store = OHLCVStore()
    replayer = Replayer()
    labeler = TripleBarrierLabeler()
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None

    all_signals, all_trades = [], []
    for symbol in args.symbols:
        slug = symbol.split("/")[0].replace("USDT", "").lower()
        sig_path = OUT_DIR / f"signals_{slug}.parquet"

        m15 = store.load(symbol, "15m")
        if m15.empty:
            print(f"!! no 15m data stored for {symbol} — run download_history.py")
            continue

        if args.reuse_signals and sig_path.exists():
            signals = pd.read_parquet(sig_path)
            print(f"\n=== {symbol}: reusing {len(signals)} cached signals ===")
        else:
            print(f"\n=== {symbol}: replaying {len(m15):,} 15m bars ===")
            signals = replayer.run(symbol, m15, start=start, end=end, stride=args.stride)
            if signals.empty:
                print("   no bars scored (insufficient history?)")
                continue
            signals.to_parquet(sig_path)

        m1 = store.load(symbol, "1m")
        if m1.empty:
            print(f"!! no 1m data for {symbol} — outcomes cannot be resolved")
            continue

        report = labeler.run(signals, m1)
        print(
            f"   {len(signals):,} bars scored | "
            f"{report.signals_in:,} actionable | "
            f"{report.resolved:,} resolved ({report.coverage:.0%} coverage)"
        )
        all_signals.append(signals)
        if not report.trades.empty:
            all_trades.append(report.trades)

    if not all_trades:
        print("\nNo trades produced. Nothing to evaluate.")
        return

    trades = pd.concat(all_trades).sort_index()
    trades.to_parquet(OUT_DIR / "trades.parquet")
    signals = pd.concat(all_signals).sort_index()

    print("\n" + "=" * 78)
    print("OVERALL PERFORMANCE")
    print("=" * 78)
    overall = performance(trades)
    for key, value in overall.as_dict().items():
        print(f"  {key:<16} {value}")
    print(f"  {'signal rate':<16} {len(trades) / max(len(signals), 1):.2%} of bars")

    print("\n" + "=" * 78)
    print("SCORE CALIBRATION  (the decisive test)")
    print("=" * 78)
    calib = score_calibration(trades)
    _print_table("Expectancy by score bucket", calib.table)
    print(
        f"\n  Spearman rho = {calib.spearman_rho:+.3f} "
        f"(p = {calib.spearman_p:.4f}), monotonic = {calib.monotonic}"
    )
    for note in calib.notes:
        print(f"  note: {note}")
    print(f"\n  >>> {calib.verdict}")

    _print_table("By regime", breakdown(trades, "regime"))
    _print_table("By symbol", breakdown(trades, "symbol"))
    _print_table("By side", breakdown(trades, "side"))
    _print_table("By stop basis", breakdown(trades, "sl_basis"))
    _print_table("Exit mix", barrier_mix(trades))

    print("\n" + "=" * 78)
    print("READ THIS BEFORE BELIEVING ANY OF THE ABOVE")
    print("=" * 78)
    print(
        "  - Whale/liquidity scoring (25 pts) is absent from replay; these are\n"
        "    price-structure results only.\n"
        "  - BTC/ETH/SOL are strongly correlated intraday, so the effective\n"
        "    independent sample is well below the trade count shown.\n"
        "  - Parameters were chosen by inspection, so the p-values carry an\n"
        "    unpriced multiple-testing burden.\n"
        "  - Ties inside a 1m bar are resolved as losses; results are a lower bound."
    )
    print(f"\nWrote {OUT_DIR}/trades.parquet ({len(trades):,} trades)")


if __name__ == "__main__":
    main()
