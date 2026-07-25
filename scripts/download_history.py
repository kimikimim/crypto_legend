"""Download and validate historical candles for the backtest store.

Usage:
    python scripts/download_history.py --years 3
    python scripts/download_history.py --years 1 --timeframes 15m
    python scripts/download_history.py --validate-only

15m drives the signals; 1m resolves which barrier a signal hit first.
Re-running only fetches what is missing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ALLOWED_SYMBOLS  # noqa: E402
from app.store import OHLCVStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local OHLCV store")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument(
        "--timeframes", nargs="+", default=["15m", "1m"],
        help="default: 15m 1m (1m is needed for intrabar path resolution)",
    )
    parser.add_argument("--symbols", nargs="+", default=list(ALLOWED_SYMBOLS))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = OHLCVStore()
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365.25 * args.years)

    reports = []
    for symbol in args.symbols:
        for tf in args.timeframes:
            if args.validate_only:
                reports.append(store.validate(symbol, tf))
            else:
                print(f"\n=== {symbol} {tf} (from {start.date()}) ===")
                reports.append(store.sync(symbol, tf, start=start, progress=True))

    print("\n" + "=" * 70)
    print("INTEGRITY REPORT")
    print("=" * 70)
    bad = 0
    for report in reports:
        print(report.summary())
        if report.missing_samples:
            print(f"    first missing: {', '.join(report.missing_samples)}")
        bad += 0 if report.ok else 1
    print("=" * 70)
    print(f"{len(reports) - bad}/{len(reports)} series clean")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
