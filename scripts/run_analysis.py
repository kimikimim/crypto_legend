"""CLI smoke test: score a whitelisted symbol against live Binance data.

Usage:
    python scripts/run_analysis.py BTCUSDT
    python scripts/run_analysis.py SOLUSDT --open-candle

Note: the CLI has no websocket stream, so liquidation validation always
uses the OI/volume proxy here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ALLOWED_SYMBOLS  # noqa: E402
from app.engine import MTFAnalysisEngine  # noqa: E402
from app.scoring import DirectionScore  # noqa: E402


def print_direction(name: str, d: DirectionScore) -> None:
    print(f"\n{name} score: {d.total:.1f} / 100")
    for label, cat in (
        ("Trend", d.trend),
        ("Location", d.location),
        ("Whale", d.whale),
        ("Momentum", d.momentum),
        ("Volatility", d.volatility),
    ):
        print(f"  {label:<10} {cat.points:>5.1f} / {cat.max_points:.0f}")
        for reason in cat.reasons:
            print(f"      - {reason}")
    if d.penalty:
        print(f"  Penalty    -{d.penalty:.0f}")
        for reason in d.penalty_reasons:
            print(f"      - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MTF smart-money scoring smoke test")
    parser.add_argument("symbol", help=f"one of: {', '.join(ALLOWED_SYMBOLS)}")
    parser.add_argument(
        "--open-candle",
        action="store_true",
        help="Include the still-forming candle (score may repaint)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    engine = MTFAnalysisEngine()
    result = engine.analyze(args.symbol, use_closed_candle=not args.open_candle)
    sm = result.smart_money

    print(f"\n=== {result.symbol} @ {result.price} "
          f"(candle open {result.evaluated_at}) ===")
    print(f"1D regime: {result.regime}   primary: {result.primary_direction}")
    print_direction("LONG", result.scores.long)
    print_direction("SHORT", result.scores.short)

    print("\n--- Trade plans ---")
    for plan in (result.long_plan, result.short_plan):
        if plan is None:
            continue
        print(f"{plan.side.upper():<5} entry [{plan.entry_zone_low:.6g}, "
              f"{plan.entry_zone_high:.6g}]  SL {plan.suggested_sl:.6g} "
              f"({plan.sl_basis})  TP1 {plan.suggested_tp1:.6g}  "
              f"TP2 {plan.suggested_tp2:.6g}  RR1 {plan.rr_tp1}  "
              f"lev {plan.suggested_leverage}x  weight {plan.risk_weight}")

    print("\n--- Smart money ---")
    if result.zones:
        print("Fib confluence zones (4h x 1h):")
        for z in result.zones:
            print(f"  [{z.low:.6g}, {z.high:.6g}]  {', '.join(z.sources)}")
    for ob in sm.order_blocks:
        print(f"Order block  {ob.side:<8} {ob.timeframe:>3} "
              f"[{ob.low:.6g}, {ob.high:.6g}]  ({ob.time})")
    for g in sm.fvgs:
        print(f"FVG          {g.side:<8} {g.timeframe:>3} "
              f"[{g.low:.6g}, {g.high:.6g}]  ({g.time})")
    for s in sm.sweeps:
        print(f"WHALE SWEEP  {s.side} of {s.level_timeframe} level {s.level:.6g} ({s.time})")
    for b in sm.breakouts:
        print(f"Raw breakout {b.side} through {b.level_timeframe} {b.level:.6g} ({b.time})")

    oi = sm.open_interest
    if oi.value is not None:
        print(f"Open interest: {oi.value:,.0f} USDT (z={oi.z:+.2f}, "
              f"change z={oi.change_z if oi.change_z is None else round(oi.change_z, 2)})")
    cvd = sm.cvd
    if cvd.last is not None:
        div = f", divergence={cvd.divergence}" if cvd.divergence else ""
        print(f"CVD: {cvd.last:,.0f} (last delta {cvd.delta:,.0f}, "
              f"3-bar {cvd.slope3:,.0f}{div})")
    liq = sm.liquidations
    if liq.source != "none":
        print(f"Liquidations [{liq.source}]: long_flush={liq.long_flush} "
              f"short_flush={liq.short_flush} {'; '.join(liq.detail)}")
    if sm.squeeze.active:
        print("\n*** SQUEEZE WARNING ***")
        for reason in sm.squeeze.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
