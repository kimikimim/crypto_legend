"""Pydantic response models for the API layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.engine import AnalysisResult
from app.risk import TradePlan
from app.scoring import CategoryScore, DirectionScore


class CategoryScoreModel(BaseModel):
    points: float
    max_points: float
    reasons: list[str]


class DirectionScoreModel(BaseModel):
    total: float = Field(ge=0, le=100)
    trend: CategoryScoreModel
    location: CategoryScoreModel
    whale: CategoryScoreModel
    momentum: CategoryScoreModel
    volatility: CategoryScoreModel
    penalty: float
    penalty_reasons: list[str]


class ZoneModel(BaseModel):
    low: float
    high: float
    center: float
    ratios: list[float]
    sources: list[str]


class FibLevelModel(BaseModel):
    price: float
    ratio: float
    timeframe: str


class OrderBlockModel(BaseModel):
    low: float
    high: float
    side: str
    timeframe: str
    time: datetime


class FvgModel(BaseModel):
    low: float
    high: float
    side: str
    timeframe: str
    time: datetime


class SweepModel(BaseModel):
    level: float
    level_timeframe: str
    side: str
    time: datetime


class BreakoutModel(BaseModel):
    level: float
    level_timeframe: str
    side: str
    time: datetime


class LiquidationSignalModel(BaseModel):
    long_flush: bool
    short_flush: bool
    source: str
    detail: list[str]


class SqueezeModel(BaseModel):
    active: bool
    oi_z: float | None
    reasons: list[str]


class OpenInterestModel(BaseModel):
    value: float | None
    z: float | None
    change_z: float | None


class CvdModel(BaseModel):
    last: float | None
    delta: float | None
    slope3: float | None
    divergence: str | None


class SmartMoneyModel(BaseModel):
    order_blocks: list[OrderBlockModel]
    fvgs: list[FvgModel]
    sweeps: list[SweepModel]
    breakouts: list[BreakoutModel]
    liquidations: LiquidationSignalModel
    squeeze: SqueezeModel
    open_interest: OpenInterestModel
    cvd: CvdModel


class EntryZoneModel(BaseModel):
    low: float
    high: float


class KlineModel(BaseModel):
    time: str          # ISO 8601, UTC candle open time
    open: float
    high: float
    low: float
    close: float


class StructureZoneModel(BaseModel):
    """Unified chart zone: fib confluence, order block, or fair value gap."""

    type: str          # "fib" | "ob" | "fvg"
    sentiment: str     # "bullish" | "bearish" | "neutral"
    min_price: float
    max_price: float
    label: str


class TradePlanModel(BaseModel):
    side: str
    entry_zone: EntryZoneModel
    suggested_sl: float
    suggested_tp1: float
    suggested_tp2: float
    risk_weight: float = Field(ge=0, le=1)
    suggested_leverage: float
    rr_tp1: float
    sl_basis: str


class ScoreResponse(BaseModel):
    symbol: str
    evaluated_at: datetime
    price: float
    use_closed_candle: bool
    long_score: float = Field(ge=0, le=100)
    short_score: float = Field(ge=0, le=100)
    # --- headline risk fields (from the higher-scoring direction) ---
    is_squeeze_warning: bool
    regime: str
    primary_direction: str
    entry_zone: EntryZoneModel | None
    suggested_sl: float | None
    suggested_tp1: float | None
    suggested_tp2: float | None
    risk_weight: float | None
    # --- full detail ---
    long_plan: TradePlanModel | None
    short_plan: TradePlanModel | None
    long_breakdown: DirectionScoreModel
    short_breakdown: DirectionScoreModel
    confluence_zones: list[ZoneModel]
    fib_levels_4h: list[FibLevelModel]
    fib_levels_1h: list[FibLevelModel]
    smart_money: SmartMoneyModel
    klines: list[KlineModel]
    structure_zones: list[StructureZoneModel]


def _structure_zones(result: AnalysisResult) -> list[StructureZoneModel]:
    zones: list[StructureZoneModel] = [
        StructureZoneModel(
            type="fib",
            sentiment="neutral",
            min_price=z.low,
            max_price=z.high,
            label=f"FIB {' x '.join(z.sources)}",
        )
        for z in result.zones
    ]
    zones += [
        StructureZoneModel(
            type="ob",
            sentiment=ob.side,
            min_price=ob.low,
            max_price=ob.high,
            label=f"OB {ob.timeframe} {ob.side}",
        )
        for ob in result.smart_money.order_blocks
    ]
    zones += [
        StructureZoneModel(
            type="fvg",
            sentiment=g.side,
            min_price=g.low,
            max_price=g.high,
            label=f"FVG {g.timeframe} {g.side}",
        )
        for g in result.smart_money.fvgs
    ]
    return zones


def _category(c: CategoryScore) -> CategoryScoreModel:
    return CategoryScoreModel(
        points=c.points, max_points=c.max_points, reasons=list(c.reasons)
    )


def _direction(d: DirectionScore) -> DirectionScoreModel:
    return DirectionScoreModel(
        total=d.total,
        trend=_category(d.trend),
        location=_category(d.location),
        whale=_category(d.whale),
        momentum=_category(d.momentum),
        volatility=_category(d.volatility),
        penalty=d.penalty,
        penalty_reasons=list(d.penalty_reasons),
    )


def _plan(p: TradePlan | None) -> TradePlanModel | None:
    if p is None:
        return None
    return TradePlanModel(
        side=p.side,
        entry_zone=EntryZoneModel(low=p.entry_zone_low, high=p.entry_zone_high),
        suggested_sl=p.suggested_sl,
        suggested_tp1=p.suggested_tp1,
        suggested_tp2=p.suggested_tp2,
        risk_weight=p.risk_weight,
        suggested_leverage=p.suggested_leverage,
        rr_tp1=p.rr_tp1,
        sl_basis=p.sl_basis,
    )


def to_response(result: AnalysisResult) -> ScoreResponse:
    sm = result.smart_money
    primary = _plan(result.primary_plan)
    return ScoreResponse(
        symbol=result.symbol,
        evaluated_at=result.evaluated_at.to_pydatetime(),
        price=result.price,
        use_closed_candle=result.use_closed_candle,
        long_score=result.scores.long.total,
        short_score=result.scores.short.total,
        is_squeeze_warning=sm.squeeze.active,
        regime=result.regime,
        primary_direction=result.primary_direction,
        entry_zone=primary.entry_zone if primary else None,
        suggested_sl=primary.suggested_sl if primary else None,
        suggested_tp1=primary.suggested_tp1 if primary else None,
        suggested_tp2=primary.suggested_tp2 if primary else None,
        risk_weight=primary.risk_weight if primary else None,
        long_plan=_plan(result.long_plan),
        short_plan=_plan(result.short_plan),
        klines=[KlineModel(**k) for k in result.klines],
        structure_zones=_structure_zones(result),
        long_breakdown=_direction(result.scores.long),
        short_breakdown=_direction(result.scores.short),
        confluence_zones=[
            ZoneModel(
                low=z.low,
                high=z.high,
                center=z.center,
                ratios=list(z.ratios),
                sources=list(z.sources),
            )
            for z in result.zones
        ],
        fib_levels_4h=[
            FibLevelModel(price=l.price, ratio=l.ratio, timeframe=l.timeframe)
            for l in result.levels_4h
        ],
        fib_levels_1h=[
            FibLevelModel(price=l.price, ratio=l.ratio, timeframe=l.timeframe)
            for l in result.levels_1h
        ],
        smart_money=SmartMoneyModel(
            order_blocks=[
                OrderBlockModel(
                    low=ob.low, high=ob.high, side=ob.side,
                    timeframe=ob.timeframe, time=ob.time.to_pydatetime(),
                )
                for ob in sm.order_blocks
            ],
            fvgs=[
                FvgModel(
                    low=g.low, high=g.high, side=g.side,
                    timeframe=g.timeframe, time=g.time.to_pydatetime(),
                )
                for g in sm.fvgs
            ],
            sweeps=[
                SweepModel(
                    level=s.level, level_timeframe=s.level_timeframe,
                    side=s.side, time=s.time.to_pydatetime(),
                )
                for s in sm.sweeps
            ],
            breakouts=[
                BreakoutModel(
                    level=b.level, level_timeframe=b.level_timeframe,
                    side=b.side, time=b.time.to_pydatetime(),
                )
                for b in sm.breakouts
            ],
            liquidations=LiquidationSignalModel(
                long_flush=sm.liquidations.long_flush,
                short_flush=sm.liquidations.short_flush,
                source=sm.liquidations.source,
                detail=list(sm.liquidations.detail),
            ),
            squeeze=SqueezeModel(
                active=sm.squeeze.active,
                oi_z=sm.squeeze.oi_z,
                reasons=list(sm.squeeze.reasons),
            ),
            open_interest=OpenInterestModel(
                value=sm.open_interest.value,
                z=sm.open_interest.z,
                change_z=sm.open_interest.change_z,
            ),
            cvd=CvdModel(
                last=sm.cvd.last,
                delta=sm.cvd.delta,
                slope3=sm.cvd.slope3,
                divergence=sm.cvd.divergence,
            ),
        ),
    )
