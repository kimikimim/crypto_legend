import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import {
  CandlestickSeries,
  ColorType,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Crosshair,
  Gauge,
  Minus,
  RefreshCw,
  ShieldCheck,
  Zap,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Backend schema (actual FastAPI response, subset used by the dashboard)
// ---------------------------------------------------------------------------
interface BackendPlan {
  side: 'long' | 'short'
  entry_zone: { low: number; high: number }
  suggested_sl: number
  suggested_tp1: number
  suggested_tp2: number
  risk_weight: number
  suggested_leverage: number
  rr_tp1: number
  rr_tp1_net: number
  sl_basis: string
  tradeable: boolean
  cost_pct: number
  stressed_entry: boolean
  time_stop_at: string | null
  reject_reasons: string[]
}

interface BackendKline {
  time: string
  open: number
  high: number
  low: number
  close: number
}

interface StructureZone {
  type: 'fib' | 'ob' | 'fvg' | 'sr'
  sentiment: 'bullish' | 'bearish' | 'neutral'
  min_price: number
  max_price: number
  label: string
}

interface BackendResponse {
  symbol: string
  price: number
  regime: 'bull' | 'bear' | 'chop'
  is_squeeze_warning: boolean
  verdict: 'LONG' | 'SHORT' | 'NEUTRAL'
  verdict_reasons: string[]
  primary_direction: 'long' | 'short'
  long_score: number
  short_score: number
  long_plan: BackendPlan | null
  short_plan: BackendPlan | null
  klines: BackendKline[]
  structure_zones: StructureZone[]
  atr_15m: number | null
  evaluated_at: string
}

// ---------------------------------------------------------------------------
// View model
// ---------------------------------------------------------------------------
interface TradePlan {
  direction: 'LONG' | 'SHORT' | 'NEUTRAL'
  score: number
  entry_price: number
  entry_zone: { low: number; high: number } | null
  sl: number
  tp1: number
  tp2: number
  leverage: number
  risk_weight: number
  rr_net: number
  sl_basis: string
  cost_pct: number
  stressed_entry: boolean
  time_stop_at: string | null
  reject_reasons: string[]
}

interface SignalData {
  symbol: string
  price: number
  regime: 'bull' | 'bear' | 'chop'
  is_squeeze_warning: boolean
  long_score: number
  short_score: number
  active_plan: TradePlan
  klines: BackendKline[]
  structure_zones: StructureZone[]
  atr_15m: number | null
  evaluated_at: string
}

type ZoneType = StructureZone['type']
type ZoneVisibility = Record<ZoneType, boolean>

/** ATR proximity filter: keep only zones overlapping price ± mult × ATR. */
function filterZonesByAtr(
  zones: StructureZone[],
  price: number,
  atr: number | null,
  mult = 2,
): StructureZone[] {
  if (atr == null || atr <= 0) return zones
  const lo = price - mult * atr
  const hi = price + mult * atr
  return zones.filter((z) => z.max_price >= lo && z.min_price <= hi)
}

const NEUTRAL_PLAN: TradePlan = {
  direction: 'NEUTRAL',
  score: 0,
  entry_price: 0,
  entry_zone: null,
  sl: 0,
  tp1: 0,
  tp2: 0,
  leverage: 0,
  risk_weight: 0,
  rr_net: 0,
  sl_basis: '',
  cost_pct: 0,
  stressed_entry: false,
  time_stop_at: null,
  reject_reasons: [],
}

function toSignalData(r: BackendResponse): SignalData {
  const bestScore = Math.max(r.long_score, r.short_score)
  // The engine owns the verdict (score threshold + net-RR gate) so the
  // dashboard and any backtest agree on what counts as a trade.
  const plan = r.verdict === 'LONG' ? r.long_plan : r.verdict === 'SHORT' ? r.short_plan : null

  const active_plan: TradePlan =
    !plan || r.verdict === 'NEUTRAL'
      ? { ...NEUTRAL_PLAN, score: bestScore, reject_reasons: r.verdict_reasons ?? [] }
      : {
          direction: r.verdict,
          score: plan.side === 'long' ? r.long_score : r.short_score,
          entry_price: (plan.entry_zone.low + plan.entry_zone.high) / 2,
          entry_zone: plan.entry_zone,
          sl: plan.suggested_sl,
          tp1: plan.suggested_tp1,
          tp2: plan.suggested_tp2,
          leverage: plan.suggested_leverage,
          risk_weight: plan.risk_weight,
          rr_net: plan.rr_tp1_net,
          sl_basis: plan.sl_basis,
          cost_pct: plan.cost_pct,
          stressed_entry: plan.stressed_entry,
          time_stop_at: plan.time_stop_at,
          reject_reasons: [],
        }

  return {
    symbol: r.symbol,
    price: r.price,
    regime: r.regime,
    is_squeeze_warning: r.is_squeeze_warning,
    long_score: r.long_score,
    short_score: r.short_score,
    active_plan,
    klines: r.klines,
    structure_zones: r.structure_zones ?? [],
    atr_15m: r.atr_15m,
    evaluated_at: r.evaluated_at,
  }
}

/** Line styling for a structure zone's min/max boundary lines. */
function zoneLineStyle(zone: StructureZone): {
  color: string
  lineStyle: LineStyle
  axisLabelVisible: boolean
} {
  if (zone.type === 'fib') {
    return { color: '#22d3ee', lineStyle: LineStyle.Dotted, axisLabelVisible: true }
  }
  if (zone.type === 'sr') {
    // Support / resistance swing levels: solid, clearly visible.
    return {
      color: zone.sentiment === 'bullish' ? '#4ade80' : '#f87171',
      lineStyle: LineStyle.Solid,
      axisLabelVisible: true,
    }
  }
  return {
    color:
      zone.sentiment === 'bullish' ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
    lineStyle: LineStyle.SparseDotted,
    axisLabelVisible: false,
  }
}

/** Translucent fill for zone bands (the red/blue shaded areas). */
function zoneBandFill(zone: StructureZone): string {
  if (zone.type === 'fib') return 'rgba(34, 211, 238, 0.07)'
  return zone.sentiment === 'bullish'
    ? 'rgba(59, 130, 246, 0.16)' // blue = support-side zones
    : 'rgba(239, 68, 68, 0.14)'  // red  = resistance-side zones
}

// ---------------------------------------------------------------------------
// Zone band primitive: lightweight-charts has no native rectangles, so we
// paint translucent full-width bands on the pane canvas, under the candles.
// ---------------------------------------------------------------------------
interface BitmapScope {
  context: CanvasRenderingContext2D
  bitmapSize: { width: number; height: number }
  verticalPixelRatio: number
}

interface RenderTarget {
  useBitmapCoordinateSpace(fn: (scope: BitmapScope) => void): void
}

class ZoneBandsPrimitive {
  private zones: StructureZone[] = []
  private series: ISeriesApi<'Candlestick'> | null = null
  private requestUpdate: (() => void) | null = null

  attached(param: { series: unknown; requestUpdate: () => void }): void {
    this.series = param.series as ISeriesApi<'Candlestick'>
    this.requestUpdate = param.requestUpdate
  }

  detached(): void {
    this.series = null
    this.requestUpdate = null
  }

  setZones(zones: StructureZone[]): void {
    this.zones = zones.filter((z) => z.max_price > z.min_price)
    this.requestUpdate?.()
  }

  paneViews() {
    return [
      {
        zOrder: () => 'bottom' as const,
        renderer: () => ({
          draw: (target: RenderTarget) => {
            const series = this.series
            if (!series) return
            target.useBitmapCoordinateSpace(
              ({ context, bitmapSize, verticalPixelRatio }: BitmapScope) => {
                for (const zone of this.zones) {
                  const yTop = series.priceToCoordinate(zone.max_price)
                  const yBottom = series.priceToCoordinate(zone.min_price)
                  if (yTop == null || yBottom == null) continue
                  const top = Math.min(yTop, yBottom) * verticalPixelRatio
                  const height =
                    Math.abs(yBottom - yTop) * verticalPixelRatio || 1
                  context.fillStyle = zoneBandFill(zone)
                  context.fillRect(0, top, bitmapSize.width, height)
                }
              },
            )
          },
        }),
      },
    ]
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] as const
type Symbol_ = (typeof SYMBOLS)[number]

const CHART_TFS = ['1m', '3m', '5m', '15m', '1h', '4h', '1d'] as const
type ChartTf = (typeof CHART_TFS)[number]

const API_BASE = 'http://localhost:8000/api/v1'
const REFRESH_MS = 60_000

const fmt = (value: number): string =>
  value >= 1000
    ? value.toLocaleString('en-US', { maximumFractionDigits: 1 })
    : value.toLocaleString('en-US', { maximumFractionDigits: 4 })

// ---------------------------------------------------------------------------
// Live price ticker (Binance futures websocket — display only; scoring and
// candles stay closed-candle based)
// ---------------------------------------------------------------------------
function useLivePrice(symbol: Symbol_): {
  price: number | null
  change24h: number | null
  tickDir: 'up' | 'down' | null
} {
  const [price, setPrice] = useState<number | null>(null)
  const [change24h, setChange24h] = useState<number | null>(null)
  const [tickDir, setTickDir] = useState<'up' | 'down' | null>(null)
  const lastRef = useRef<number | null>(null)
  const lastMsgAtRef = useRef(0)

  useEffect(() => {
    setPrice(null)
    setChange24h(null)
    setTickDir(null)
    lastRef.current = null
    lastMsgAtRef.current = 0

    let ws: WebSocket | null = null
    let retryTimer: number | undefined
    let disposed = false

    const applyTick = (last: number, changePct: number | null) => {
      if (!Number.isFinite(last)) return
      if (lastRef.current != null && last !== lastRef.current) {
        setTickDir(last > lastRef.current ? 'up' : 'down')
      }
      lastRef.current = last
      setPrice(last)
      setChange24h(changePct)
    }

    const connect = () => {
      ws = new WebSocket(
        `wss://fstream.binance.com/ws/${symbol.toLowerCase()}@miniTicker`,
      )
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data as string) as { c: string; o: string }
          const last = parseFloat(m.c)
          const open24 = parseFloat(m.o)
          lastMsgAtRef.current = Date.now()
          applyTick(last, open24 > 0 ? ((last - open24) / open24) * 100 : null)
        } catch {
          /* ignore malformed frames */
        }
      }
      ws.onclose = () => {
        if (!disposed) retryTimer = window.setTimeout(connect, 3000)
      }
      ws.onerror = () => ws?.close()
    }
    connect()

    // REST fallback: if the websocket stays silent (blocked network,
    // proxy, etc.), poll the backend ticker proxy every 3s instead.
    const poll = window.setInterval(async () => {
      if (Date.now() - lastMsgAtRef.current < 4000) return
      try {
        const resp = await axios.get<{ last: number | null; change_24h_pct: number | null }>(
          `${API_BASE}/ticker/${symbol}`,
          { timeout: 5000 },
        )
        if (typeof resp.data.last === 'number') {
          applyTick(resp.data.last, resp.data.change_24h_pct)
        }
      } catch {
        /* backend down — the score payload price remains as fallback */
      }
    }, 3000)

    return () => {
      disposed = true
      window.clearTimeout(retryTimer)
      window.clearInterval(poll)
      ws?.close()
    }
  }, [symbol])

  return { price, change24h, tickDir }
}

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------
function PriceChart({
  data,
  klines,
  zones,
}: {
  data: SignalData
  klines: BackendKline[]
  zones: StructureZone[]
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const priceLinesRef = useRef<IPriceLine[]>([])
  const bandsRef = useRef<ZoneBandsPrimitive | null>(null)
  const barCountRef = useRef(0)

  /** Default viewport: the most recent ~120 candles, anchored right. */
  const applyDefaultRange = useCallback(() => {
    const n = barCountRef.current
    if (n > 0) {
      chartRef.current?.timeScale().setVisibleLogicalRange({
        from: Math.max(n - 120, 0),
        to: n + 3,
      })
    }
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const chart = createChart(el, {
      width: el.clientWidth || 600,
      height: el.clientHeight || 440,
      layout: {
        background: { type: ColorType.Solid, color: '#0c0c0f' },
        textColor: '#a1a1aa',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#18181b' },
        horzLines: { color: '#18181b' },
      },
      rightPriceScale: { borderColor: '#27272a' },
      timeScale: { borderColor: '#27272a', timeVisible: true, secondsVisible: false },
      crosshair: {
        horzLine: { labelBackgroundColor: '#3f3f46' },
        vertLine: { labelBackgroundColor: '#3f3f46' },
      },
    })
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#16a34a',
      wickDownColor: '#dc2626',
    })
    chartRef.current = chart
    seriesRef.current = series

    const bands = new ZoneBandsPrimitive()
    series.attachPrimitive(bands as Parameters<typeof series.attachPrimitive>[0])
    bandsRef.current = bands

    const observer = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        chart.applyOptions({ width: rect.width, height: rect.height })
        applyDefaultRange()
      }
    })
    observer.observe(el)

    return () => {
      observer.disconnect()
      priceLinesRef.current = []
      bandsRef.current = null
      seriesRef.current = null
      chartRef.current = null
      chart.remove()
    }
  }, [applyDefaultRange])

  useEffect(() => {
    const series = seriesRef.current
    if (!series) return

    series.setData(
      klines.map((k) => ({
        time: Math.floor(Date.parse(k.time) / 1000) as UTCTimestamp,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
      })),
    )
    barCountRef.current = klines.length
    applyDefaultRange()

    // Clear every previously drawn line (symbol switch / refresh).
    priceLinesRef.current.forEach((line) => series.removePriceLine(line))
    priceLinesRef.current = []

    // Shaded red/blue bands for real (non-zero-height) zones.
    bandsRef.current?.setZones(zones)

    // Boundary / S-R lines (pre-filtered by ATR proximity + type toggles).
    for (const zone of zones) {
      const style = zoneLineStyle(zone)
      priceLinesRef.current.push(
        series.createPriceLine({
          price: zone.max_price,
          color: style.color,
          lineStyle: style.lineStyle,
          lineWidth: 1,
          axisLabelVisible: style.axisLabelVisible,
          title: zone.label,
        }),
      )
      if (zone.min_price < zone.max_price) {
        priceLinesRef.current.push(
          series.createPriceLine({
            price: zone.min_price,
            color: style.color,
            lineStyle: style.lineStyle,
            lineWidth: 1,
            axisLabelVisible: style.axisLabelVisible,
            title: '',
          }),
        )
      }
    }

    // Execution levels on top.
    const plan = data.active_plan
    if (plan.direction !== 'NEUTRAL') {
      priceLinesRef.current.push(
        series.createPriceLine({
          price: plan.sl,
          color: '#ef4444',
          lineStyle: LineStyle.Dashed,
          lineWidth: 2,
          title: 'SL',
        }),
        series.createPriceLine({
          price: plan.tp1,
          color: '#22c55e',
          lineStyle: LineStyle.Solid,
          lineWidth: 2,
          title: 'TP1',
        }),
        series.createPriceLine({
          price: plan.tp2,
          color: '#15803d',
          lineStyle: LineStyle.Solid,
          lineWidth: 2,
          title: 'TP2',
        }),
        series.createPriceLine({
          price: plan.entry_price,
          color: '#38bdf8',
          lineStyle: LineStyle.Dotted,
          lineWidth: 1,
          title: 'ENTRY',
        }),
      )
    }
  }, [data, klines, zones, applyDefaultRange])

  return (
    <div className="relative h-full min-h-[420px]">
      <div ref={containerRef} className="absolute inset-0" />
      {data.is_squeeze_warning && (
        <div className="absolute right-3 top-3 z-10 flex items-center gap-1.5 rounded-md border border-amber-400/50 bg-amber-500/15 px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wider text-amber-300 shadow-lg shadow-amber-900/30">
          <Zap size={14} className="animate-pulse" />
          Squeeze warning
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Right-column widgets
// ---------------------------------------------------------------------------
const ZONE_TOGGLE_META: Record<ZoneType, { label: string; dot: string }> = {
  fib: { label: 'FIB', dot: 'bg-cyan-400' },
  ob: { label: 'OB', dot: 'bg-emerald-400' },
  fvg: { label: 'FVG', dot: 'bg-red-400' },
  sr: { label: 'S/R', dot: 'bg-amber-400' },
}

function ZoneToggles({
  visibility,
  counts,
  onToggle,
}: {
  visibility: ZoneVisibility
  counts: Record<ZoneType, number>
  onToggle: (type: ZoneType) => void
}) {
  return (
    <div className="flex items-center gap-1.5">
      {(Object.keys(ZONE_TOGGLE_META) as ZoneType[]).map((type) => {
        const active = visibility[type]
        const meta = ZONE_TOGGLE_META[type]
        return (
          <button
            key={type}
            onClick={() => onToggle(type)}
            title={`Show/hide ${meta.label} zones`}
            className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-[10px] font-bold uppercase tracking-widest transition-colors ${
              active
                ? 'border-zinc-600 bg-zinc-800 text-zinc-100'
                : 'border-zinc-800 bg-zinc-900/40 text-zinc-600 hover:text-zinc-400'
            }`}
          >
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                active ? meta.dot : 'bg-zinc-600'
              }`}
            />
            {meta.label}
            <span className="font-mono tabular-nums opacity-70">{counts[type]}</span>
          </button>
        )
      })}
    </div>
  )
}

function RegimeBadge({ regime }: { regime: SignalData['regime'] }) {
  const styles = {
    bull: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400',
    bear: 'border-red-500/40 bg-red-500/10 text-red-400',
    chop: 'border-zinc-500/40 bg-zinc-500/10 text-zinc-400',
  }[regime]
  const Icon = { bull: ArrowUpRight, bear: ArrowDownRight, chop: Minus }[regime]
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-bold uppercase tracking-widest ${styles}`}
    >
      <Icon size={13} />
      {regime}
    </span>
  )
}

function ScoreBar({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone: 'long' | 'short'
}) {
  const barColor = tone === 'long' ? 'bg-emerald-500' : 'bg-red-500'
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="font-medium uppercase tracking-wider text-zinc-400">{label}</span>
        <span className="font-mono text-sm font-semibold tabular-nums text-zinc-100">
          {value.toFixed(1)}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full transition-all duration-700 ${barColor}`}
          style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }}
        />
      </div>
    </div>
  )
}

function Row({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-center justify-between border-b border-zinc-800/60 py-2 last:border-0">
      <span className="text-xs uppercase tracking-wider text-zinc-500">{label}</span>
      <span className={`font-mono text-sm font-semibold tabular-nums ${accent ?? 'text-zinc-100'}`}>
        {value}
      </span>
    </div>
  )
}

function PlanPanel({ plan }: { plan: TradePlan }) {
  if (plan.direction === 'NEUTRAL') {
    return (
      <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-zinc-700 bg-zinc-900/40 px-4 py-6 text-center">
        <ShieldCheck size={22} className="text-zinc-500" />
        <p className="text-sm font-medium text-zinc-300">No active high-probability setup</p>
        <p className="text-xs text-zinc-500">Best score {plan.score.toFixed(1)} — standing aside.</p>
        {plan.reject_reasons.length > 0 && (
          <ul className="mt-1 space-y-1 text-left text-[11px] leading-snug text-zinc-600">
            {plan.reject_reasons.map((reason) => (
              <li key={reason}>· {reason}</li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  const isLong = plan.direction === 'LONG'
  return (
    <div className="space-y-4">
      <div
        className={`flex items-center justify-between rounded-lg border px-3 py-2 ${
          isLong
            ? 'border-emerald-500/40 bg-emerald-500/10'
            : 'border-red-500/40 bg-red-500/10'
        }`}
      >
        <span
          className={`flex items-center gap-1.5 text-sm font-bold tracking-wider ${
            isLong ? 'text-emerald-400' : 'text-red-400'
          }`}
        >
          {isLong ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
          {plan.direction}
        </span>
        <span className="font-mono text-sm font-semibold tabular-nums text-zinc-200">
          score {plan.score.toFixed(1)}
        </span>
      </div>

      <div>
        <h3 className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-zinc-400">
          <Crosshair size={13} /> Trade setup
        </h3>
        <Row
          label="Entry zone"
          value={
            plan.entry_zone
              ? `${fmt(plan.entry_zone.low)} – ${fmt(plan.entry_zone.high)}`
              : fmt(plan.entry_price)
          }
          accent="text-sky-400"
        />
        <Row label="Stop loss" value={fmt(plan.sl)} accent="text-red-400" />
        <Row label="Take profit 1" value={fmt(plan.tp1)} accent="text-emerald-400" />
        <Row label="Take profit 2" value={fmt(plan.tp2)} accent="text-emerald-600" />
        <Row
          label="Net RR (TP1)"
          value={`${plan.rr_net.toFixed(2)} : 1`}
          accent={plan.rr_net >= 2 ? 'text-emerald-400' : 'text-amber-300'}
        />
        <Row
          label={plan.stressed_entry ? 'Cost (thin book)' : 'Round-trip cost'}
          value={`${plan.cost_pct.toFixed(2)}%`}
          accent={plan.stressed_entry ? 'text-orange-400' : 'text-zinc-400'}
        />
        {plan.time_stop_at && (
          <Row
            label="Time stop"
            value={new Date(plan.time_stop_at).toLocaleString()}
            accent="text-zinc-400"
          />
        )}
        <Row label="SL basis" value={plan.sl_basis.replace('_', ' ')} accent="text-zinc-400" />
      </div>

      <div>
        <h3 className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-zinc-400">
          <Gauge size={13} /> Risk
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-center">
            <div className="font-mono text-xl font-bold tabular-nums text-amber-300">
              {plan.leverage.toFixed(2)}x
            </div>
            <div className="mt-0.5 text-[10px] uppercase tracking-widest text-zinc-500">
              Rec. leverage
            </div>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-center">
            <div className="font-mono text-xl font-bold tabular-nums text-amber-300">
              {(plan.risk_weight * 100).toFixed(1)}%
            </div>
            <div className="mt-0.5 text-[10px] uppercase tracking-widest text-zinc-500">
              Risk weight
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  const [symbol, setSymbol] = useState<Symbol_>('BTCUSDT')
  const [chartTf, setChartTf] = useState<ChartTf>('15m')
  const [data, setData] = useState<SignalData | null>(null)
  const [tfKlines, setTfKlines] = useState<BackendKline[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const live = useLivePrice(symbol)
  const [zoneVisibility, setZoneVisibility] = useState<ZoneVisibility>({
    fib: true,
    ob: true,
    fvg: true,
    sr: true,
  })

  // Default noise control: only zones overlapping price ± 2 × ATR(15m).
  const nearbyZones = useMemo(
    () =>
      data ? filterZonesByAtr(data.structure_zones, data.price, data.atr_15m) : [],
    [data],
  )
  const zoneCounts = useMemo(() => {
    const counts: Record<ZoneType, number> = { fib: 0, ob: 0, fvg: 0, sr: 0 }
    nearbyZones.forEach((z) => (counts[z.type] += 1))
    return counts
  }, [nearbyZones])
  const visibleZones = useMemo(
    () => nearbyZones.filter((z) => zoneVisibility[z.type]),
    [nearbyZones, zoneVisibility],
  )
  const toggleZoneType = useCallback(
    (type: ZoneType) =>
      setZoneVisibility((v) => ({ ...v, [type]: !v[type] })),
    [],
  )

  const fetchSignal = useCallback(async (sym: Symbol_) => {
    setLoading(true)
    setError(null)
    try {
      const resp = await axios.get<BackendResponse>(`${API_BASE}/score/${sym}`, {
        timeout: 30_000,
      })
      setData(toSignalData(resp.data))
      setUpdatedAt(new Date())
    } catch (err) {
      setError(axios.isAxiosError(err) ? err.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchSignal(symbol)
    const id = setInterval(() => void fetchSignal(symbol), REFRESH_MS)
    return () => clearInterval(id)
  }, [symbol, fetchSignal])

  // Chart candles for non-15m timeframes (15m reuses the score payload).
  const fetchTfKlines = useCallback(async (sym: Symbol_, tf: ChartTf) => {
    if (tf === '15m') {
      setTfKlines(null)
      return
    }
    try {
      const resp = await axios.get<{ klines: BackendKline[] }>(
        `${API_BASE}/klines/${sym}`,
        { params: { timeframe: tf }, timeout: 30_000 },
      )
      setTfKlines(resp.data.klines)
    } catch (err) {
      setError(axios.isAxiosError(err) ? err.message : 'Unexpected error')
    }
  }, [])

  useEffect(() => {
    setTfKlines(null)
    void fetchTfKlines(symbol, chartTf)
    if (chartTf === '15m') return
    const id = setInterval(() => void fetchTfKlines(symbol, chartTf), REFRESH_MS)
    return () => clearInterval(id)
  }, [symbol, chartTf, fetchTfKlines])

  const chartKlines = chartTf === '15m' ? data?.klines ?? [] : tfKlines ?? []

  const header = useMemo(
    () => (
      <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-950/90 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <Activity size={20} className="text-emerald-400" />
          <h1 className="text-base font-semibold tracking-tight text-zinc-100">
            Quant Engine Dashboard
          </h1>
          <span className="hidden rounded border border-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-widest text-zinc-500 sm:inline">
            Binance USDT-M
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex overflow-hidden rounded-lg border border-zinc-800">
            {SYMBOLS.map((s) => (
              <button
                key={s}
                onClick={() => setSymbol(s)}
                className={`px-3.5 py-1.5 text-xs font-semibold tracking-wider transition-colors ${
                  s === symbol
                    ? 'bg-zinc-100 text-zinc-950'
                    : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                }`}
              >
                {s.replace('USDT', '')}
              </button>
            ))}
          </div>
          <button
            onClick={() => void fetchSignal(symbol)}
            disabled={loading}
            title="Refresh"
            className="rounded-lg border border-zinc-800 bg-zinc-900 p-2 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-50"
          >
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </header>
    ),
    [symbol, loading, fetchSignal],
  )

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      {header}

      {error && (
        <div className="mx-5 mt-4 flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2.5 text-sm text-red-300">
          <AlertTriangle size={15} />
          Backend unreachable: {error} — is uvicorn running on :8000?
        </div>
      )}

      <main className="grid flex-1 grid-cols-3 gap-4 overflow-auto p-4">
        {/* Chart — 2/3 */}
        <section className="col-span-3 flex flex-col overflow-hidden rounded-xl border border-zinc-800 bg-[#0c0c0f] lg:col-span-2">
          <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
            <div className="flex items-center gap-2.5">
              <span className="font-mono text-sm font-bold text-zinc-100">
                {data?.symbol ?? symbol}
              </span>
              <span className="hidden text-[10px] uppercase tracking-widest text-zinc-500 sm:inline">
                closed candles
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex overflow-hidden rounded-md border border-zinc-800">
                {CHART_TFS.map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setChartTf(tf)}
                    className={`px-2 py-1 text-[10px] font-semibold uppercase tracking-wider transition-colors ${
                      tf === chartTf
                        ? 'bg-zinc-200 text-zinc-950'
                        : 'bg-zinc-900 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
              {(live.price ?? data?.price) != null && (
                <span className="flex items-center gap-2">
                  {live.price != null && (
                    <span className="flex items-center gap-1 text-[9px] font-bold uppercase tracking-widest text-emerald-500">
                      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
                      Live
                    </span>
                  )}
                  <span
                    className={`font-mono text-sm font-semibold tabular-nums transition-colors ${
                      live.tickDir === 'up'
                        ? 'text-emerald-400'
                        : live.tickDir === 'down'
                          ? 'text-red-400'
                          : 'text-zinc-200'
                    }`}
                  >
                    {fmt((live.price ?? data?.price) as number)}
                  </span>
                  {live.change24h != null && (
                    <span
                      className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums ${
                        live.change24h >= 0
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : 'bg-red-500/10 text-red-400'
                      }`}
                    >
                      {live.change24h >= 0 ? '+' : ''}
                      {live.change24h.toFixed(2)}% 24h
                    </span>
                  )}
                </span>
              )}
            </div>
          </div>
          <div className="flex-1">
            {data && chartKlines.length > 0 ? (
              <PriceChart data={data} klines={chartKlines} zones={visibleZones} />
            ) : (
              <div className="flex h-full min-h-[420px] items-center justify-center text-sm text-zinc-600">
                {loading ? 'Loading market data…' : 'No data'}
              </div>
            )}
          </div>
        </section>

        {/* Metrics — 1/3 */}
        <aside className="col-span-3 flex flex-col gap-4 lg:col-span-1">
          <div className="flex items-center justify-between rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
              Zones · ±2 ATR
            </span>
            <ZoneToggles
              visibility={zoneVisibility}
              counts={zoneCounts}
              onToggle={toggleZoneType}
            />
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
                Market regime (1D)
              </h2>
              {data && <RegimeBadge regime={data.regime} />}
            </div>
            <div className="space-y-3.5">
              <ScoreBar label="Long score" value={data?.long_score ?? 0} tone="long" />
              <ScoreBar label="Short score" value={data?.short_score ?? 0} tone="short" />
            </div>
          </div>

          <div className="flex-1 rounded-xl border border-zinc-800 bg-zinc-950 p-4">
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-zinc-400">
              Execution plan
            </h2>
            {data ? (
              <PlanPanel plan={data.active_plan} />
            ) : (
              <div className="py-8 text-center text-sm text-zinc-600">—</div>
            )}
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5 text-[11px] text-zinc-600">
            {updatedAt
              ? `Last updated ${updatedAt.toLocaleTimeString()} · auto-refresh 60s`
              : 'Waiting for first update…'}
            {data && ` · candle ${new Date(data.evaluated_at).toLocaleTimeString()}`}
          </div>
        </aside>
      </main>
    </div>
  )
}
