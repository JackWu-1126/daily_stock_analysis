import type React from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { stocksApi, type KLinePoint } from '../../api/stocks';
import { getParsedApiError, type ParsedApiError } from '../../api/error';
import { ApiErrorAlert, Card } from '../common';
import { DashboardPanelHeader, DashboardStateBlock } from '../dashboard';
import type { ReportLanguage } from '../../types/analysis';
import { getReportText, normalizeReportLanguage } from '../../utils/reportLanguage';

interface StockPriceChartProps {
  stockCode: string;
  stockName?: string;
  language?: ReportLanguage;
}

const MA_COLORS = {
  ma5: '#f59e0b',
  ma10: '#3b82f6',
  ma20: '#a855f7',
} as const;

const GRID_STROKE = 'rgba(148, 163, 184, 0.18)';
const AXIS_STROKE = 'rgba(148, 163, 184, 0.5)';
const AXIS_WIDTH = 56;

type RangeOption = { days: number; labelKey: 'chartRange1m' | 'chartRange3m' | 'chartRange6m' | 'chartRange1y' };

const RANGE_OPTIONS: RangeOption[] = [
  { days: 30, labelKey: 'chartRange1m' },
  { days: 90, labelKey: 'chartRange3m' },
  { days: 180, labelKey: 'chartRange6m' },
  { days: 365, labelKey: 'chartRange1y' },
];

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const formatPrice = (value?: number): string => (isFiniteNumber(value) ? value.toFixed(2) : '--');

const formatChangePercent = (value?: number): string => {
  if (!isFiniteNumber(value)) return '--';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

const formatCompactNumber = (value?: number): string => {
  if (!isFiniteNumber(value)) return '--';
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
};

const getPriceColor = (point: Pick<KLinePoint, 'open' | 'close'>): string => {
  if (!isFiniteNumber(point.open) || !isFiniteNumber(point.close)) return AXIS_STROKE;
  return point.close >= point.open ? 'var(--home-price-up)' : 'var(--home-price-down)';
};

type CandleShapeProps = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: KLinePoint;
};

const CandleShape: React.FC<CandleShapeProps> = ({ x = 0, y = 0, width = 0, height = 0, payload }) => {
  if (!payload) return null;
  const { open, close, high, low } = payload;
  if (![open, close, high, low].every(isFiniteNumber)) return null;

  const color = getPriceColor(payload);
  const range = high - low;
  const toY = (value: number) => (range <= 0 ? y + height / 2 : y + height * ((high - value) / range));
  const bodyTop = toY(Math.max(open, close));
  const bodyBottom = toY(Math.min(open, close));
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1);
  const bodyWidth = Math.max(width * 0.6, 1);
  const bodyX = x + (width - bodyWidth) / 2;
  const centerX = x + width / 2;

  return (
    <g>
      <line x1={centerX} x2={centerX} y1={y} y2={y + height} stroke={color} strokeWidth={1} />
      <rect x={bodyX} y={bodyTop} width={bodyWidth} height={bodyHeight} fill={color} />
    </g>
  );
};

type ChartTooltipContentProps = {
  active?: boolean;
  payload?: Array<{ payload?: KLinePoint }>;
  text: ReturnType<typeof getReportText>;
};

const ChartTooltipContent: React.FC<ChartTooltipContentProps> = ({ active, payload, text }) => {
  if (!active || !payload?.[0]?.payload) return null;
  const point = payload[0].payload;
  const priceColor = getPriceColor(point);

  return (
    <div className="rounded-lg border border-border/70 bg-card/95 px-3 py-2 text-xs shadow-lg backdrop-blur">
      <p className="mb-1.5 font-medium text-foreground">{point.date}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-secondary-text">
        <span>{text.chartOpen}: {formatPrice(point.open)}</span>
        <span>{text.chartHigh}: {formatPrice(point.high)}</span>
        <span>{text.chartLow}: {formatPrice(point.low)}</span>
        <span style={{ color: priceColor }}>{text.chartClose}: {formatPrice(point.close)}</span>
        {isFiniteNumber(point.changePercent) && (
          <span style={{ color: priceColor }}>{text.chartChangePercent}: {formatChangePercent(point.changePercent)}</span>
        )}
        {isFiniteNumber(point.volume) && <span>{text.chartVolume}: {formatCompactNumber(point.volume)}</span>}
      </div>
      {(isFiniteNumber(point.ma5) || isFiniteNumber(point.ma10) || isFiniteNumber(point.ma20)) && (
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px]">
          {isFiniteNumber(point.ma5) && <span style={{ color: MA_COLORS.ma5 }}>{text.chartMa5} {formatPrice(point.ma5)}</span>}
          {isFiniteNumber(point.ma10) && <span style={{ color: MA_COLORS.ma10 }}>{text.chartMa10} {formatPrice(point.ma10)}</span>}
          {isFiniteNumber(point.ma20) && <span style={{ color: MA_COLORS.ma20 }}>{text.chartMa20} {formatPrice(point.ma20)}</span>}
        </div>
      )}
    </div>
  );
};

const RangeControls: React.FC<{
  rangeDays: number;
  onChange: (days: number) => void;
  text: ReturnType<typeof getReportText>;
}> = ({ rangeDays, onChange, text }) => (
  <div className="flex flex-wrap items-center gap-1.5">
    {RANGE_OPTIONS.map((option) => (
      <button
        key={option.days}
        type="button"
        onClick={() => onChange(option.days)}
        className={`rounded-lg border px-2.5 py-1 text-xs font-medium transition-colors ${
          rangeDays === option.days
            ? 'border-primary/50 bg-primary/10 text-primary'
            : 'border-border/70 bg-background/50 text-secondary-text hover:bg-hover hover:text-foreground'
        }`}
      >
        {text[option.labelKey]}
      </button>
    ))}
  </div>
);

/**
 * 个股走势图：K 线蜡烛图 + 成交量 + MA5/MA10/MA20 均线
 */
export const StockPriceChart: React.FC<StockPriceChartProps> = ({ stockCode, stockName, language = 'zh' }) => {
  const reportLanguage = normalizeReportLanguage(language);
  const text = getReportText(reportLanguage);

  const [rangeDays, setRangeDays] = useState(90);
  const [data, setData] = useState<KLinePoint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ParsedApiError | null>(null);

  const fetchData = useCallback(async () => {
    if (!stockCode) return;
    setIsLoading(true);
    setError(null);

    try {
      const response = await stocksApi.getHistory(stockCode, rangeDays);
      setData(response.data || []);
    } catch (err) {
      setError(getParsedApiError(err));
    } finally {
      setIsLoading(false);
    }
  }, [stockCode, rangeDays]);

  useEffect(() => {
    setData([]);
    setError(null);
    if (stockCode) {
      fetchData();
    }
  }, [stockCode, rangeDays, fetchData]);

  const candleDataKey = useMemo(() => (point: KLinePoint) => [point.low, point.high], []);

  if (!stockCode) {
    return null;
  }

  return (
    <Card variant="bordered" padding="md" className="home-panel-card">
      <DashboardPanelHeader
        eyebrow={text.chartTitle}
        title={stockName ? `${stockName} ${stockCode}` : stockCode}
        actions={(
          <div className="flex items-center gap-2">
            {isLoading ? (
              <div className="home-spinner h-3.5 w-3.5 animate-spin border-2" aria-hidden="true" />
            ) : null}
            <RangeControls rangeDays={rangeDays} onChange={setRangeDays} text={text} />
          </div>
        )}
      />

      {error && !isLoading && (
        <ApiErrorAlert
          error={error}
          actionLabel={text.retry}
          onAction={() => void fetchData()}
          dismissLabel={text.dismiss}
        />
      )}

      {isLoading && !error && (
        <DashboardStateBlock compact loading title={text.chartLoading} />
      )}

      {!isLoading && !error && data.length === 0 && (
        <DashboardStateBlock
          compact
          title={text.chartEmpty}
          description={text.chartEmptyDescription}
        />
      )}

      {!isLoading && !error && data.length > 0 && (
        <div className="space-y-1">
          <div className="h-[260px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 8 }}>
                <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value: string) => value.slice(5)}
                  tick={{ fontSize: 11, fill: 'currentColor' }}
                  stroke={AXIS_STROKE}
                  hide
                />
                <YAxis
                  domain={['dataMin', 'dataMax']}
                  tick={{ fontSize: 11, fill: 'currentColor' }}
                  stroke={AXIS_STROKE}
                  width={AXIS_WIDTH}
                  tickFormatter={(value: number) => value.toFixed(1)}
                />
                <ChartTooltip
                  cursor={{ stroke: AXIS_STROKE, strokeDasharray: '3 3' }}
                  content={(props: unknown) => <ChartTooltipContent {...(props as ChartTooltipContentProps)} text={text} />}
                />
                <Bar dataKey={candleDataKey} isAnimationActive={false} shape={(props: unknown) => <CandleShape {...(props as CandleShapeProps)} />} />
                <Line type="monotone" dataKey="ma5" stroke={MA_COLORS.ma5} dot={false} strokeWidth={1.25} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="ma10" stroke={MA_COLORS.ma10} dot={false} strokeWidth={1.25} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="ma20" stroke={MA_COLORS.ma20} dot={false} strokeWidth={1.25} isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="h-[90px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 0, right: 12, bottom: 4, left: 8 }}>
                <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value: string) => value.slice(5)}
                  tick={{ fontSize: 10, fill: 'currentColor' }}
                  stroke={AXIS_STROKE}
                  minTickGap={24}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'currentColor' }}
                  stroke={AXIS_STROKE}
                  width={AXIS_WIDTH}
                  tickFormatter={(value: number) => formatCompactNumber(value)}
                />
                <Bar dataKey="volume" isAnimationActive={false}>
                  {data.map((point) => (
                    <Cell key={point.date} fill={getPriceColor(point)} fillOpacity={0.6} />
                  ))}
                </Bar>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-3 pt-1 text-[11px] text-secondary-text">
            <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ backgroundColor: MA_COLORS.ma5 }} />{text.chartMa5}</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ backgroundColor: MA_COLORS.ma10 }} />{text.chartMa10}</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ backgroundColor: MA_COLORS.ma20 }} />{text.chartMa20}</span>
          </div>
        </div>
      )}
    </Card>
  );
};
