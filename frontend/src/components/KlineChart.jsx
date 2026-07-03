import { memo, useEffect, useRef } from "react";
import {
  CandlestickSeries,
  CrosshairMode,
  LineSeries,
  LineStyle,
  PriceScaleMode,
  TickMarkType,
  createChart,
  isBusinessDay,
  isUTCTimestamp,
} from "lightweight-charts";
import { mergeChartSeries } from "../utils/klineFormingCandle";
import { computeMovingAverage } from "../utils/klineIndicators";

const DEFAULT_CHART_HEIGHT = 440;
const CHART_COLORS = Object.freeze({
  background: "#0c111b",
  grid: "#1f2937",
  scale: "#253041",
  text: "#d1d7e0",
  up: "#26a69a",
  down: "#ef5350",
});

const MA_STYLES = Object.freeze({
  ma7: { period: 7, color: "#f3d49b", label: "MA7" },
  ma20: { period: 20, color: "#5ba4ff", label: "MA20" },
  ma60: { period: 60, color: "#c084fc", label: "MA60" },
});

function KlineChart({
  clearDrawingsToken = 0,
  data,
  drawingTool = "cursor",
  drawingsLocked = false,
  fitToken = 0,
  indicators = {},
  latest,
  settings = {},
}) {
  const wrapRef = useRef(null);
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const maSeriesRef = useRef({});
  const priceLinesRef = useRef([]);
  const trendSeriesRef = useRef([]);
  const trendDraftRef = useRef(null);
  const lastTimeRef = useRef(null);
  const drawingToolRef = useRef(drawingTool);
  const drawingsLockedRef = useRef(drawingsLocked);

  drawingToolRef.current = drawingTool;
  drawingsLockedRef.current = drawingsLocked;

  useEffect(() => {
    if (!containerRef.current || !wrapRef.current) {
      return undefined;
    }

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: CHART_COLORS.background },
        textColor: CHART_COLORS.text,
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      ...chartSize(wrapRef.current),
      localization: {
        locale: "zh-CN",
        timeFormatter: (time) => formatCrosshairTime(time),
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      timeScale: {
        borderColor: CHART_COLORS.scale,
        tickMarkFormatter: (time, tickMarkType, locale) => {
          const d = timeToDate(time);
          if (tickMarkType === TickMarkType.Time || tickMarkType === TickMarkType.TimeWithSeconds) {
            return d.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit", hour12: false });
          }
          if (tickMarkType === TickMarkType.DayOfMonth) {
            const day = d.getDate();
            const hh = String(d.getHours()).padStart(2, "0");
            const mm = String(d.getMinutes()).padStart(2, "0");
            const label = `${day}日${hh}:${mm}`;
            return label.length > 8 ? `${hh}:${mm}` : label;
          }
          return null;
        },
      },
      rightPriceScale: { borderColor: CHART_COLORS.scale },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: CHART_COLORS.up,
      downColor: CHART_COLORS.down,
      borderVisible: false,
      wickUpColor: CHART_COLORS.up,
      wickDownColor: CHART_COLORS.down,
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const onCrosshairMove = (param) => {
      const tip = tooltipRef.current;
      const box = wrapRef.current;
      if (!tip || !box) return;

      const bar = param.seriesData.get(series);
      const hasOhlc = bar && typeof bar.open === "number" && typeof bar.close === "number";

      if (param.point === undefined || param.time === undefined || !hasOhlc) {
        tip.style.display = "none";
        tip.textContent = "";
        return;
      }

      const timeStr = formatCrosshairTime(param.time);
      const openStr = fmtPx(bar.open);
      const closeStr = fmtPx(bar.close);

      tip.style.display = "block";
      tip.style.whiteSpace = "pre-line";
      tip.textContent = `时间：${timeStr}\n开盘：${openStr}\n收盘：${closeStr}`;

      const pad = 14;
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      let left = param.point.x + pad;
      let top = param.point.y + pad;
      if (left + tw > box.clientWidth) left = Math.max(0, param.point.x - tw - pad);
      if (top + th > box.clientHeight) top = Math.max(0, param.point.y - th - pad);
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    };

    const onClick = (param) => {
      if (drawingsLockedRef.current) return;
      const tool = drawingToolRef.current;
      if (!tool || tool === "cursor" || tool === "measure" || tool === "reset" || tool === "settings") {
        return;
      }
      if (!param.point || param.time === undefined) return;
      const price = series.coordinateToPrice(param.point.y);
      if (price == null || !Number.isFinite(price)) return;

      if (tool === "hline") {
        const line = series.createPriceLine({
          price,
          color: "#f3d49b",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
        });
        priceLinesRef.current.push(line);
        return;
      }

      if (tool === "trend") {
        const draft = trendDraftRef.current;
        if (!draft) {
          trendDraftRef.current = { time: param.time, price };
          return;
        }
        const trendSeries = chart.addSeries(LineSeries, {
          color: "#ff9f68",
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        trendSeries.setData([
          { time: draft.time, value: draft.price },
          { time: param.time, value: price },
        ]);
        trendSeriesRef.current.push(trendSeries);
        trendDraftRef.current = null;
      }
    };

    chart.subscribeCrosshairMove(onCrosshairMove);
    chart.subscribeClick(onClick);

    const observer = new ResizeObserver(() => {
      if (wrapRef.current) {
        chart.applyOptions(chartSize(wrapRef.current));
      }
    });
    observer.observe(wrapRef.current);

    return () => {
      chart.unsubscribeCrosshairMove(onCrosshairMove);
      chart.unsubscribeClick(onClick);
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      maSeriesRef.current = {};
      priceLinesRef.current = [];
      trendSeriesRef.current = [];
      trendDraftRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;

    chart.applyOptions({
      grid: {
        vertLines: { visible: settings.showGrid !== false, color: CHART_COLORS.grid },
        horzLines: { visible: settings.showGrid !== false, color: CHART_COLORS.grid },
      },
      crosshair: {
        mode: settings.crosshairMagnet ? CrosshairMode.Magnet : CrosshairMode.Normal,
      },
    });

    const scaleMode = priceScaleModeFromSetting(settings.priceScaleMode);
    chart.priceScale("right").applyOptions({ mode: scaleMode });
  }, [settings.crosshairMagnet, settings.priceScaleMode, settings.showGrid]);

  useEffect(() => {
    if (!seriesRef.current || !data.length) return;
    const normalized = normalizeSeriesData(data);
    if (!normalized.length) return;
    seriesRef.current.setData(normalized);
    lastTimeRef.current = normalized[normalized.length - 1].time;
    syncMovingAverages(chartRef.current, maSeriesRef, normalized, indicators);
  }, [data, indicators]);

  useEffect(() => {
    if (!seriesRef.current || !latest) return;
    const next = normalizeCandle(latest);
    if (!next) return;
    const lastTime = lastTimeRef.current;
    if (lastTime != null && timeValue(next.time) < timeValue(lastTime)) return;
    seriesRef.current.update(next);
    lastTimeRef.current = next.time;
    const merged = mergeChartSeries(normalizeSeriesData(data), next);
    syncMovingAverages(chartRef.current, maSeriesRef, merged, indicators);
    if (settings.autoScroll && chartRef.current) {
      chartRef.current.timeScale().scrollToRealTime();
    }
  }, [latest, settings.autoScroll, data, indicators]);

  useEffect(() => {
    if (!chartRef.current || !fitToken) return;
    chartRef.current.timeScale().fitContent();
  }, [fitToken]);

  useEffect(() => {
    if (!clearDrawingsToken) return;
    clearChartDrawings(chartRef.current, seriesRef.current, priceLinesRef, trendSeriesRef, trendDraftRef);
  }, [clearDrawingsToken]);

  useEffect(() => {
    trendDraftRef.current = null;
  }, [drawingTool]);

  return (
    <div className="chart-container kline-chart-wrap" ref={wrapRef}>
      <div className="kline-chart-canvas-host" ref={containerRef} aria-hidden />
      <div className="kline-hover-tooltip" ref={tooltipRef} role="status" />
    </div>
  );
}

function priceScaleModeFromSetting(mode) {
  if (mode === "log") return PriceScaleMode.Logarithmic;
  if (mode === "percent") return PriceScaleMode.Percentage;
  return PriceScaleMode.Normal;
}

function syncMovingAverages(chart, maSeriesRef, candles, indicators) {
  if (!chart) return;
  for (const [key, style] of Object.entries(MA_STYLES)) {
    const enabled = Boolean(indicators[key]);
    const existing = maSeriesRef.current[key];
    if (!enabled) {
      if (existing) {
        chart.removeSeries(existing);
        delete maSeriesRef.current[key];
      }
      continue;
    }
    const points = computeMovingAverage(candles, style.period);
    let series = existing;
    if (!series) {
      series = chart.addSeries(LineSeries, {
        color: style.color,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        title: style.label,
      });
      maSeriesRef.current[key] = series;
    }
    series.setData(points);
  }
}

function clearChartDrawings(chart, candleSeries, priceLinesRef, trendSeriesRef, trendDraftRef) {
  if (candleSeries) {
    for (const line of priceLinesRef.current) {
      candleSeries.removePriceLine(line);
    }
  }
  priceLinesRef.current = [];
  if (chart) {
    for (const series of trendSeriesRef.current) {
      chart.removeSeries(series);
    }
  }
  trendSeriesRef.current = [];
  trendDraftRef.current = null;
}

function chartSize(container) {
  return {
    width: container.clientWidth,
    height: container.clientHeight || DEFAULT_CHART_HEIGHT,
  };
}

function timeToDate(time) {
  if (isUTCTimestamp(time)) {
    return new Date(time * 1000);
  }
  if (isBusinessDay(time)) {
    return new Date(Date.UTC(time.year, time.month - 1, time.day));
  }
  if (typeof time === "string") {
    const parsed = new Date(time);
    return Number.isNaN(parsed.getTime()) ? new Date() : parsed;
  }
  return new Date();
}

function formatCrosshairTime(time) {
  const d = timeToDate(time);
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const da = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${y}-${mo}-${da} ${hh}:${mi}`;
}

function fmtPx(v) {
  return Number.isFinite(v) ? Number(v).toFixed(2) : "—";
}

function normalizeSeriesData(rows) {
  const byTime = new Map();
  for (const row of rows) {
    const candle = normalizeCandle(row);
    if (candle) byTime.set(timeValue(candle.time), candle);
  }
  return [...byTime.values()].sort((a, b) => timeValue(a.time) - timeValue(b.time));
}

function normalizeCandle(row) {
  const time = normalizeTime(row?.time);
  const open = Number(row?.open);
  const high = Number(row?.high);
  const low = Number(row?.low);
  const close = Number(row?.close);
  if (time == null || [open, high, low, close].some((v) => !Number.isFinite(v))) return null;
  return { time, open, high, low, close };
}

function normalizeTime(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : value;
  }
  return null;
}

function timeValue(value) {
  if (typeof value === "number") return value;
  if (typeof value === "string") return Date.parse(value) / 1000;
  return Number.NaN;
}

export default memo(KlineChart, (prev, next) => {
  // Deep comparison for data and latest to prevent unnecessary chart redraws
  const dataEqual = prev.data === next.data ||
    (prev.data?.length === next.data?.length &&
     prev.data?.[prev.data.length - 1]?.time === next.data?.[next.data.length - 1]?.time);

  const latestEqual = prev.latest === next.latest ||
    (prev.latest?.time === next.latest?.time &&
     prev.latest?.close === next.latest?.close);

  return dataEqual &&
         latestEqual &&
         prev.drawingTool === next.drawingTool &&
         prev.drawingsLocked === next.drawingsLocked &&
         prev.clearDrawingsToken === next.clearDrawingsToken &&
         prev.fitToken === next.fitToken &&
         JSON.stringify(prev.indicators) === JSON.stringify(next.indicators) &&
         JSON.stringify(prev.settings) === JSON.stringify(next.settings);
});
