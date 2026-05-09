import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  TickMarkType,
  createChart,
  isBusinessDay,
  isUTCTimestamp,
} from "lightweight-charts";

const DEFAULT_CHART_HEIGHT = 440;
const CHART_COLORS = Object.freeze({
  background: "#0c111b",
  grid: "#1f2937",
  scale: "#253041",
  text: "#d1d7e0",
  up: "#26a69a",
  down: "#ef5350",
});

export default function KlineChart({ data, latest }) {
  const wrapRef = useRef(null);
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const lastTimeRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !wrapRef.current) {
      return;
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

    chart.subscribeCrosshairMove(onCrosshairMove);

    const observer = new ResizeObserver(() => {
      if (wrapRef.current) {
        chart.applyOptions(chartSize(wrapRef.current));
      }
    });
    observer.observe(wrapRef.current);

    return () => {
      chart.unsubscribeCrosshairMove(onCrosshairMove);
      observer.disconnect();
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !data.length) {
      return;
    }
    const normalized = normalizeSeriesData(data);
    if (!normalized.length) return;
    seriesRef.current.setData(normalized);
    lastTimeRef.current = normalized[normalized.length - 1].time;
  }, [data]);

  useEffect(() => {
    if (!seriesRef.current || !latest) {
      return;
    }
    const next = normalizeCandle(latest);
    if (!next) return;
    const lastTime = lastTimeRef.current;
    if (lastTime != null && timeValue(next.time) < timeValue(lastTime)) {
      return;
    }
    seriesRef.current.update(next);
    lastTimeRef.current = next.time;
  }, [latest]);

  return (
    <div className="chart-container kline-chart-wrap" ref={wrapRef}>
      <div
        className="kline-chart-canvas-host"
        ref={containerRef}
        aria-hidden
      />
      <div className="kline-hover-tooltip" ref={tooltipRef} role="status" />
    </div>
  );
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

/** 十字光标时间轴标签与悬浮层：本地日历日期 + 时分 */
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
