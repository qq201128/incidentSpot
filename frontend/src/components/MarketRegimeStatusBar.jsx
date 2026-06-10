import { useEffect, useMemo, useState } from "react";
import { fetchLatestPrediction } from "../api/client";
import { formatFinalScore, regimePartLabel } from "../utils/eventFinalDecisionLabels";
import "./MarketRegimeStatusBar.css";

const POLL_MS = 10000;

export default function MarketRegimeStatusBar({ duration, latestPrediction, symbol }) {
  const [snapshot, setSnapshot] = useState(null);
  const [error, setError] = useState("");
  const liveSnapshot = useMemo(
    () => predictionRegimeSnapshot(latestPrediction, symbol, duration),
    [duration, latestPrediction, symbol],
  );
  const view = liveSnapshot || snapshot;

  useEffect(() => {
    let stopped = false;
    let timer;
    async function load() {
      try {
        const prediction = await fetchLatestPrediction(symbol, duration);
        if (!stopped) {
          setSnapshot(predictionRegimeSnapshot(prediction, symbol, duration));
          setError("");
        }
      } catch (err) {
        if (!stopped) setError(String(err?.response?.data?.detail || err?.message || "环境检测读取失败"));
      } finally {
        if (!stopped) timer = window.setTimeout(load, POLL_MS);
      }
    }
    void load();
    return () => {
      stopped = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [duration, symbol]);

  return (
    <section className={`market-regime-bar ${toneClass(view?.trendState)}`} aria-label="市场环境">
      <div>
        <span className="market-regime-eyebrow">市场环境</span>
        <strong>{regimeHeadline(view, error)}</strong>
      </div>
      <div className="market-regime-metrics">
        <Metric label="趋势" value={view ? regimePartLabel(view.trendState) : "—"} />
        <Metric label="波动" value={view ? regimePartLabel(view.volatilityState) : "—"} />
        <Metric label="置信度" value={view ? formatFinalScore(view.confidence) : "—"} />
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <span>
      {label}
      <b>{value}</b>
    </span>
  );
}

function predictionRegimeSnapshot(prediction, symbol, duration) {
  if (prediction?.symbol !== symbol || prediction?.duration !== duration) return null;
  const regime = prediction?.marketRegime;
  if (!regime) return null;
  if (regime.ready === false) return { reason: regime.reason || "数据不足" };
  return regime;
}

function regimeHeadline(view, error) {
  if (!view) return error || "等待环境检测";
  if (view.reason) return `数据不足：${view.reason}`;
  return `${regimePartLabel(view.trendState)} · ${regimePartLabel(view.volatilityState)}`;
}

function toneClass(trend) {
  if (trend === "trend_up") return "is-up";
  if (trend === "trend_down") return "is-down";
  if (trend === "range") return "is-range";
  return "is-uncertain";
}
