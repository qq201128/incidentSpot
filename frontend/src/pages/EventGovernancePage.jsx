import { useCallback, useEffect, useState } from "react";
import { fetchEventGovernance } from "../api/workbenchClient";
import {
  AllEvaluationsSection,
  SimulationObservationSection,
} from "./EventGovernanceTables";
import "./EventGovernancePage.css";
import "./EventGovernancePage.responsive.css";

const DURATIONS = [
  { value: "10m", label: "10分钟" },
  { value: "30m", label: "30分钟" },
  { value: "60m", label: "60分钟" },
  { value: "1d", label: "1天" },
];

const POLL_MS = 8000;

export default function EventGovernancePage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [duration, setDuration] = useState("10m");
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [latencyMs, setLatencyMs] = useState(null);

  const reload = useCallback(async () => {
    try {
      const { data, latencyMs: ms } = await fetchEventGovernance(symbol, duration);
      setPayload(data);
      setLatencyMs(ms);
      setStatus("");
    } catch (err) {
      console.error("事件观测加载失败", err);
      setStatus(`加载失败：${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [symbol, duration]);

  useEffect(() => {
    setLoading(true);
    let timer;
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      await reload();
      if (!stopped) timer = window.setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [reload]);

  const deviation = payload?.shadowEventDeviation;
  const batch = payload?.batchComboDemotion;
  const singleFactor = payload?.factorCandidateDemotion;
  const observation = payload?.simulationObservation;
  const batchWatchlist = batch?.watchlist ?? (batch?.evaluations ?? []).filter((row) => row.status === "demoted");
  const singleWatchlist =
    singleFactor?.watchlist ?? (singleFactor?.evaluations ?? []).filter((row) => row.status === "demoted");
  const totalWatchlistCount =
    observation?.watchlistCount ?? batchWatchlist.length + singleWatchlist.length;
  const summary = deviation?.summary ?? {};

  return (
    <main className="event-governance-page layout">
      <header className="event-gov-topbar">
        <div className="event-gov-topbar-main">
          <h1>事件样本观测</h1>
          <p>
            Shadow 与 Event 偏差监控 · 单因子 / 多因子模拟 Event PnL 观察（只读，不自动关闭策略）
          </p>
        </div>
        <div className="event-gov-controls">
          <label>
            交易对
            <select value={symbol} onChange={(event) => setSymbol(event.target.value)}>
              <option value="BTCUSDT">BTCUSDT</option>
              <option value="ETHUSDT">ETHUSDT</option>
            </select>
          </label>
          <label>
            周期
            <select value={duration} onChange={(event) => setDuration(event.target.value)}>
              {DURATIONS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => void reload()}>
            刷新
          </button>
          {latencyMs != null ? <span className="event-gov-latency">{Math.round(latencyMs)} ms</span> : null}
        </div>
      </header>

      {status ? <div className="event-gov-banner">{status}</div> : null}
      {loading && !payload ? <div className="event-gov-loading">正在加载观测数据…</div> : null}

      {payload ? (
        <div className="event-gov-body">
          <section className="event-gov-kpis">
            <KpiCard label="配对样本" value={summary.pairedCount ?? 0} />
            <KpiCard
              label="Shadow对/Event亏"
              value={summary.shadowWinEventLossCount ?? 0}
              hint={formatRate(summary.shadowWinEventLossRate)}
              warn={(summary.shadowWinEventLossCount ?? 0) > 0}
            />
            <KpiCard label="Event 总盈亏 (U)" value={formatNumber(summary.totalEventPnlU)} />
            <KpiCard label="关注策略数" value={totalWatchlistCount} warn={totalWatchlistCount > 0} />
            <KpiCard label="多因子关注" value={observation?.batchComboWatchlistCount ?? batchWatchlist.length} />
            <KpiCard label="单因子关注" value={observation?.factorCandidateWatchlistCount ?? singleWatchlist.length} />
          </section>

          <SimulationObservationSection
            title="需关注 · 多因子组合"
            emptyText="暂无多因子组合落入 Event PnL 观察名单。"
            demotion={batch}
            watchlist={batchWatchlist}
            kind="combo"
          />

          <SimulationObservationSection
            title="需关注 · 单因子模拟"
            emptyText="暂无单因子模拟落入 Event PnL 观察名单。"
            demotion={singleFactor}
            watchlist={singleWatchlist}
            kind="single"
          />

          <AllEvaluationsSection title="全部多因子组合评估" demotion={batch} watchlistCount={batchWatchlist.length} />

          <AllEvaluationsSection title="全部单因子模拟评估" demotion={singleFactor} watchlistCount={singleWatchlist.length} />
        </div>
      ) : null}
    </main>
  );
}

function KpiCard({ label, value, hint, warn = false }) {
  return (
    <article className={`event-gov-kpi${warn ? " event-gov-kpi-warn" : ""}`}>
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
      {hint ? <small>{hint}</small> : null}
    </article>
  );
}

function formatRate(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(2);
}
