import { useCallback, useEffect, useState } from "react";
import { fetchEventGovernance } from "../api/workbenchClient";
import { strategyLabel } from "../utils/strategyLabels";
import "./EventGovernancePage.css";

const DURATIONS = [
  { value: "10m", label: "10分钟" },
  { value: "30m", label: "30分钟" },
  { value: "60m", label: "60分钟" },
  { value: "1d", label: "1天" },
];

const POLL_MS = 8000;

const ISSUE_LABELS = {
  systemic_shadow_win_event_loss: "系统性：Shadow 正确但 Event 亏损",
  strategy_shadow_win_event_loss: "策略级：Shadow 与 Event 偏差偏高",
};

const REASON_LABELS = {
  consecutive_losses: "连续亏损",
  insufficient_event_samples: "Event 样本不足",
  insufficient_settled_samples: "已结算样本不足",
  live_win_rate_below_target: "Event 胜率低于目标",
  profit_factor_below_one: "Event 盈亏比低于 1",
  stable_live_target_met: "Event 指标达标",
  unsupported_strategy_key: "非 batch 组合策略",
};

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
  const watchlist = batch?.watchlist ?? (batch?.evaluations ?? []).filter((row) => row.status === "demoted");
  const summary = deviation?.summary ?? {};
  const issues = deviation?.issues ?? [];

  return (
    <main className="event-governance-page layout">
      <header className="event-gov-topbar">
        <div className="event-gov-topbar-main">
          <h1>事件样本观测</h1>
          <p>Shadow 与 Event 偏差监控 · Batch 组合 Event PnL 观察（只读，不自动关闭策略）</p>
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
            <KpiCard label="关注组合数" value={watchlist.length} warn={watchlist.length > 0} />
          </section>

          <section className="event-gov-panel">
            <div className="event-gov-panel-head">
              <h2>偏差告警</h2>
              <span>{issues.length ? `${issues.length} 条` : "暂无告警"}</span>
            </div>
            {issues.length === 0 ? (
              <p className="event-gov-empty">当前未发现系统性 Shadow/Event 偏差问题。</p>
            ) : (
              <ul className="event-gov-issue-list">
                {issues.map((issue, index) => (
                  <li
                    key={`${issue.code}-${issue.strategyKey ?? index}`}
                    className={`event-gov-issue severity-${issue.severity}`}
                  >
                    <strong>{ISSUE_LABELS[issue.code] ?? issue.code}</strong>
                    <span>{issue.message}</span>
                    {issue.strategyKey ? <code>{strategyLabel(issue.strategyKey)}</code> : null}
                    {issue.shadowWinEventLossRate != null ? (
                      <span className="event-gov-meta">偏差率 {formatRate(issue.shadowWinEventLossRate)}</span>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="event-gov-panel">
            <div className="event-gov-panel-head">
              <h2>需关注组合（Event PnL 未达标）</h2>
              <span>
                观察模式 · 已评估 {batch?.evaluatedCount ?? 0} 个 · 不自动 disable
              </span>
            </div>
            {watchlist.length === 0 ? (
              <p className="event-gov-empty">暂无组合落入 Event PnL 观察名单。</p>
            ) : (
              <div className="event-gov-table-wrap">
                <table className="event-gov-table">
                  <thead>
                    <tr>
                      <th>策略</th>
                      <th>原因</th>
                      <th>样本</th>
                      <th>胜率</th>
                      <th>盈亏比</th>
                      <th>连亏</th>
                      <th>Event PnL (U)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {watchlist.map((row) => (
                      <tr key={row.strategyKey}>
                        <td title={row.strategyKey}>{strategyLabel(row.strategyKey)}</td>
                        <td>{REASON_LABELS[row.reason] ?? row.reason}</td>
                        <td>{row.sampleCount ?? "—"}</td>
                        <td>{formatRate(row.winRate)}</td>
                        <td>{formatNumber(row.profitFactor)}</td>
                        <td>{row.consecutiveLosses ?? "—"}</td>
                        <td className={Number(row.totalPnlU) < 0 ? "neg" : "pos"}>
                          {formatNumber(row.totalPnlU)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {(batch?.evaluations?.length ?? 0) > watchlist.length ? (
            <section className="event-gov-panel event-gov-panel-muted">
              <div className="event-gov-panel-head">
                <h2>全部 Batch 组合评估</h2>
                <span>{batch.evaluations.length} 个</span>
              </div>
              <div className="event-gov-table-wrap">
                <table className="event-gov-table">
                  <thead>
                    <tr>
                      <th>策略</th>
                      <th>状态</th>
                      <th>样本</th>
                      <th>胜率</th>
                      <th>Event PnL (U)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batch.evaluations.map((row) => (
                      <tr key={row.strategyKey}>
                        <td title={row.strategyKey}>{strategyLabel(row.strategyKey)}</td>
                        <td>
                          <StatusBadge status={row.status} />
                        </td>
                        <td>{row.sampleCount ?? "—"}</td>
                        <td>{formatRate(row.winRate)}</td>
                        <td className={Number(row.totalPnlU) < 0 ? "neg" : "pos"}>
                          {formatNumber(row.totalPnlU)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
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

function StatusBadge({ status }) {
  const labels = {
    active: "正常",
    demoted: "需关注",
    collecting: "收集中",
    insufficient_samples: "样本不足",
  };
  return <span className={`event-gov-badge status-${status}`}>{labels[status] ?? status}</span>;
}

function formatRate(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(2);
}
