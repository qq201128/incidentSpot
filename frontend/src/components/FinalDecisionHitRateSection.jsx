import { useCallback, useEffect, useState } from "react";
import { fetchEventFinalDecisionSummary } from "../api/eventFinalDecisionClient";
import { strategyDurationLabel } from "../utils/strategyLabels";
import { finalDecisionLabel, regimePartLabel } from "../utils/eventFinalDecisionLabels";
import "./FinalDecisionHitRateSection.css";

const POLL_MS = 5000;

export default function FinalDecisionHitRateSection({ symbol, durationMinutes }) {
  const duration = durationFromMinutes(durationMinutes);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(
    async (signal) => {
      try {
        const data = await fetchEventFinalDecisionSummary(symbol, duration, { signal });
        if (!signal?.aborted) {
          setSummary(data);
          setError("");
        }
      } catch (err) {
        if (signal?.aborted || err?.code === "ERR_CANCELED") return;
        setError(err?.response?.data?.detail || err?.message || "读取最终裁判统计失败");
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [duration, symbol],
  );

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    void load(ac.signal);
    const timer = window.setInterval(() => void load(ac.signal), POLL_MS);
    return () => {
      ac.abort();
      window.clearInterval(timer);
    };
  }, [load]);

  const overall = summary?.overall || {};
  const winRate = overall.winRate;

  return (
    <section className="rhr-final-section card-surface" aria-label="事件最终裁判验证">
      <header className="rhr-final-head">
        <div>
          <span className="rhr-eyebrow">Event final decision</span>
          <h2>最终裁判验证</h2>
          <p>{symbol} · {strategyDurationLabel(duration)} · 展示 UP / DOWN / SKIP 分布与环境命中率</p>
        </div>
        <strong className={`rhr-final-rate tone-${hitTone(winRate)}`}>
          {loading ? "…" : formatWinRate(winRate)}
        </strong>
      </header>
      {error ? <p className="rhr-final-error" role="alert">{error}</p> : null}
      <div className="rhr-final-kpis">
        <FinalKpi label="总样本" value={overall.count ?? "—"} hint={`已结算 ${overall.settled ?? 0}`} />
        <FinalKpi label="命中" value={overall.wins ?? "—"} hint="仅统计已结算 UP/DOWN" />
        <FinalKpi
          label="SKIP 占比"
          value={skipShare(summary?.byDecision)}
          hint="SKIP 为正常审计结果，非错误"
        />
      </div>
      <div className="rhr-final-groups">
        <GroupTable title="按决策分布" rows={summary?.byDecision} labelFn={finalDecisionLabel} />
        <GroupTable title="按环境分布" rows={summary?.byRegime} labelFn={regimeGroupLabel} />
      </div>
    </section>
  );
}

function FinalKpi({ label, value, hint }) {
  return (
    <article className="rhr-final-kpi">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </article>
  );
}

function GroupTable({ title, rows, labelFn }) {
  const items = Array.isArray(rows) ? rows : [];
  return (
    <div className="rhr-final-group">
      <h3>{title}</h3>
      {items.length ? (
        <table>
          <thead>
            <tr>
              <th>分组</th>
              <th>样本</th>
              <th>命中率</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.key}>
                <td>{labelFn(row.key)}</td>
                <td>{row.count ?? 0}</td>
                <td>{formatWinRate(row.winRate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="rhr-final-empty">暂无数据</p>
      )}
    </div>
  );
}

function regimeGroupLabel(key) {
  const [trend, vol] = String(key || "").split(":");
  return `${regimePartLabel(trend)} · ${regimePartLabel(vol)}`;
}

function skipShare(byDecision) {
  const rows = Array.isArray(byDecision) ? byDecision : [];
  const total = rows.reduce((sum, row) => sum + Number(row.count || 0), 0);
  const skip = rows.find((row) => String(row.key).toUpperCase() === "SKIP");
  if (!total || !skip) return "—";
  return formatPct(Number(skip.count || 0) / total);
}

function formatWinRate(rate) {
  if (rate == null || Number.isNaN(rate)) return "—";
  return `${Math.round(rate * 100)}%`;
}

function hitTone(rate) {
  if (rate == null || Number.isNaN(rate)) return "neutral";
  if (rate >= 0.6) return "up";
  if (rate >= 0.45) return "mid";
  return "down";
}

function durationFromMinutes(minutes) {
  const map = { 10: "10m", 30: "30m", 60: "60m", 1440: "1d" };
  return map[minutes] || "10m";
}

function formatPct(rate) {
  if (rate == null || Number.isNaN(rate)) return "—";
  return `${Math.round(rate * 100)}%`;
}
