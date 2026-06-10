import { useCallback, useEffect, useState } from "react";
import { fetchEventFinalDecisionLatest } from "../api/eventFinalDecisionClient";
import { strategyDurationLabel } from "../utils/strategyLabels";
import {
  finalDecisionLabel,
  formatFinalScore,
  formatProbabilityUp,
  regimePartLabel,
  settlementResultLabel,
} from "../utils/eventFinalDecisionLabels";
import "./EventFinalDecisionPanel.css";

const POLL_MS = 5000;

export default function EventFinalDecisionPanel({ symbol, duration }) {
  const [latest, setLatest] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async (signal) => {
    try {
      const data = await fetchEventFinalDecisionLatest(symbol, duration, { signal });
      if (!signal?.aborted) {
        setLatest(data);
        setError("");
      }
    } catch (err) {
      if (signal?.aborted || err?.code === "ERR_CANCELED") return;
      setError(err?.response?.data?.detail || err?.message || "读取最终裁判失败");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [duration, symbol]);

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

  if (loading && !latest) {
    return <p className="event-final-empty" role="status">正在读取事件最终裁判…</p>;
  }
  if (error) {
    return <p className="event-final-alert is-error" role="alert">{error}</p>;
  }
  if (!latest) {
    return (
      <p className="event-final-empty" role="status">
        当前周期尚无最终裁判记录；自动策略运行后会出现 UP / DOWN / SKIP 审计结果。
      </p>
    );
  }

  const decisionClass = decisionTone(latest.decision);
  return (
    <section className="event-final-panel" aria-label="事件最终裁判">
      <header className="event-final-head">
        <div>
          <h3>事件最终裁判</h3>
          <small>
            {symbol} · {strategyDurationLabel(duration)} · 周期 #{latest.openTime || "—"}
          </small>
        </div>
        <span className={`event-final-decision-pill ${decisionClass}`}>
          {finalDecisionLabel(latest.decision)}
        </span>
      </header>
      <dl className="event-final-grid">
        <Metric label="趋势环境" value={regimePartLabel(latest.trendRegime)} />
        <Metric label="波动环境" value={regimePartLabel(latest.volRegime)} />
        <Metric label="环境置信度" value={formatFinalScore(latest.confidence)} />
        <Metric label="probability_up" value={formatProbabilityUp(latest.probabilityUp)} />
        <Metric label="final_score" value={formatFinalScore(latest.finalScore)} />
        <Metric label="候选模型数" value={latest.candidateCount ?? "—"} />
        <Metric label="最近结算" value={settlementResultLabel(latest.decisionCorrect)} />
      </dl>
      {latest.decision === "SKIP" && latest.skipReason ? (
        <p className="event-final-skip">SKIP 原因：{latest.skipReason}</p>
      ) : null}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function decisionTone(decision) {
  const key = String(decision || "").toUpperCase();
  if (key === "UP") return "is-up";
  if (key === "DOWN") return "is-down";
  return "is-skip";
}
