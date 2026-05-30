import {
  EMPTY,
  DURATIONS,
  SYMBOLS,
  EVIDENCE_TARGETS,
  formatPct,
} from "./researchDashboardData";

export function ResearchHeader({
  duration,
  loading,
  onDurationChange,
  onRunDailyLoop,
  onSymbolChange,
  status,
  symbol,
}) {
  return (
    <header className="research-topbar">
      <div className="research-title">
        <span className="eyebrow">Paper-live Research /</span>
        <h1>研究驾驶舱</h1>
        <p>{symbol} · {duration} · settled sample first</p>
      </div>
      <div className="research-controls" aria-label="研究过滤条件">
        <label>
          <span>交易对</span>
          <select value={symbol} onChange={(event) => onSymbolChange(event.target.value)}>
            {SYMBOLS.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label>
          <span>周期</span>
          <select value={duration} onChange={(event) => onDurationChange(event.target.value)}>
            {DURATIONS.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <button type="button" className="research-run-button" disabled={loading} onClick={onRunDailyLoop}>
          {loading ? "运行中" : "运行日闭环"}
        </button>
      </div>
      <div className="research-status" role="status">
        <span className={`status-dot${loading ? " status-dot--muted" : ""}`} />
        <p>{status}</p>
      </div>
    </header>
  );
}

export function SummaryStrip({ summary }) {
  const metrics = summary.reportLoaded ? summary : null;
  return (
    <section className="research-summary-area" aria-label="结算样本概览">
      <div className="research-evidence-card">
        <span className="section-kicker">Evidence priority</span>
        <strong>只用已结算 paper-live 样本判断稳定性</strong>
        <p>回测和 OOS 仅保留为预筛背景；主表排序、生命周期和风险提示都来自 settled sample。</p>
      </div>
      <div className="research-summary-strip">
        <SummaryMetric label="已结算样本" value={metrics?.sampleCount} strong />
        <SummaryMetric label="有证据候选" value={metrics?.settledCandidateCount} />
        <SummaryMetric label="结算覆盖" value={formatPct(metrics?.settledCoverage)} />
        <SummaryMetric label={`胜率目标 ${formatPct(EVIDENCE_TARGETS.winRateMin)}`} value={formatPct(metrics?.weightedWinRate)} />
        <SummaryMetric label="稳定候选" value={metrics?.stableCount} tone="stable" />
        <SummaryMetric label="观察候选" value={metrics?.collectingCount} tone="collecting" />
        <SummaryMetric label="失败候选" value={metrics?.failedCount} tone="failed" />
        <SummaryMetric label="显式失败" value={explicitFailureCount(metrics)} tone="failed" />
      </div>
    </section>
  );
}

function explicitFailureCount(summary) {
  if (!summary) return null;
  return summary.predictionFailureCount + summary.stageFailureCount;
}

function SummaryMetric({ label, value, strong = false, tone = "" }) {
  const display = value == null ? EMPTY : value;
  return (
    <span className={`research-summary-metric ${tone ? `is-${tone}` : ""}${strong ? " is-strong" : ""}`}>
      <b>{display}</b>
      <small>{label}</small>
    </span>
  );
}
