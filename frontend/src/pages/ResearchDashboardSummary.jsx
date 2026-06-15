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

export function LiveTradingOverview({ error, overview }) {
  const groups = Array.isArray(overview?.groups) ? overview.groups : [];
  return (
    <section className="research-live-overview" aria-label="全局实盘候选总览">
      <header>
        <span className="section-kicker">Live trading overview</span>
        <strong>实盘开启总览</strong>
        <b>{overview?.activeCount ?? 0}</b>
      </header>
      {error ? (
        <p className="research-live-overview-empty" role="alert">读取失败：{error}</p>
      ) : groups.length ? (
        <div className="research-live-overview-grid">
          {groups.map((group) => (
            <LiveTradingGroup key={`${group.symbol}:${group.duration}`} group={group} />
          ))}
        </div>
      ) : (
        <p className="research-live-overview-empty">当前没有候选开启实盘。</p>
      )}
    </section>
  );
}

function LiveTradingGroup({ group }) {
  const candidates = Array.isArray(group.candidates) ? group.candidates : [];
  return (
    <article className="research-live-overview-group">
      <div>
        <strong>{group.symbol}</strong>
        <span>{group.duration}</span>
        <b>{group.activeCount}</b>
      </div>
      <ul>
        {candidates.map((candidate) => (
          <li key={`${candidate.symbol}:${candidate.duration}:${candidate.strategyKey}`}>
            <span title={candidate.candidateName}>{candidate.candidateName || candidate.strategyKey || EMPTY}</span>
            <small>{candidate.strategyKey || EMPTY}</small>
            <small className={`research-live-runtime ${runtimeTone(candidate)}`}>
              {runtimeLabel(candidate)} · {predictionLabel(candidate)}
            </small>
          </li>
        ))}
      </ul>
    </article>
  );
}

function runtimeLabel(candidate) {
  const reason = candidate?.runtimeReason || "unknown";
  const labels = {
    ready_to_place_order: "可下单",
    waiting_fresh_prediction: "等待新预测",
    waiting_prediction: "待预测",
    waiting_open_position_settled: "等持仓结算",
    confidence_below_threshold: "置信不足",
    production_target_failed: "生产目标未过",
    high_winrate_gate_failed: "胜率门未过",
    quality_gate_failed: "质量门未过",
    signal_condition_not_met: "信号未触发",
    disabled: "已停用",
  };
  return labels[reason] || reason;
}

function predictionLabel(candidate) {
  if (!candidate?.latestPredictionAt) return "无预测";
  const freshness = candidate.latestPredictionFresh ? "新鲜" : "过期";
  return `${freshness} ${formatRuntimeDate(candidate.latestPredictionAt)}`;
}

function formatRuntimeDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return EMPTY;
  return date.toLocaleString();
}

function runtimeTone(candidate) {
  if (candidate?.latestPredictionFresh && candidate?.runtimeReason === "ready_to_place_order") {
    return "is-ready";
  }
  if (candidate?.runtimeReason === "waiting_fresh_prediction" || candidate?.runtimeReason === "waiting_prediction") {
    return "is-waiting";
  }
  return "is-muted";
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
