import { formatPct } from "./miningFormatters";
import { miningWorkerStatusView } from "./workerStatus";
import "./MiningKpiCards.css";
import "./MiningWorkerStatus.css";

export default function MiningKpiCards({
  summary,
  trainingRules,
  busy,
  onRefreshLocal,
  onRefreshAgent,
  onSearchAll,
  onRetrainAll,
}) {
  const accuracy = formatPct(summary?.overallAccuracy, 0);
  const accuracyRatio = Number(summary?.overallAccuracy);
  const meterWidth = Number.isFinite(accuracyRatio) ? `${Math.round(accuracyRatio * 100)}%` : "0%";
  const worker = miningWorkerStatusView(trainingRules?.workerStatus);

  return (
    <section className="mining-kpi-row">
      <article className="mining-kpi-card mining-kpi-card--accent">
        <span className="mining-kpi-label">总体准确率</span>
        <strong className="mining-kpi-value">{accuracy}</strong>
        <div className="mining-kpi-meter" aria-hidden>
          <i style={{ width: meterWidth }} />
        </div>
        <small>基于最近 30 天验证集</small>
      </article>

      <article className="mining-kpi-card">
        <span className="mining-kpi-label">学习样本</span>
        <strong className="mining-kpi-value">{summary?.sampleCount ?? "—"}</strong>
        <small>
          loss 样本 {summary?.lossSampleCount ?? "—"} | win 样本 {summary?.winSampleCount ?? "—"}
        </small>
      </article>

      <article className="mining-kpi-card">
        <span className="mining-kpi-label">搜索中</span>
        <strong className="mining-kpi-value">{summary?.searchingCount ?? 0}</strong>
        <small>
          {summary?.searchPendingCount ?? 0} pending | {summary?.searchRunningCount ?? 0} running
        </small>
      </article>

      <article className="mining-kpi-card">
        <span className="mining-kpi-label">候选记录</span>
        <strong className="mining-kpi-value">{summary?.candidateRecordCount ?? 0}</strong>
        <small>
          待验证 {summary?.candidatePending ?? 0} | 已完成 {summary?.candidateCompleted ?? 0}
        </small>
      </article>

      <article className="mining-kpi-card mining-kpi-card--rules">
        <div className="mining-kpi-rules-text">
          <span className="mining-kpi-label">训练规则</span>
          <p>{trainingRules?.text || "—"}</p>
          <WorkerStatusSummary worker={worker} />
        </div>
        <div className="mining-kpi-actions">
          <button type="button" disabled={busy === "local"} onClick={onRefreshLocal}>
            {busy === "local" ? "排队中…" : "刷新本地记忆"}
          </button>
          <button type="button" disabled={busy === "agent"} onClick={onRefreshAgent}>
            {busy === "agent" ? "排队中…" : "联网 Agent 挖掘"}
          </button>
          <button type="button" className="is-primary" disabled={busy === "search-all"} onClick={onSearchAll}>
            {busy === "search-all" ? "全部排队中" : "快速补搜全部算法"}
          </button>
          <button type="button" className="is-danger" disabled={busy === "retrain-all"} onClick={onRetrainAll}>
            {busy === "retrain-all" ? "重训排队中" : "重训当前全部"}
          </button>
        </div>
      </article>
    </section>
  );
}

function WorkerStatusSummary({ worker }) {
  return (
    <div className={`mining-worker-status is-${worker.tone}`}>
      <strong>{worker.label}</strong>
      <span>{worker.detail}</span>
      {worker.state === "worker_required" ? <code>{worker.command}</code> : null}
      {worker.latestLogPath ? <span title={worker.latestLogPath}>日志 {worker.latestLogPath}</span> : null}
      {worker.failureReason ? <span title={worker.failureReason}>失败原因 {worker.failureReason}</span> : null}
    </div>
  );
}
