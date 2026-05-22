import { formatPct } from "./miningFormatters";

export default function MiningKpiCards({
  summary,
  trainingRules,
  busy,
  onRefreshLocal,
  onRefreshAgent,
  onSearchAll,
}) {
  const accuracy = formatPct(summary?.overallAccuracy, 0);
  const accuracyRatio = Number(summary?.overallAccuracy);
  const meterWidth = Number.isFinite(accuracyRatio) ? `${Math.round(accuracyRatio * 100)}%` : "0%";

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
        <small>并行任务 {summary?.searchParallel ?? "0 / 10"}</small>
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
        </div>
        <div className="mining-kpi-actions">
          <button type="button" disabled={busy === "local"} onClick={onRefreshLocal}>
            {busy === "local" ? "排队中…" : "刷新本地记忆"}
          </button>
          <button type="button" disabled={busy === "agent"} onClick={onRefreshAgent}>
            {busy === "agent" ? "排队中…" : "联网 Agent 挖掘"}
          </button>
          <button type="button" className="is-primary" disabled={busy === "search-all"} onClick={onSearchAll}>
            {busy === "search-all" ? "全部排队中" : "全量搜索全部算法"}
          </button>
        </div>
      </article>
    </section>
  );
}
