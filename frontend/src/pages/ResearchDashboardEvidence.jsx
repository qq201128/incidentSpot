import {
  EMPTY,
  formatDateTime,
  formatNum,
  formatPct,
  prefilterRows,
  stageLogRows,
  statusChangeRows,
  topReasons,
} from "./researchDashboardData";
import ModelRegimeValidationPanel from "../components/ModelRegimeValidationPanel";
import { reasonLabel, statusLabel } from "./researchDashboardLabels";

export function ResearchSidePanel({ report, rows, summary, modelStatuses }) {
  const reasons = topReasons(rows, report);
  const models = rows.filter((row) => row.type === "model");
  const stages = stageLogRows(report).filter((row) => row.status !== "passed").slice(0, 5);
  const changes = statusChangeRows(report).slice(0, 5);
  const statuses = modelStatuses?.length ? modelStatuses : report?.modelFamilyStatuses || [];
  return (
    <aside className="research-side-panel">
      <ModelRegimeValidationPanel statuses={statuses} />
      <LifecyclePanel summary={summary} />
      <SampleQualityPanel summary={summary} />
      <ReasonPanel reasons={reasons} />
      <StatusChangePanel rows={changes} />
      <StageLogPanel rows={stages} />
      <ModelEvidencePanel models={models} />
      <PrefilterPanel rows={prefilterRows(report)} />
    </aside>
  );
}

function LifecyclePanel({ summary }) {
  if (!summary.reportLoaded) {
    return (
      <section className="research-side-section">
        <h3>生命周期</h3>
        <p className="research-empty small">等待 paper-live 报告返回</p>
      </section>
    );
  }
  const total = Math.max(summary.stableCount + summary.collectingCount + summary.failedCount, 1);
  return (
    <section className="research-side-section">
      <h3>生命周期</h3>
      <div className="research-lifecycle-bars">
        <LifecycleBar label="稳定" value={summary.stableCount} samples={summary.stableSampleCount} total={total} tone="stable" />
        <LifecycleBar label="观察" value={summary.collectingCount} samples={summary.collectingSampleCount} total={total} tone="collecting" />
        <LifecycleBar label="失败" value={summary.failedCount} samples={summary.failedSampleCount} total={total} tone="failed" />
      </div>
      <p className="research-side-note">待结算候选 {summary.unsettledCandidateCount} 个，未进入本视图核心统计。</p>
    </section>
  );
}

function SampleQualityPanel({ summary }) {
  return (
    <section className="research-side-section">
      <h3>样本厚度</h3>
      <div className="research-quality-grid">
        <QualityMetric label="候选均样本" value={formatNum(summary.avgSamplesPerCandidate, 1)} />
        <QualityMetric label="样本≥30" value={summary.sampleRichCandidateCount} />
        <QualityMetric label="模型族证据" value={summary.modelEvidenceCount} />
        <QualityMetric label="近期偏弱" value={summary.recentWeakCount} />
        <QualityMetric label="回测落差" value={summary.backtestGapRiskCount} />
        <QualityMetric label="状态变化" value={summary.statusChangeCount} />
        <QualityMetric label="数据异常" value={summary.dataIssueCount} />
        <QualityMetric label="特征异常" value={summary.featureIssueCount} />
      </div>
    </section>
  );
}

function ReasonPanel({ reasons }) {
  return (
    <section className="research-side-section">
      <h3>失败原因</h3>
      {reasons.length ? (
        <ul className="research-reason-list">
          {reasons.map((row) => (
            <li key={row.reason}>
              <span>{reasonLabel(row.reason)}</span>
              <b>{row.count}</b>
            </li>
          ))}
        </ul>
      ) : <p className="research-empty small">暂无失败原因</p>}
    </section>
  );
}

function StatusChangePanel({ rows }) {
  return (
    <section className="research-side-section">
      <h3>生命周期变化</h3>
      {rows.length ? rows.map((row) => (
        <article key={`${row.candidateKey}-${row.changedAt}`} className="research-event-row">
          <strong title={row.candidateKey}>{row.candidateKey || EMPTY}</strong>
          <span>{statusLabel(row.oldStatus)} → {statusLabel(row.newStatus)} · {reasonLabel(row.reason)}</span>
          <small>{formatDateTime(row.changedAt)}</small>
        </article>
      )) : <p className="research-empty small">暂无状态变化</p>}
    </section>
  );
}

function StageLogPanel({ rows }) {
  return (
    <section className="research-side-section">
      <h3>阶段异常</h3>
      {rows.length ? rows.map((row) => (
        <article key={`${row.signalKey}-${row.stage}-${row.createdAt}`} className="research-event-row">
          <strong title={row.signalKey}>{row.stage || EMPTY}</strong>
          <span>{row.status || EMPTY} · {reasonLabel(row.reason)}</span>
          <small>{formatDateTime(row.createdAt)}</small>
        </article>
      )) : <p className="research-empty small">暂无 pending/failed 阶段日志</p>}
    </section>
  );
}

function ModelEvidencePanel({ models }) {
  return (
    <section className="research-side-section">
      <h3>模型族证据</h3>
      {models.length ? models.map((row) => (
        <article key={row.rowKey} className="research-model-evidence">
          <strong>{row.name}</strong>
          <span>{modelEvidenceText(row)}</span>
        </article>
      )) : <p className="research-empty small">暂无模型族结算样本</p>}
    </section>
  );
}

function modelEvidenceText(row) {
  const samples = row.sampleCount > 0 ? `${row.sampleCount} 纸盘 / 验证 ${row.validationSampleCount || 0}` : `验证 ${row.validationSampleCount || 0} 样本`;
  return `${samples} · ${formatPct(row.winRate)} · ${statusLabel(row.status)} · ${reasonLabel(row.reason)}`;
}

function PrefilterPanel({ rows }) {
  return (
    <section className="research-side-section">
      <h3>预筛背景</h3>
      <p className="research-side-note">仅展示未结算候选的 OOS/回测背景，不参与主表证据。</p>
      {rows.length ? rows.map((row) => (
        <article key={row.rowKey} className="research-prefilter-row">
          <strong title={row.name}>{row.name}</strong>
          <span>OOS {formatPct(row.oosWinRate)} · 回测 {formatPct(row.backtestWinRate)}</span>
        </article>
      )) : <p className="research-empty small">暂无未结算预筛候选</p>}
    </section>
  );
}

function LifecycleBar({ label, value, samples, total, tone }) {
  return (
    <span className="research-lifecycle-row">
      <small>{label}</small>
      <i><b className={`is-${tone}`} style={{ width: `${Math.round((value / total) * 100)}%` }} /></i>
      <strong>{value}</strong>
      <em>{samples} samples</em>
    </span>
  );
}

function QualityMetric({ label, value }) {
  return (
    <span>
      <b>{value ?? 0}</b>
      <small>{label}</small>
    </span>
  );
}
