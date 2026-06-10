import {
  EVIDENCE_TARGETS,
  EMPTY,
  TOP_ROW_LIMIT,
  candidateTypeLabel,
  formatDateTime,
  formatGap,
  formatNum,
  formatPct,
  formatWindow,
  prefilterRows,
  stageLogRows,
  statusChangeRows,
  topReasons,
  visibleSettledRows,
} from "./researchDashboardData";
import ModelRegimeValidationPanel from "../components/ModelRegimeValidationPanel";
import { reasonLabel, statusClass, statusLabel } from "./researchDashboardLabels";

export function SettledSampleMatrix({ duration, loadError, loading, reportLoaded, rows, symbol }) {
  const candidateCountText = matrixStatusText({
    loadError,
    loading,
    reportLoaded,
    rowCount: rows.length,
  });
  return (
    <section className="research-matrix">
      <header className="research-section-head">
        <div>
          <span className="section-kicker">Settled sample matrix</span>
          <h2>结算样本矩阵</h2>
        </div>
        <small>{candidateCountText}</small>
      </header>
      {rows.length ? (
        <div className="research-table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>候选</th>
                <th>状态</th>
                <th>样本</th>
                <th>胜率</th>
                <th>近30</th>
                <th>近60</th>
                <th>近100</th>
                <th>PF</th>
                <th>均收益</th>
                <th>回测差</th>
                <th>连续亏损</th>
                <th>数据/特征</th>
                <th>原因</th>
              </tr>
            </thead>
            <tbody>
              {visibleSettledRows(rows, TOP_ROW_LIMIT).map((row) => (
                <SettledRow key={row.rowKey} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="research-empty" role={loadError ? "alert" : undefined}>
          {matrixEmptyText(loadError, loading, reportLoaded)}
        </p>
      )}
    </section>
  );
}

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

function SettledRow({ row }) {
  const gapRisk = row.backtestGap != null && row.backtestGap >= EVIDENCE_TARGETS.backtestGapWarn;
  return (
    <tr>
      <td>
        <div className="research-candidate-cell">
          <strong title={row.name}>{row.name}</strong>
          <span>{candidateTypeLabel(row)}</span>
        </div>
      </td>
      <td><StatusPill status={row.status} /></td>
      <td>{sampleCountText(row)}</td>
      <td className={metricClass(row.winRate, EVIDENCE_TARGETS.winRateMin)}>{formatPct(row.winRate)}</td>
      <td>{formatWindow(row.windows?.recent30)}</td>
      <td>{formatWindow(row.windows?.recent60)}</td>
      <td>{formatWindow(row.windows?.recent100)}</td>
      <td className={metricClass(row.profitFactor, EVIDENCE_TARGETS.profitFactorMin)}>{formatNum(row.profitFactor, 2)}</td>
      <td className={metricClass(row.avgReturn, 0)}>{formatPct(row.avgReturn)}</td>
      <td className={gapRisk ? "is-warn" : ""}>{formatGap(row)}</td>
      <td className={row.maxConsecutiveLosses >= EVIDENCE_TARGETS.lossStreakLimit ? "is-bad" : ""}>{row.maxConsecutiveLosses}</td>
      <td>
        <span className="research-state-stack">
          <small>{row.dataFreshnessStatus || EMPTY}</small>
          <small>{row.missingFeatureStatus || EMPTY}</small>
        </span>
      </td>
      <td><span className="research-reason">{reasonLabel(row.reason)}</span></td>
    </tr>
  );
}

function StatusPill({ status }) {
  return <span className={`research-status-pill ${statusClass(status)}`}>{statusLabel(status)}</span>;
}

function sampleCountText(row) {
  if (row.type === "model" && row.sampleCount > 0 && row.validationSampleCount > 0) return `${row.sampleCount} / 验证 ${row.validationSampleCount}`;
  if (row.sampleCount > 0) return row.sampleCount;
  if (row.type === "model" && row.validationSampleCount > 0) return `验证 ${row.validationSampleCount}`;
  return row.sampleCount;
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

function metricClass(value, target) {
  if (value == null) return "";
  return Number(value) >= target ? "is-good" : "is-bad";
}

function matrixStatusText({ loadError, loading, reportLoaded, rowCount }) {
  if (loadError) return "候选报告读取失败";
  if (loading) return "正在读取候选报告";
  if (reportLoaded) return `${rowCount} 个候选，含模型族观察行`;
  return "等待报告返回";
}

function matrixEmptyText(loadError, loading, reportLoaded) {
  if (loadError) return `候选报告读取失败：${loadError}`;
  if (loading) return "正在读取候选报告…";
  if (reportLoaded) return "暂无候选或模型族状态";
  return "尚未返回候选报告";
}
