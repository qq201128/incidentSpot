import { factorLabel } from "../utils/factorLearningLabels";
import {
  simulationAmount,
  simulationCandidateTypeLabel,
  simulationLatestEventLabel,
  simulationLatestPnlLabel,
  simulationPnlClass,
  simulationRejectionReasonLabel,
  simulationSlotState,
  simulationSourceLabel,
  simulationThresholdLabel,
} from "./simulationSlotLabels";

export default function SimulationSlotDetails({ slots }) {
  const rows = slots.filter((slot) => slot.simulationStatus);
  if (!rows.length) return null;
  return (
    <div className="simulation-slot-list">
      {rows.map((slot) => (
        <SimulationSlotRow key={`${slot.strategyKey}:${slot.symbol}:${slot.duration}`} slot={slot} />
      ))}
    </div>
  );
}

export function SimulationSlotReports({ reports }) {
  const rows = Array.isArray(reports) ? reports.filter(Boolean) : [];
  if (!rows.length) return null;
  return rows.map((report) => <SimulationSlotReport key={`${report.symbol}:${report.duration}`} report={report} />);
}

function SimulationSlotReport({ report }) {
  const rows = Array.isArray(report?.items) ? report.items : [];
  if (!report) return null;
  return (
    <section className="simulation-slot-report">
      <header className="simulation-slot-report-head">
        <strong>模拟候选池</strong>
        <span>{report.symbol} · {report.duration} · 更新 {report.updatedAt || "—"}</span>
      </header>
      <div className="simulation-slot-summary">
        <span>单因子 {report.singleFactorSlots ?? 0}</span>
        <span>多因子 {report.comboFactorSlots ?? 0}</span>
        <span>启用 {report.enabledSlots ?? 0}</span>
        <span>拒绝 {report.rejectedCount ?? 0}</span>
        <span>{simulationThresholdLabel(report.thresholds)}</span>
      </div>
      <div className="simulation-slot-list">
        {rows.map((status, index) => (
          <SimulationStatusRow key={status.strategyKey || `${status.candidateType}:${status.source}:${index}`} status={status} />
        ))}
        {!rows.length ? <span className="strategy-empty">暂无模拟候选状态</span> : null}
      </div>
    </section>
  );
}

function SimulationSlotRow({ slot }) {
  const status = slot.simulationStatus;
  return <SimulationStatusRow status={status} slot={slot} />;
}

function SimulationStatusRow({ status, slot = {} }) {
  const state = simulationSlotState(status);
  return (
    <div className="simulation-slot-row">
      <span className={`simulation-slot-state ${state.className}`}>
        {state.label}
      </span>
      <span>{simulationCandidateTypeLabel(status.candidateType)} · {simulationSourceLabel(status.source)}</span>
      <span>{slot.duration} · qty {simulationAmount(status.slot?.qty ?? slot.qty)}</span>
      <span>{_factorNameLabel(status.factorName)}</span>
      <span>{simulationLatestEventLabel(status.latestEvent)}</span>
      <span className={simulationPnlClass(status.latestEvent)}>
        {simulationLatestPnlLabel(status.latestEvent)}
      </span>
      {status.rejectionReason ? (
        <span className="simulation-slot-reason">{simulationRejectionReasonLabel(status.rejectionReason)}</span>
      ) : null}
      {status.latestFailure ? (
        <span className="simulation-slot-reason">
          预测失败：{status.latestFailure.reason}
        </span>
      ) : null}
    </div>
  );
}

function _factorNameLabel(value) {
  if (!value) return "候选缓存不可用";
  return factorLabel(value);
}
