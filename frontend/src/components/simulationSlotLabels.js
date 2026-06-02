import { failureReasonLabel } from "../utils/failureReasonLabels.js";

const SOURCE_LABELS = Object.freeze({
  factor_ranking_cache: "排名缓存",
  agent_mined_factor_library: "Agent因子库",
  factor_combo_ranking_cache: "组合缓存",
  high_winrate_combo_ranking_cache: "高胜率组合缓存",
  auto_trade_strategies: "执行槽位",
});

export function simulationSlotState(status) {
  if (status?.gateStatus === "enabled") return { label: "已启用", className: "enabled" };
  if (status?.gateStatus === "rejected") return { label: "已拒绝", className: "rejected" };
  return { label: "未启用", className: "idle" };
}

export function simulationCandidateTypeLabel(type) {
  if (type === "single_factor") return "单因子";
  if (type === "factor_combo") return "多因子";
  return "候选";
}

export function simulationSourceLabel(source) {
  return SOURCE_LABELS[source] || source || "未知来源";
}

export function simulationRejectionReasonLabel(reason) {
  return failureReasonLabel(reason);
}

export function simulationLatestFailureLabel(failure) {
  if (!failure) return "";
  return `预测失败：${failureReasonLabel(failure.reason)}`;
}

export function simulationThresholdLabel(thresholds) {
  if (!thresholds) return "阈值：—";
  const win = Number(thresholds.minWinRate);
  const pf = Number(thresholds.minProfitFactor);
  const periods = Number(thresholds.minTotalPeriods);
  const winLabel = Number.isFinite(win) ? `${(win * 100).toFixed(0)}%` : "—";
  const pfLabel = Number.isFinite(pf) ? pf : "—";
  const periodLabel = Number.isFinite(periods) ? periods : "—";
  return `阈值：胜率 ${winLabel} / PF ${pfLabel} / 样本 ${periodLabel}`;
}

export function simulationLatestEventLabel(event) {
  if (!event) return "最近事件：无";
  const order = event.orderId ? ` / ORD-${event.orderId}` : "";
  const mode = event.externalStatus === "SIMULATED" ? "SIM" : event.externalStatus || "未知";
  return `最近事件：EVT-${String(event.id).padStart(6, "0")}${order} · ${event.status} · ${mode}`;
}

export function simulationLatestPnlLabel(event) {
  const pnl = Number(event?.settlementPnl);
  if (!Number.isFinite(pnl)) return "最近PnL：未结算";
  return `最近PnL：${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}`;
}

export function simulationPnlClass(event) {
  const pnl = Number(event?.settlementPnl);
  if (!Number.isFinite(pnl)) return "";
  return pnl >= 0 ? "value-up" : "value-down";
}

export function simulationAmount(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3).replace(/\.?0+$/, "") : "—";
}
