export { modelFamilyLabel } from "../utils/modelFamilies";

export function statusLabel(status) {
  const labels = {
    shadow_active: "影子激活",
    trade_active: "交易激活",
    initial_baseline: "初始基线",
    promoted_shadow_active: "已发布影子",
    promoted_trade_active: "已发布交易",
    promoted_initial_baseline: "已发布基线",
    rejected_validation: "验证拒绝",
    rejected_insufficient_samples: "样本拒绝",
    queued: "排队中",
    training: "训练中",
    trained: "已训练",
    validation_failed: "验证失败",
    insufficient_samples: "样本不足",
    failed: "训练失败",
  };
  return labels[status] || "未训练";
}

export function blockedReasonLabel(reason) {
  const labels = {
    passed: "—",
    torch_unavailable: "Torch不可用",
    artifacts_incomplete: "模型文件不完整",
    model_status_untrained: "模型未训练",
    validation_gate_missing: "验证闸门缺失",
    no_validation_confidence_threshold_met: "未达验证门槛",
    dependency_unavailable: "依赖不可用",
    trained_combo_snapshot_missing: "训练组合快照缺失",
    trained_combo_snapshot_incomplete: "训练组合不足Top3",
    current_combo_snapshot_missing: "当前组合排名缺失",
    current_combo_snapshot_incomplete: "当前组合不足Top3",
  };
  return labels[reason] || reason || "—";
}

export function comboStatusLabel(shadow) {
  const reason = shadow?.comboSnapshotReason;
  const labels = {
    passed: "一致",
    trained_combo_snapshot_missing: hasPublishedModel(shadow) ? "训练快照缺失" : "等待发布快照",
    trained_combo_snapshot_incomplete: "训练快照不足",
    current_combo_snapshot_missing: "当前排名缺失",
    current_combo_snapshot_incomplete: "当前排名不足",
    combo_snapshot_mismatch: "已变化，继续学习",
  };
  return labels[reason] || reason || "—";
}

export function publishedSnapshotLabel(shadow) {
  const count = snapshotCount(shadow?.comboSnapshotTrained);
  if (count) return topCountLabel(shadow.comboSnapshotTrained);
  return hasPublishedModel(shadow) ? "缺失" : "未发布";
}

export function recentSnapshotLabel(shadow) {
  const attempt = shadow?.lastTrainingAttempt || {};
  return topCountLabel(attempt.comboSnapshot);
}

export function topCountLabel(rows) {
  const count = snapshotCount(rows);
  return count ? `Top${count}` : "—";
}

export function statusClass(status) {
  if (isTrainedStatus(status)) return "is-trained";
  return ["failed", "validation_failed", "rejected_validation"].includes(status) ? "is-failed" : "";
}

export function isTrainedStatus(status) {
  return ["initial_baseline", "shadow_active", "trade_active", "trained"].includes(status);
}

export function compareLabel(strategyKey) {
  if (strategyKey?.includes("top2")) return "Top2";
  if (strategyKey?.includes("top3")) return "Top3";
  return strategyKey?.startsWith("factor_lstm_shadow") ? "LSTM" : "Top1";
}

export function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString();
}

export function shortVersion(value) {
  if (!value) return "—";
  return String(value).replace(/^lstm_/, "");
}

function snapshotCount(rows) {
  return Array.isArray(rows) ? rows.length : 0;
}

function hasPublishedModel(shadow) {
  return Boolean(shadow?.artifactsReady && (shadow?.modelVersion || isTrainedStatus(shadow?.status)));
}
