const REGIME_LABELS = {
  trend_up: "上升趋势",
  trend_down: "下降趋势",
  range: "震荡",
  uncertain: "趋势不明",
  high_vol: "高波动",
  normal_vol: "正常波动",
  low_vol: "低波动",
  unknown: "未知",
};

const DECISION_LABELS = {
  UP: "做多 UP",
  DOWN: "做空 DOWN",
  SKIP: "跳过 SKIP",
};

const BLOCK_REASON_LABELS = {
  clean_event_retrain_required: "需按新版环境特征重训",
};

export function regimePartLabel(key) {
  return REGIME_LABELS[String(key || "").trim()] || key || "—";
}

export function finalDecisionLabel(decision) {
  const key = String(decision || "").trim().toUpperCase();
  return DECISION_LABELS[key] || key || "—";
}

export function modelBlockReasonLabel(reason) {
  const raw = String(reason || "").trim();
  return BLOCK_REASON_LABELS[raw] || raw || "—";
}

export function formatFinalScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

export function formatProbabilityUp(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n * 100)}%` : "—";
}

export function settlementResultLabel(correct) {
  if (correct === null || correct === undefined) return "待结算";
  return correct ? "正确" : "错误";
}
