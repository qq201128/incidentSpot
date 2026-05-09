const MODEL_LABELS = {
  "10m_enhanced": "10分钟增强模型",
  "10m": "10分钟基础模型",
  "30m": "30分钟基础模型",
  "60m": "60分钟基础模型",
  "1d": "1天基础模型",
};

const METRIC_LABELS = {
  "test AUC": "测试区分度",
  "test F1": "测试综合分数",
  "calibrated Brier": "校准误差",
  "calibrated logloss": "校准损失",
  "backtest win rate": "回测胜率",
  "direction hit rate": "方向命中率",
  "average trade return": "平均交易收益",
};

export function formatTime(value) {
  if (!value) return "--";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return "--";
  return dt.toLocaleString("zh-CN", { hour12: false });
}

export function formatMetric(value, mode = "number") {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  if (mode === "percent") return `${(n * 100).toFixed(1)}%`;
  if (mode === "price") return n.toFixed(4);
  if (mode === "signed") return `${n >= 0 ? "+" : ""}${(n * 100).toFixed(3)}%`;
  return n.toFixed(3);
}

export function metricValue(metrics, path) {
  return path.split(".").reduce((node, key) => {
    if (!node || typeof node !== "object") return undefined;
    return node[key];
  }, metrics);
}

export function statusText(status) {
  const map = {
    active: "当前",
    archived: "历史",
    rejected: "未发布",
    published: "已发布",
    failed: "失败",
  };
  return map[status] || status || "--";
}

export function runStatusText(status) {
  const map = {
    completed: "完成",
    completed_with_errors: "完成但有错误",
    completed_with_rejections: "完成但有未发布模型",
    failed: "失败",
  };
  return map[status] || status || "--";
}

export function errorText(err, fallback) {
  const raw = err?.response?.data?.detail || err?.message || fallback;
  const map = {
    "model training is already running": "模型训练正在运行",
    "model version not found": "找不到这个模型版本",
    "rejected model versions cannot be activated": "未发布模型不能启用",
  };
  return map[raw] || raw;
}

export function modelLabel(item) {
  return MODEL_LABELS[item?.key] || item?.label || "--";
}

export function metricLabel(label) {
  return METRIC_LABELS[label] || label || "--";
}
