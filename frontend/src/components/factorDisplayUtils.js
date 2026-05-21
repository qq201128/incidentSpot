export function directionLabel(value) {
  if (value === "higher_better") return "正向";
  if (value === "lower_better") return "负向";
  return "中性";
}

export function directionDetailLabel(value) {
  if (value === "higher_better") return "正向 (数值越大，信号越偏多)";
  if (value === "lower_better") return "负向 (数值越小，信号越偏多)";
  return "中性";
}

export function sourceLabel(factor) {
  if (factor?.sourceLabel) return factor.sourceLabel;
  const raw = String(factor?.sourceFile || "");
  if (raw === "mined_factor_library.json") return "组合缓存";
  if (raw === "agent_mined_factor_library.json") return "Agent候选";
  if (String(factor?.name || "").toLowerCase().includes("lstm")) return "LSTM影子";
  return raw ? "本地定义" : "—";
}

export function sourceTagClass(factor) {
  const kind = factor?.sourceKind || inferSourceKind(factor);
  if (kind === "agent_candidate") return "is-agent";
  if (kind === "lstm_shadow") return "is-lstm";
  if (kind === "composite_cache") return "is-combo";
  return "is-local";
}

export function sourcePillClass(kind) {
  if (kind === "agent_candidate") return "src-agent";
  if (kind === "lstm_shadow") return "src-lstm";
  if (kind === "composite_cache") return "src-combo";
  return "src-local";
}

function inferSourceKind(factor) {
  const raw = String(factor?.sourceFile || "");
  if (raw === "agent_mined_factor_library.json") return "agent_candidate";
  if (raw === "mined_factor_library.json") return "composite_cache";
  if (String(factor?.name || "").toLowerCase().includes("lstm")) return "lstm_shadow";
  return "local_definition";
}

export function durationLabel(value) {
  const labels = { "10m": "10分钟", "30m": "30分钟", "60m": "60分钟", "1d": "1天" };
  return labels[value] || value;
}

export function formatUpdatedTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleTimeString("zh-CN", { hour12: false });
}

export function copyToClipboard(text) {
  if (!text) return Promise.resolve(false);
  return navigator.clipboard?.writeText(text).then(() => true).catch(() => false);
}

export function factorTitle(factor) {
  return (
    factor?.displayName ||
    factor?.factorDisplayName ||
    factor?.description ||
    factor?.name ||
    "未命名因子"
  );
}

export function formatNum(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

export function formatPct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
