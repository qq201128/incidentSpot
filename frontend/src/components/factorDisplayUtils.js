export function directionLabel(value) {
  if (value === "higher_better") return "高优";
  if (value === "lower_better") return "低优";
  return "中性";
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
