/**
 * 与后端 FactorCategory 枚举一致（factor_registry.py），顺序保持同步。
 * label 为侧栏/表格用的简短中文名。
 */
export const FACTOR_CATEGORY_CATALOG = Object.freeze([
  { key: "return", label: "收益" },
  { key: "volatility", label: "波动" },
  { key: "moving_average", label: "均线" },
  { key: "momentum", label: "动量" },
  { key: "volume", label: "成交量" },
  { key: "structure", label: "形态" },
  { key: "multi_timeframe", label: "多周期" },
  { key: "orderbook", label: "订单簿" },
  { key: "funding", label: "资金费率" },
  { key: "positioning", label: "持仓" },
  { key: "taker_flow", label: "成交流" },
  { key: "smc", label: "SMC" },
  { key: "sentiment", label: "情绪" },
  { key: "statistic", label: "统计" },
  { key: "onchain", label: "链上" },
  { key: "performance", label: "绩效" },
]);

/** 左栏目录分类 chips（中文展示，key 对应后端 category）。 */
export const SIDEBAR_CATEGORY_CHIPS = Object.freeze([
  { key: "", label: "全部" },
  ...FACTOR_CATEGORY_CATALOG,
]);

const TABLE_CATEGORY_LABELS = Object.freeze(
  Object.fromEntries(FACTOR_CATEGORY_CATALOG.map((item) => [item.key, item.label])),
);

const TABLE_CATEGORY_ALIASES = Object.freeze({
  lstm_shadow: "模型",
});

export function sidebarCategoryLabel(key) {
  const item = SIDEBAR_CATEGORY_CHIPS.find((chip) => chip.key === key);
  return item?.label ?? "全部";
}

export function factorTableCategoryLabel(factor) {
  const name = String(factor?.name || "").toLowerCase();
  if (name.includes("lstm") || factor?.sourceKind === "lstm_shadow") {
    return TABLE_CATEGORY_ALIASES.lstm_shadow;
  }
  const key = String(factor?.category || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
  if (TABLE_CATEGORY_LABELS[key]) return TABLE_CATEGORY_LABELS[key];
  const categoryName = String(factor?.categoryName || "").trim();
  if (categoryName && /[\u4e00-\u9fff]/.test(categoryName)) return categoryName;
  return key || "—";
}

export function catalogLabelForKey(key) {
  const normalized = String(key || "").trim().toLowerCase();
  if (!normalized) return "全部";
  return TABLE_CATEGORY_LABELS[normalized] || normalized;
}

export function toolbarCategoryOptions(categories) {
  const counts = new Map((categories || []).map((item) => [item.key, item.count ?? 0]));
  return SIDEBAR_CATEGORY_CHIPS.map((chip) => ({
    key: chip.key,
    label: chip.label,
    count: chip.key ? counts.get(chip.key) ?? 0 : null,
  }));
}
