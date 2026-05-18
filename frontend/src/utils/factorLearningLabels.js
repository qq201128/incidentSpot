const CATEGORY_LABELS = {
  return: "收益率",
  volatility: "波动率",
  moving_average: "均线趋势",
  momentum: "动量",
  volume: "成交量",
  structure: "K线结构",
  multi_timeframe: "多周期",
  orderbook: "订单簿",
  funding: "资金费率",
  positioning: "合约持仓",
  taker_flow: "主动买卖",
  smc: "结构交易",
  sentiment: "情绪",
  statistic: "统计",
  onchain: "链上",
  performance: "组合表现",
};

const OPERATOR_LABELS = {
  abs: "绝对值",
  pct_change: "变化率",
  diff: "差分",
  daily_volume: "日成交量",
  ADX: "ADX趋势强度",
  adx: "ADX趋势强度",
  Add: "加法",
  Sub: "差值",
  Mul: "交互项",
  Div: "比率",
  Log: "对数压缩",
  Sign: "方向符号",
  Mean: "均值",
  Std: "标准差",
  Max: "最大值",
  Min: "最小值",
  Rank: "排名",
  TsRank: "时间序列排名",
  TsZScore: "历史标准分",
  Delta: "变化量",
  Corr: "滚动相关",
};

const EXACT_FACTOR_LABELS = {
  tf_1d_volume_share: "1天成交量占比",
  dollar_volume_ma_20: "20周期成交额均值",
  vol_median_ratio_20: "20周期成交量中位比",
};

const TOKEN_LABELS = {
  ret: "收益率",
  vol: "成交量",
  ma: "均线",
  ratio: "比率",
  realized: "已实现",
  skew: "偏度",
  bb: "布林带",
  width: "宽度",
  efficiency: "效率",
  tf: "多周期",
  volume: "成交量",
  share: "占比",
  dollar: "成交额",
  median: "中位数",
  adx: "趋势强度",
};

export function learningPatternLabel(item, key) {
  const value = String(item?.[key] || item?.pattern || "");
  if (value.startsWith("category=")) return `${categoryLabel(value.slice(9))}因子族`;
  if (value.startsWith("category:")) return `${categoryLabel(value.slice(9))}因子族`;
  if (value.startsWith("operator=")) return `${operatorLabel(value.slice(9))}算子`;
  if (value.startsWith("operator:")) return `${operatorLabel(value.slice(9))}算子`;
  if (value.startsWith("correlation_cluster:")) {
    return `${factorLabel(value.slice(20))}相关性禁区`;
  }
  if (key === "feature") return factorLabel(value);
  return value ? factorLabel(value) : "—";
}

export function factorLabel(name) {
  const raw = String(name || "");
  if (!raw) return "—";
  if (EXACT_FACTOR_LABELS[raw]) return EXACT_FACTOR_LABELS[raw];
  if (raw.startsWith("combo__")) return comboLabel(raw);
  return regexFactorLabel(raw) || tokenizedLabel(raw);
}

export function operatorLabel(name) {
  const raw = String(name || "");
  return OPERATOR_LABELS[raw] || OPERATOR_LABELS[raw.toLowerCase()] || tokenizedLabel(raw);
}

export function categoryLabel(name) {
  const raw = String(name || "");
  return CATEGORY_LABELS[raw] || tokenizedLabel(raw);
}

export function operatorTraceLabel(items) {
  if (!Array.isArray(items)) return [];
  return items.filter(Boolean).map(operatorLabel);
}

export function columnLabels(items) {
  if (!Array.isArray(items)) return [];
  return items.filter(Boolean).map(factorLabel);
}

function regexFactorLabel(raw) {
  const rules = [
    [/^ret_(\d+)$/, (n) => `${n}周期收益率`],
    [/^ma_ratio_(\d+)$/, (n) => `${n}周期均线偏离`],
    [/^vol_ma_(\d+)$/, (n) => `${n}周期成交量均值`],
    [/^bb_width_(\d+)$/, (n) => `布林带宽度（${n}周期）`],
    [/^adx_(\d+)$/, (n) => `ADX趋势强度（${n}周期）`],
    [/^efficiency_ratio_(\d+)$/, (n) => `效率比率（${n}周期）`],
    [/^realized_skew_(\d+)$/, (n) => `${n}周期收益偏度`],
    [/^dollar_volume_ma_(\d+)$/, (n) => `${n}周期成交额均值`],
    [/^vol_median_ratio_(\d+)$/, (n) => `${n}周期成交量中位比`],
  ];
  for (const [pattern, render] of rules) {
    const match = raw.match(pattern);
    if (match) return render(match[1]);
  }
  return "";
}

function comboLabel(raw) {
  const names = comboLeafNames(raw).map(factorLabel);
  return `组合：${names.join(" + ")}`;
}

function comboLeafNames(raw) {
  const seen = new Set();
  const names = [];
  raw.split("__")
    .filter((part) => part && part !== "combo" && part !== "goal_combo")
    .forEach((part) => {
      if (seen.has(part)) return;
      seen.add(part);
      names.push(part);
    });
  return names;
}

function tokenizedLabel(raw) {
  const parts = raw.split("_").filter(Boolean);
  if (!parts.length) return "自定义因子";
  const translated = parts.map((part) => TOKEN_LABELS[part] || numberToken(part)).filter(Boolean);
  return translated.length ? translated.join("") : "自定义因子";
}

function numberToken(value) {
  return /^\d+$/.test(value) ? `${value}周期` : "";
}
