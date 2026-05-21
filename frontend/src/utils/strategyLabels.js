import { factorLabel } from "./factorLearningLabels";

/** 与后端 strategy_registry 策略 key 对齐的展示名 */
export const STRATEGY_LABELS = {
  manual: "手动",
  factor_combo_ranker_v1: "多因子组合胜率榜",
  factor_combo_ranker_v1_top2: "多因子组合胜率榜·Top2",
  factor_combo_ranker_v1_top3: "多因子组合胜率榜·Top3",
  high_winrate_factor_combo_v1: "高胜率目标组合",
  high_winrate_factor_combo_v1_top2: "高胜率目标组合·Top2",
  high_winrate_factor_combo_v1_top3: "高胜率目标组合·Top3",
  ensemble_ranker_v1: "综合裁判模拟策略",
  optimized_rules_10m: "优化规则集（回测）",
};

/** 已下线但仍可能出现在历史预测/排名中的策略 key */
const LEGACY_STRATEGY_LABELS = {
  complete_day_10m_production: "全日规则·10分钟",
  vegas_fib_resonance: "维加斯斐波共振",
  high_winrate_rules: "高胜率规则集",
  pure_rule_precision: "纯规则精度",
  win70_trade_max_rules: "70%胜率交易上限规则",
  daily_trade_floor_tree: "日交易下限树规则",
  orderbook_notional_40m: "订单簿名义额·40分钟",
  orderbook_notional_40m_mg: "订单簿名义额·40分钟·马丁",
  orderbook_notional_10m: "订单簿名义额·10分钟",
  orderbook_notional_10m_mg_5102045: "订单簿名义额·10分钟·马丁",
  orderbook_notional_15m: "订单簿名义额·15分钟",
  orderbook_notional_15m_mg_51020: "订单簿名义额·15分钟·马丁",
  orderbook_trade_flow_1k: "订单簿成交流·1k",
  orderbook_trade_flow_1k_invert_mg: "订单簿成交流·1k·反向马丁",
  blind_reverse_martingale_v1: "盲反马丁",
  three_bar_10m_reverse_martingale_v1: "三根K线·10分钟·反马丁",
  four_bar_10m_reverse_martingale_v1: "四根K线·10分钟·反马丁",
  five_bar_10m_reverse_martingale_v1: "五根K线·10分钟·反马丁",
};

const MODEL_FAMILY_LABELS = {
  lstm: "LSTM",
  gru: "GRU",
  cnn: "CNN",
  transformer: "Transformer",
  random_forest: "随机森林",
  xgboost: "XGBoost",
  svm: "SVM",
  bayesian: "贝叶斯",
  knn: "KNN",
  rl_strategy: "强化学习",
};

const DURATION_LABELS = {
  "10m": "10分钟",
  "30m": "30分钟",
  "60m": "1小时",
  "1d": "1天",
};

const TOKEN_LABELS = {
  orderbook: "订单簿",
  notional: "名义额",
  trade: "成交",
  flow: "流",
  invert: "反向",
  reverse: "反",
  martingale: "马丁",
  mg: "马丁",
  blind: "盲",
  bar: "K线",
  three: "三",
  four: "四",
  five: "五",
  production: "生产",
  rules: "规则",
  combo: "组合",
  ranker: "排名",
  factor: "因子",
  high: "高",
  winrate: "胜率",
  ensemble: "综合",
  shadow: "影子",
  optimized: "优化",
  simulation: "模拟",
  agent: "Agent",
};

const BATCH_COMBO_PREFIX = "factor_combo_ranker_v1_combo_";
const BATCH_HIGH_WINRATE_PREFIX = "high_winrate_factor_combo_v1_combo_";

export function strategyLabel(key) {
  const raw = String(key || "").trim();
  if (!raw) return STRATEGY_LABELS.manual;

  const exact =
    STRATEGY_LABELS[raw] ||
    STRATEGY_LABELS[raw.toLowerCase()] ||
    LEGACY_STRATEGY_LABELS[raw] ||
    LEGACY_STRATEGY_LABELS[raw.toLowerCase()];
  if (exact) return exact;

  const model = modelShadowLabel(raw);
  if (model) return model;

  const batch = batchComboLabel(raw);
  if (batch) return batch;

  const top = topShadowLabel(raw);
  if (top) return top;

  if (looksLikeFactorName(raw)) {
    return factorLabel(raw);
  }

  return humanizeStrategyKey(raw);
}

function modelShadowLabel(key) {
  const lowered = key.toLowerCase();
  for (const [family, label] of Object.entries(MODEL_FAMILY_LABELS)) {
    const prefix = `factor_${family}_shadow_`;
    if (lowered.startsWith(prefix)) {
      const duration = key.slice(prefix.length);
      return `${label}影子·${durationLabel(duration)}`;
    }
  }
  return "";
}

function batchComboLabel(key) {
  const lowered = key.toLowerCase();
  if (lowered.startsWith(BATCH_COMBO_PREFIX)) {
    const suffix = key.slice(BATCH_COMBO_PREFIX.length);
    return suffix ? `多因子批量模拟·${suffix.slice(-4)}` : "多因子批量模拟";
  }
  if (lowered.startsWith(BATCH_HIGH_WINRATE_PREFIX)) {
    const suffix = key.slice(BATCH_HIGH_WINRATE_PREFIX.length);
    return suffix ? `高胜率批量模拟·${suffix.slice(-4)}` : "高胜率批量模拟";
  }
  return "";
}

function topShadowLabel(key) {
  const match = key.match(/^factor_combo_ranker_v1_top(\d+)$/i);
  if (match) return `多因子组合胜率榜·Top${match[1]}`;
  const hw = key.match(/^high_winrate_factor_combo_v1_top(\d+)$/i);
  if (hw) return `高胜率目标组合·Top${hw[1]}`;
  return "";
}

function looksLikeFactorName(key) {
  const lowered = key.toLowerCase();
  return (
    lowered.startsWith("combo__") ||
    lowered.startsWith("goal_combo__") ||
    /^[a-z]+_\d+/.test(lowered) ||
    lowered.startsWith("agent__")
  );
}

function humanizeStrategyKey(key) {
  const parts = key
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .toLowerCase()
    .split("_")
    .filter(Boolean);
  if (!parts.length) return key;

  const translated = [];
  for (const part of parts) {
    if (DURATION_LABELS[part]) {
      translated.push(DURATION_LABELS[part]);
      continue;
    }
    if (/^v\d+$/.test(part)) continue;
    if (/^\d+[km]?$/.test(part)) {
      translated.push(part);
      continue;
    }
    if (TOKEN_LABELS[part]) {
      translated.push(TOKEN_LABELS[part]);
      continue;
    }
    if (/^\d+$/.test(part)) {
      translated.push(`${part}周期`);
      continue;
    }
  }

  if (translated.length) {
    return translated.join("");
  }
  return key.replace(/_/g, " ");
}

function durationLabel(duration) {
  return DURATION_LABELS[String(duration || "").toLowerCase()] || duration || "—";
}
