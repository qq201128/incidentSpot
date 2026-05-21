/** 与后端 strategy_registry 策略 key 对齐的展示名 */
export const STRATEGY_LABELS = {
  manual: "手动",
  factor_combo_ranker_v1: "多因子组合胜率榜",
  factor_combo_ranker_v1_top2: "多因子组合胜率榜·Top2",
  factor_combo_ranker_v1_top3: "多因子组合胜率榜·Top3",
  high_winrate_factor_combo_v1: "高胜率目标组合",
  high_winrate_factor_combo_v1_top2: "高胜率目标组合·Top2",
  high_winrate_factor_combo_v1_top3: "高胜率目标组合·Top3",
};

export function strategyLabel(key) {
  const model = modelShadowLabel(key);
  if (model) {
    return model;
  }
  return STRATEGY_LABELS[key] || key || STRATEGY_LABELS.manual;
}

function modelShadowLabel(key) {
  const families = {
    lstm: "LSTM",
    gru: "GRU",
    cnn: "CNN",
    transformer: "Transformer",
    random_forest: "RandomForest",
    xgboost: "XGBoost",
    svm: "SVM",
    bayesian: "Bayesian",
    knn: "KNN",
    rl_strategy: "RL策略",
  };
  for (const [family, label] of Object.entries(families)) {
    const prefix = `factor_${family}_shadow_`;
    if (key?.startsWith(prefix)) return `${label}影子策略·${key.replace(prefix, "")}`;
  }
  return "";
}
