export const MODEL_FAMILIES = [
  "lstm",
  "gru",
  "cnn",
  "transformer",
  "random_forest",
  "extra_trees",
  "xgboost",
  "lightgbm",
  "catboost",
  "logistic_elasticnet",
  "svm",
  "bayesian",
  "knn",
  "rl_strategy",
];

const MODEL_FAMILY_LABELS = {
  lstm: "LSTM",
  gru: "GRU",
  cnn: "CNN",
  transformer: "Transformer",
  random_forest: "RandomForest",
  extra_trees: "ExtraTrees",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  logistic_elasticnet: "LogisticElasticNet",
  svm: "SVM",
  bayesian: "GaussianNB",
  knn: "KNN",
  rl_strategy: "QTable方向",
};

export function modelFamilyLabel(family, fallback = "模型族") {
  return MODEL_FAMILY_LABELS[family] || fallback;
}
