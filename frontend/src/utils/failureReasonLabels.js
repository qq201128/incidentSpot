const EMPTY_LABEL = "—";

const REASON_LABELS = Object.freeze({
  cache_table_missing: "缓存表不存在",
  cache_unavailable: "缓存不可用",
  candidate_cache_unavailable: "候选缓存不可用",
  event_status_failed: "事件状态失败",
  factor_candidate_signal_missing_column: "候选因子信号缺少字段",
  factor_candidate_signal_score_history_insufficient: "候选因子信号历史样本不足",
  factor_candidate_source_row_missing: "候选因子信号缺少已收盘K线",
  index_price_unavailable: "最新指数价不可用",
  missing_failure_reason: "失败状态缺少原因",
  model_shadow_not_ready: "模型影子预测未就绪",
  offline_ranking_empty: "离线候选为空",
  order_external_status_failed: "外部订单状态失败",
  order_response_failed: "订单响应失败",
  order_status_failed: "订单状态失败",
  prediction_failed: "预测失败",
  profit_factor_below_min: "盈亏比不足",
  profit_factor_missing: "盈亏比缺失",
  ranking_cache_empty: "排名缓存为空",
  ranking_cache_legacy_without_fingerprint: "排名缓存缺少行情指纹",
  ranking_cache_malformed: "排名缓存格式异常",
  ranking_cache_market_data_changed: "排名缓存行情已变更",
  ranking_cache_missing: "排名缓存缺失",
  real_trading_disabled: "真实交易当前关闭",
  sample_count_below_min: "样本数不足",
  settlement_error: "结算失败",
  timeout: "请求超时",
  validation_gate_failed: "验证闸门未通过",
  validation_gate_missing: "验证闸门缺失",
  validation_profit_factor_below_min: "验证集盈亏比不足",
  validation_sample_count_below_min: "验证集样本数不足",
  validation_win_rate_below_min: "验证集胜率不足",
  win_rate_below_min: "胜率不足",
  win_rate_missing: "胜率缺失",
});

const CACHE_SOURCE_LABELS = Object.freeze({
  factor_combo_ranking_cache: "组合缓存",
  factor_ranking_cache: "排名缓存",
  high_winrate_combo_ranking_cache: "高胜率组合缓存",
});

export function failureReasonLabel(reason) {
  const key = normalizedReason(reason);
  if (!key) return EMPTY_LABEL;
  if (key.startsWith("cache_unavailable:")) return cacheUnavailableLabel(key);
  const direct = REASON_LABELS[key];
  if (direct) return direct;
  const inferred = inferredReasonLabel(key);
  return inferred || "未归类失败原因";
}

export function eventFailureReasonLabel(event) {
  if (!event) return EMPTY_LABEL;
  if (event.failureReason) return failureReasonLabel(event.failureReason);
  if (isFailureText(event.externalResponse)) return failureReasonLabel(event.externalResponse);
  if (isFailureText(event.settlementSource)) return failureReasonLabel(event.settlementSource);
  if (event.status === "FAILED") return failureReasonLabel("event_status_failed");
  if (event.orderStatus === "FAILED") return failureReasonLabel("order_status_failed");
  if (isFailureText(event.externalStatus)) {
    return failureReasonLabel("order_external_status_failed");
  }
  return failureReasonLabel("missing_failure_reason");
}

function normalizedReason(reason) {
  if (reason == null) return "";
  if (typeof reason === "string") return reason.trim();
  return String(reason).trim();
}

function cacheUnavailableLabel(reason) {
  const suffix = reason.split(":").slice(1).join(":").trim();
  const sourceLabel = CACHE_SOURCE_LABELS[suffix] || REASON_LABELS[suffix];
  return sourceLabel ? `缓存不可用：${sourceLabel}` : "缓存不可用";
}

function inferredReasonLabel(reason) {
  const lower = reason.toLowerCase();
  if (lower.includes("factor candidate signal missing column")) {
    return REASON_LABELS.factor_candidate_signal_missing_column;
  }
  if (lower.includes("factor candidate signal has insufficient score history")) {
    return REASON_LABELS.factor_candidate_signal_score_history_insufficient;
  }
  if (lower.includes("factor candidate signal missing completed")) {
    return REASON_LABELS.factor_candidate_source_row_missing;
  }
  if (lower.includes("latest index price unavailable")) return REASON_LABELS.index_price_unavailable;
  if (lower.includes("real trading is disabled")) return REASON_LABELS.real_trading_disabled;
  if (lower.includes("settlement_error")) return REASON_LABELS.settlement_error;
  if (lower.includes("timeout")) return REASON_LABELS.timeout;
  if (lower.includes("exchange reject") || lower.includes("reject")) return "交易所拒绝";
  if (lower.includes("missing completed")) return "缺少已收盘K线";
  if (lower.includes("missing column")) return "缺少必要字段";
  if (lower.includes("not ready")) return "未就绪";
  if (lower.includes("failed")) return "执行失败";
  return "";
}

function isFailureText(value) {
  const text = normalizedReason(value).toLowerCase();
  return text.includes("fail") || text.includes("error") || text.includes("reject");
}
