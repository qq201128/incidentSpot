import { EMPTY } from "./researchDashboardData.js";

export function statusClass(status) {
  if (status === "paper_stable") return "is-stable";
  if (status === "paper_failed" || status === "invalid_data_leakage" || status === "model_status_failed") return "is-failed";
  return "is-collecting";
}

export function statusLabel(status) {
  const labels = {
    backtest_candidate: "预筛",
    paper_stable: "稳定",
    paper_collecting: "观察",
    paper_failed: "失败",
    invalid_data_leakage: "泄漏",
    model_status_failed: "状态失败",
  };
  return labels[status] || status || EMPTY;
}

export function reasonLabel(reason) {
  const labels = {
    consecutive_losses: "连续亏损",
    insufficient_settled_samples: "样本不足",
    invalid_data_leakage: "数据泄漏",
    paper_live_avg_return_below_target: "均收益不足",
    paper_live_profit_factor_below_target: "PF不足",
    paper_live_win_rate_below_target: "胜率不足",
    prediction_failed: "预测失败",
    model_status_untrained: "模型未训练",
    shadow_observation_allowed_without_trade_gate: "影子观察",
    candidate_beats_active_model: "优于当前模型",
    candidate_win_rate_beats_active_model: "胜率优于当前模型",
    active_model_win_rate_missing: "当前模型胜率缺失",
    no_active_shadow_model: "无当前模型",
    candidate_sample_count_below_active: "样本低于当前模型",
    candidate_win_rate_not_improved: "胜率未提升",
    candidate_profit_factor_not_improved: "PF未提升",
    active_model_metrics_missing: "当前模型指标缺失",
    candidate_not_relative_observation_eligible: "未满足相对观察条件",
    no_validation_confidence_threshold_met: "未达交易门槛",
    recent_profit_factor_below_target: "近期PF不足",
    recent_samples_below_min: "近期样本不足",
    recent_win_rate_below_target: "近期胜率不足",
    rolling_windows_below_min: "滚动窗口不足",
    rolling_window_samples_below_min: "滚动样本不足",
    rolling_window_win_rate_below_target: "滚动胜率不足",
    stable_paper_live_target_met: "纸盘达标",
  };
  return labels[reason] || reason || EMPTY;
}
