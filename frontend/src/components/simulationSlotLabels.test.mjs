import assert from "node:assert/strict";
import {
  simulationAmount,
  simulationCandidateTypeLabel,
  simulationLatestEventLabel,
  simulationLatestFailureLabel,
  simulationLatestPnlLabel,
  simulationPnlClass,
  simulationRejectionReasonLabel,
  simulationSlotState,
  simulationSourceLabel,
  simulationThresholdLabel,
} from "./simulationSlotLabels.js";

assert.deepEqual(simulationSlotState({ gateStatus: "enabled" }), {
  label: "已启用",
  className: "enabled",
});
assert.deepEqual(simulationSlotState({ gateStatus: "rejected" }), {
  label: "已拒绝",
  className: "rejected",
});
assert.deepEqual(simulationSlotState({ gateStatus: "idle" }), {
  label: "未启用",
  className: "idle",
});

assert.equal(simulationCandidateTypeLabel("single_factor"), "单因子");
assert.equal(simulationCandidateTypeLabel("factor_combo"), "多因子");
assert.equal(simulationSourceLabel("factor_combo_ranking_cache"), "组合缓存");
assert.equal(simulationRejectionReasonLabel("win_rate_below_min"), "胜率不足");
assert.equal(
  simulationRejectionReasonLabel("cache_unavailable:factor_ranking_cache"),
  "缓存不可用：排名缓存",
);
assert.equal(
  simulationLatestFailureLabel({ reason: "factor candidate signal missing column: ret_good" }),
  "预测失败：候选因子信号缺少字段",
);
assert.equal(
  simulationThresholdLabel({ minWinRate: 0.62, minProfitFactor: 1.2, minTotalPeriods: 30 }),
  "阈值：胜率 62% / PF 1.2 / 样本 30",
);

const event = { id: 7, orderId: 9, status: "SETTLED", externalStatus: "SIMULATED", settlementPnl: 1.5 };
assert.equal(simulationLatestEventLabel(event), "最近事件：EVT-000007 / ORD-9 · SETTLED · SIM");
assert.equal(simulationLatestPnlLabel(event), "最近PnL：+1.50");
assert.equal(simulationPnlClass(event), "value-up");
assert.equal(simulationLatestPnlLabel({ settlementPnl: -0.25 }), "最近PnL：-0.25");
assert.equal(simulationPnlClass({ settlementPnl: -0.25 }), "value-down");
assert.equal(simulationAmount(5), "5");
assert.equal(simulationAmount(0.125), "0.125");
