import assert from "node:assert/strict";
import {
  eventFailureReasonLabel,
  failureReasonLabel,
} from "./failureReasonLabels.js";

assert.equal(failureReasonLabel("win_rate_below_min"), "胜率不足");
assert.equal(failureReasonLabel("cache_unavailable:factor_ranking_cache"), "缓存不可用：排名缓存");
assert.equal(
  failureReasonLabel("factor candidate signal missing column: ret_good"),
  "候选因子信号缺少字段",
);
assert.equal(
  failureReasonLabel("factor candidate signal missing completed 10m source row"),
  "候选因子信号缺少已收盘K线",
);
assert.equal(failureReasonLabel("exchange reject: min notional"), "交易所拒绝");

assert.equal(
  eventFailureReasonLabel({ externalResponse: "exchange reject: min notional", status: "OPEN" }),
  "交易所拒绝",
);
assert.equal(
  eventFailureReasonLabel({ externalResponse: '{"simulation":true}', orderStatus: "FAILED" }),
  "订单状态失败",
);
assert.equal(
  eventFailureReasonLabel({ settlementSource: "settlement_error: timeout", status: "OPEN" }),
  "结算失败",
);
assert.equal(eventFailureReasonLabel({ status: "FAILED" }), "事件状态失败");
