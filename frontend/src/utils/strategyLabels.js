/** 与后端 strategy_registry 策略 key 对齐的展示名 */
export const STRATEGY_LABELS = {
  manual: "手动",
  orderbook_notional_40m: "订单簿8M差额",
  orderbook_notional_10m: "订单簿10M差额",
  orderbook_notional_15m: "订单簿15M差额",
  orderbook_notional_40m_mg: "订单簿8M差额·倍投",
  orderbook_notional_10m_mg_5102045: "订单簿10M差额·倍投",
  orderbook_trade_flow_1k: "挂单波动+成交流向",
  orderbook_trade_flow_1k_invert_mg: "挂单波动+成交流向·反向倍投",
  blind_reverse_martingale_v1: "随意首单·反向倍投(10/20/45)",
  three_bar_10m_reverse_martingale_v1: "三连10m反向·不倍投",
  four_bar_10m_reverse_martingale_v1: "四连10m反向·不倍投",
  five_bar_10m_reverse_martingale_v1: "五连10m反向·不倍投",
};

export function strategyLabel(key) {
  return STRATEGY_LABELS[key] || key || STRATEGY_LABELS.manual;
}
