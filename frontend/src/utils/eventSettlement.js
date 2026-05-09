/** 已结算事件的预期盈亏（USDT），与持仓卡片「盈亏」口径一致（猜对 +qty*price，猜错 -qty）。 */
export function settledExpectedProfitUsdt(item) {
  const qty = Number(item?.orderQty);
  const price = Number(item?.orderPrice);
  if (!Number.isFinite(qty) || qty <= 0 || !Number.isFinite(price) || price < 0) return null;
  if (item?.status !== "SETTLED") return null;
  const result = item?.result;
  if (result == null) return null;
  const side = item?.orderSide;
  const isCorrect = (side === "BUY" && result === "YES") || (side === "SELL" && result === "NO");
  return isCorrect ? qty * price : -qty;
}

export function formatPnlU(value) {
  if (!Number.isFinite(value)) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}U`;
}
