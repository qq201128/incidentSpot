import { simulationKindLabel } from "./strategyLabels";

/** 模拟盘批量组合 / 因子候选：后端把历史回测胜率写入 probability_up，与模型置信不是同一概念 */
export function eventUsesBacktestWinRate(event) {
  if (simulationKindLabel(event?.strategyKey)) return true;
  const key = String(event?.strategyKey || "").toLowerCase();
  if (key.startsWith("factor_combo_ranker_v1") || key.startsWith("high_winrate_factor_combo_v1")) {
    return true;
  }
  return false;
}

/** 展示用回测胜率（0–100，一位小数）；优先 aiHighWinrateValue */
export function eventBacktestWinRatePercent(event) {
  const gate = Number(event?.aiHighWinrateValue);
  if (Number.isFinite(gate)) return Math.round(gate * 1000) / 10;
  if (!eventUsesBacktestWinRate(event)) return null;
  const p = Number(event?.aiProbabilityUp);
  if (!Number.isFinite(p)) return null;
  return Math.round(Math.max(p, 1 - p) * 1000) / 10;
}

/** 模型方向置信（仅非回测胜率类事件） */
export function eventDirectionalConfidencePercent(event) {
  if (eventUsesBacktestWinRate(event)) return null;
  const p = Number(event?.aiProbabilityUp);
  if (!Number.isFinite(p)) return null;
  if (event?.orderSide === "BUY") return Math.round(p * 1000) / 10;
  if (event?.orderSide === "SELL") return Math.round((1 - p) * 1000) / 10;
  return Math.round(Math.max(p, 1 - p) * 1000) / 10;
}
