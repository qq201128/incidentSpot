import { factorLabel } from "./factorLearningLabels";
import { strategyLabel } from "./strategyLabels";

/** 规则命中率行展示名：优先因子中文名，否则策略名 */
export function aiHistoryRowLabel(row) {
  const factorName = String(row?.factorName || "").trim();
  if (factorName) return factorLabel(factorName);
  return strategyLabel(row?.strategyKey);
}
