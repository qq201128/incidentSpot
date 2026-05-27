import { factorLabel } from "./factorLearningLabels";
import {
  isModelShadowStrategyKey,
  modelVersionLabel,
  strategyLabel,
} from "./strategyLabels";

/** 规则命中率行展示名：优先因子中文名，否则策略名 */
export function aiHistoryRowLabel(row) {
  const strategyKey = String(row?.strategyKey || "").trim();
  const factorName = String(row?.factorName || "").trim();

  if (isModelShadowStrategyKey(strategyKey)) {
    return strategyLabel(strategyKey);
  }
  if (factorName) {
    const modelLabel = modelVersionLabel(factorName);
    if (modelLabel) return modelLabel;
    return factorLabel(factorName);
  }
  return strategyLabel(strategyKey);
}

/** 规则命中率行英文/字段名：组合或因子原始标识 */
export function aiHistoryRowEnglishName(row) {
  const factorName = String(row?.factorName || "").trim();
  if (factorName) return factorName;
  const strategyKey = String(row?.strategyKey || "").trim();
  return strategyKey || "";
}
