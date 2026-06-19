export const DURATIONS = Object.freeze(["10m", "30m", "60m", "1d"]);
export const SYMBOLS = Object.freeze(["BTCUSDT", "ETHUSDT"]);
export const EMPTY = "—";
export const TOP_ROW_LIMIT = 18;
export const MODEL_ROW_RESERVE = 14;
export const EVIDENCE_TARGETS = Object.freeze({
  backtestGapWarn: 0.1,
  lossStreakLimit: 4,
  profitFactorMin: 1.05,
  recentWinRateMin: 0.58,
  rollingWinRateMin: 0.5,
  sampleRichMin: 30,
  winRateMin: 0.62,
});

const TOP_REASON_LIMIT = 6;
const STATUS_KEYS = Object.freeze({
  collecting: new Set(["paper_collecting", "backtest_candidate"]),
  failed: new Set(["paper_failed", "invalid_data_leakage", "model_status_failed"]),
  stable: new Set(["paper_stable"]),
});
const STATUS_PRIORITY = Object.freeze({
  paper_stable: 3,
  paper_collecting: 2,
  backtest_candidate: 1,
  paper_failed: 0,
  invalid_data_leakage: 0,
  model_status_failed: 0,
});

const SETTLED_SOURCE_LIMIT = 120;
const SORT_SAMPLE_PRIOR = 50;
const WILSON_Z = 1.96;
const SORT_WEIGHTS = Object.freeze({
  winRateLowerBound: 0.3,
  recentWindowStability: 0.2,
  rollingWindowStability: 0.15,
  returnQuality: 0.15,
  regimeConsistency: 0.1,
  sampleConfidence: 0.1,
});

export function settledRows(report) {
  const rows = [
    ...dashboardCandidateSources(report),
    ...(Array.isArray(report?.modelFamilyStatusRows) ? report.modelFamilyStatusRows : []),
  ];
  return rows.map(rowPayload).filter(visibleResearchRow).sort(sampleSort);
}

function dashboardCandidateSources(report) {
  const liveRows = liveCandidateSources(report);
  const pooled = [
    ...(Array.isArray(report?.stable) ? report.stable : []),
    ...(Array.isArray(report?.collecting) ? report.collecting : []),
    ...(Array.isArray(report?.failed) ? report.failed : []),
    ...(Array.isArray(report?.candidates) ? report.candidates : []),
  ];
  if (pooled.length) {
    return dedupeCandidates([...liveRows, ...pooled]).slice(0, SETTLED_SOURCE_LIMIT);
  }
  const all = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  return dedupeCandidateRows([...liveRows, ...all]).slice(0, SETTLED_SOURCE_LIMIT);
}

function liveCandidateSources(report) {
  const all = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  return all.filter((row) => row?.liveTradingEnabled);
}

function dedupeCandidateRows(rows) {
  const seen = new Set();
  const unique = [];
  for (const row of rows) {
    const key = rowIdentity(row);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(row);
  }
  return unique;
}

function dedupeCandidates(rows) {
  const seen = new Set();
  const unique = [];
  for (const row of rows) {
    const key = row?.candidateKey || row?.strategyKey;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(row);
  }
  return unique;
}

function rowIdentity(row) {
  return [
    row?.candidateKey,
    row?.candidateType,
    row?.factorName,
    row?.modelFamily,
    row?.modelVersion,
    row?.strategyKey,
    row?.firstPredictionCreatedAt,
  ].filter(Boolean).join("::");
}

export function mergeModelFamilyStatusRows(report, models) {
  const statusRows = Array.isArray(models) ? models.map(normalizeModelStatusRow) : [];
  const statusByFamily = modelStatusByFamily(statusRows);
  const allCandidates = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  const mergedCandidates = allCandidates.map((row) => enrichModelCandidate(row, statusByFamily));
  const presentFamilies = new Set(
    mergedCandidates.filter((row) => row?.candidateType === "model").map((row) => row.modelFamily),
  );
  for (const family of report?.candidateModelFamilies || []) {
    presentFamilies.add(family);
  }
  const modelFamilyStatusRows = statusRows.filter((row) => row?.modelFamily && !presentFamilies.has(row.modelFamily));
  return {
    ...report,
    allCandidates: mergedCandidates,
    allCandidateCount: candidateTotal(report) + modelFamilyStatusRows.length,
    modelFamilyStatusRows,
    modelFamilyStatuses: statusRows,
  };
}

export function visibleSettledRows(rows, limit = TOP_ROW_LIMIT) {
  const liveRows = rows.filter((row) => row.liveTradingEnabled).slice(0, limit);
  const reserved = new Set(liveRows.map((row) => row.rowKey));
  const stableRows = rows
    .filter((row) => STATUS_KEYS.stable.has(row.status) && !reserved.has(row.rowKey))
    .slice(0, Math.max(limit - liveRows.length, 0));
  for (const row of stableRows) reserved.add(row.rowKey);
  const slotsAfterStable = Math.max(limit - stableRows.length, 0);
  const modelRows = rows
    .filter((row) => row.type === "model" && !reserved.has(row.rowKey))
    .slice(0, Math.min(MODEL_ROW_RESERVE, Math.max(slotsAfterStable - liveRows.length, 0)));
  for (const row of modelRows) reserved.add(row.rowKey);
  const slotsAfterModels = Math.max(limit - liveRows.length - stableRows.length - modelRows.length, 0);
  const primary = rows.filter((row) => !reserved.has(row.rowKey)).slice(0, slotsAfterModels);
  return [...liveRows, ...stableRows, ...primary, ...modelRows].sort(sampleSort);
}

export function researchSummary(report, rows) {
  const settled = settledEvidenceRows(rows);
  const sampleCount = settled.reduce((sum, row) => sum + row.sampleCount, 0);
  const weightedWins = settled.reduce((sum, row) => sum + (row.winRate ?? 0) * row.sampleCount, 0);
  const settledCandidateCount = settled.length;
  const totalCandidates = candidateTotal(report);
  const localSummary = {
    reportLoaded: Boolean(report),
    sampleCount,
    settledCandidateCount,
    settledCoverage: totalCandidates > 0 ? settledCandidateCount / totalCandidates : null,
    weightedWinRate: sampleCount > 0 ? weightedWins / sampleCount : null,
    avgSamplesPerCandidate: settledCandidateCount > 0 ? sampleCount / settledCandidateCount : null,
    sampleRichCandidateCount: settled.filter((row) => row.sampleCount >= EVIDENCE_TARGETS.sampleRichMin).length,
    stableCount: countRowStatus(settled, STATUS_KEYS.stable),
    collectingCount: countRowStatus(settled, STATUS_KEYS.collecting),
    failedCount: countRowStatus(settled, STATUS_KEYS.failed),
    stableSampleCount: sumRowStatus(settled, STATUS_KEYS.stable),
    collectingSampleCount: sumRowStatus(settled, STATUS_KEYS.collecting),
    failedSampleCount: sumRowStatus(settled, STATUS_KEYS.failed),
    modelEvidenceCount: rows.filter((row) => row.type === "model").length,
    backtestGapRiskCount: settled.filter(hasBacktestGapRisk).length,
    recentWeakCount: settled.filter(hasRecentWeakness).length,
    dataIssueCount: issueCount(settled, "dataFreshnessStatus"),
    featureIssueCount: issueCount(settled, "missingFeatureStatus"),
    unsettledCandidateCount: unsettledCandidateCount(report),
    predictionFailureCount: Array.isArray(report?.predictionFailures) ? report.predictionFailures.length : 0,
    stageFailureCount: stageLogRows(report).filter((row) => row.status === "failed").length,
    statusChangeCount: statusChangeRows(report).length,
  };
  return applyGlobalEvidenceSummary(localSummary, report?.summary || report?.evidenceSummary);
}

export function prefilterRows(report) {
  const rows = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  return rows
    .map(rowPayload)
    .filter((row) => row.sampleCount === 0 && row.type !== "model")
    .sort(prefilterSort)
    .slice(0, 6);
}

export function topReasons(rows, report) {
  const counts = new Map();
  for (const row of rows) {
    if (row.status !== "paper_failed" && row.status !== "invalid_data_leakage" && row.status !== "model_status_failed") continue;
    incrementReason(counts, row.reason || "unknown");
  }
  for (const failure of report?.predictionFailures || []) {
    incrementReason(counts, failure.reason || "prediction_failed");
  }
  return [...counts.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((left, right) => right.count - left.count)
    .slice(0, TOP_REASON_LIMIT);
}

export function stageLogRows(report) {
  return Array.isArray(report?.stageLogs) ? report.stageLogs : [];
}

export function statusChangeRows(report) {
  return Array.isArray(report?.statusChanges) ? report.statusChanges : [];
}

export function formatWindow(window) {
  if (!window || Number(window.sampleCount || 0) <= 0) return EMPTY;
  return `${formatPct(window.winRate)} / ${window.sampleCount}`;
}

export function rollingWindowItems(row) {
  const windows = Array.isArray(row?.stability?.rollingWindows) ? row.stability.rollingWindows : [];
  const threshold = numberOrNull(row?.stability?.thresholds?.rollingWindowWinRateMin);
  return windows
    .filter((window) => Number(window?.sampleCount || 0) > 0)
    .map((window) => {
      const winRate = numberOrNull(window.winRate);
      return {
        key: `${window.sampleCount}:${winRate}:${window.avgReturn ?? ""}`,
        passed: threshold == null || winRate == null || winRate >= threshold,
        text: formatWindow(window),
      };
    });
}

export function formatGap(row) {
  const gap = row.backtestGap;
  if (gap == null) return EMPTY;
  return `${gap >= 0 ? "+" : ""}${formatPct(gap)}`;
}

export function candidateTypeLabel(row) {
  if (row.type === "model") return "模型族";
  if (row.type === "factor_combo" || isComboName(row.name)) return "组合";
  return "因子";
}

export function reportStatus(report) {
  if (!report) return "未返回结算样本";
  const total = candidateTotal(report);
  const settled = settledCandidateTotal(report);
  return `settled ${settled}/${total} · updated ${formatDateTime(report.updatedAt)}`;
}

export function formatPct(value) {
  if (value == null || Number.isNaN(Number(value))) return EMPTY;
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return EMPTY;
  return Number(value).toFixed(digits);
}

export function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || "unknown_error";
}

export function formatDateTime(value) {
  if (!value) return EMPTY;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return EMPTY;
  return date.toLocaleString();
}

function rowPayload(row) {
  const metrics = row.metrics || {};
  const type = row.candidateType || "factor";
  const paperWinRate = numberOrNull(row.paperLiveWinRate ?? (type === "model" ? null : metrics.winRate));
  const validationWinRate = numberOrNull(row.validationWinRate ?? metrics.validationWinRate);
  const winRate = paperWinRate;
  const validationSampleCount = Number(row.validationSampleCount ?? metrics.validationSampleCount ?? 0);
  const backtestWinRate = numberOrNull(row.backtestWinRate);
  const name = candidateName(row, type);
  return {
    rowKey: rowKey(row, name),
    candidateKey: row.candidateKey,
    strategyKey: row.strategyKey,
    name,
    modelVersion: row.modelVersion,
    type,
    status: row.paperLiveStatus || row.status,
    reason: row.reason,
    autoTradeEnabled: Boolean(row.autoTradeEnabled),
    liveTradingEnabled: Boolean(row.liveTradingEnabled),
    liveTradingUpdatedAt: row.liveTradingUpdatedAt,
    sampleCount: Number(row.paperLiveSampleCount || (type === "model" ? 0 : metrics.sampleCount) || 0),
    validationSampleCount,
    validationWinRate,
    winRate,
    backtestWinRate,
    backtestGap: gapValue(backtestWinRate, paperWinRate),
    oosWinRate: numberOrNull(row.oosWinRate),
    robustScore: numberOrNull(row.robustScore),
    scoreBreakdown: row.scoreBreakdown,
    profitFactor: numberOrNull(metrics.profitFactor),
    avgReturn: numberOrNull(metrics.avgReturn),
    maxConsecutiveLosses: Number(metrics.maxConsecutiveLosses || 0),
    dataFreshnessStatus: row.dataFreshnessStatus,
    missingFeatureStatus: row.missingFeatureStatus,
    regimeValidation: row.regimeValidation,
    stability: metrics.paperStability || {},
    windows: metrics.paperLiveWindows || {},
  };
}

function visibleResearchRow(row) {
  return row.sampleCount > 0;
}

function modelStatusByFamily(rows) {
  return new Map(rows.filter((row) => row?.modelFamily).map((row) => [row.modelFamily, row]));
}

function normalizeModelStatusRow(row) {
  const family = row?.modelFamily;
  const symbol = row?.symbol || "";
  const duration = row?.duration || "";
  return {
    ...row,
    candidateKey: row?.candidateKey || row?.modelVersion || `${family}:${symbol}:${duration}`,
    candidateType: "model",
    paperLiveStatus: normalizedPaperLiveStatus(row),
    paperLiveSampleCount: Number(row?.paperLiveSampleCount || 0),
    paperLiveWinRate: numberOrNull(row?.paperLiveWinRate),
    validationSampleCount: row?.validationSampleCount ?? row?.sampleCounts?.validation ?? row?.validationGate?.validation?.sampleCount,
    metrics: { sampleCount: 0, ...(row?.metrics || {}) },
  };
}

function normalizedPaperLiveStatus(row) {
  const status = row?.paperLiveStatus || row?.paperLiveAdmission?.status;
  if (status) return status;
  const fallback = String(row?.status || "");
  if (fallback.startsWith("paper_") || fallback === "backtest_candidate" || fallback === "model_status_failed") {
    return fallback;
  }
  return "backtest_candidate";
}

function enrichModelCandidate(row, statusByFamily) {
  if (row?.candidateType !== "model" || !row.modelFamily) return row;
  const status = statusByFamily.get(row.modelFamily);
  if (!status) return row;
  return {
    ...status,
    ...row,
    metrics: { ...(status.metrics || {}), ...(row.metrics || {}) },
  };
}

function settledEvidenceRows(rows) {
  return rows.filter((row) => row.sampleCount > 0);
}

function candidateName(row, type) {
  if (type === "factor_combo") {
    return row.factorName || row.modelVersion || row.candidateKey || row.strategyKey || EMPTY;
  }
  if (type === "model" && row.modelFamily && row.modelVersion) return `${row.modelFamily} · ${row.modelVersion}`;
  if (type === "model") return row.modelFamily || row.modelVersion || row.candidateKey || row.strategyKey || EMPTY;
  return row.factorName || row.candidateKey || row.strategyKey || row.modelFamily || EMPTY;
}

function isComboName(name) {
  const raw = String(name || "");
  return raw.startsWith("combo__") || raw.startsWith("goal_combo__");
}

function rowKey(row, name) {
  return [
    row.candidateKey,
    row.candidateType,
    row.factorName,
    row.modelFamily,
    row.modelVersion,
    row.strategyKey,
    row.firstPredictionCreatedAt,
    name,
  ].filter(Boolean).join("::");
}

function sampleSort(left, right) {
  return statusPriority(right) - statusPriority(left)
    || robustSortScore(right) - robustSortScore(left)
    || effectiveSampleCount(right) - effectiveSampleCount(left)
    || (right.winRate ?? 0) - (left.winRate ?? 0)
    || (right.profitFactor ?? 0) - (left.profitFactor ?? 0);
}

function statusPriority(row) {
  return STATUS_PRIORITY[row.status] ?? 0;
}

function robustSortScore(row) {
  if (row.robustScore != null) return row.robustScore;
  const score = SORT_WEIGHTS.winRateLowerBound * winRateLowerBound(row)
    + SORT_WEIGHTS.recentWindowStability * recentWindowStabilityScore(row)
    + SORT_WEIGHTS.rollingWindowStability * rollingWindowStabilityScore(row)
    + SORT_WEIGHTS.returnQuality * returnQualityScore(row)
    + SORT_WEIGHTS.regimeConsistency * regimeConsistencyScore(row)
    + SORT_WEIGHTS.sampleConfidence * sampleConfidence(row);
  return score - backtestGapPenalty(row) - lossStreakPenalty(row);
}

function winRateLowerBound(row) {
  return wilsonLowerBound(row.winRate, effectiveSampleCount(row));
}

function recentWindowStabilityScore(row) {
  const windows = [row.windows?.recent30, row.windows?.recent60, row.windows?.recent100]
    .map(windowScorePayload)
    .filter(Boolean);
  if (!windows.length) return winRateLowerBound(row);
  return stableWinRateScore(windows.map((window) => window.winRate), row.winRate);
}

function rollingWindowStabilityScore(row) {
  const windows = Array.isArray(row?.stability?.rollingWindows) ? row.stability.rollingWindows : [];
  const winRates = windows.map(windowScorePayload).filter(Boolean).map((window) => window.winRate);
  if (!winRates.length) return recentWindowStabilityScore(row);
  return stableWinRateScore(winRates, row.winRate);
}

function returnQualityScore(row) {
  const pfScore = row.profitFactor == null ? 0 : clamp((row.profitFactor - 1) / 1);
  const avgReturnScore = row.avgReturn == null ? 0 : clamp(0.5 + row.avgReturn / 0.01);
  return 0.7 * pfScore + 0.3 * avgReturnScore;
}

function regimeConsistencyScore(row) {
  const buckets = regimeBuckets(row.regimeValidation);
  if (!buckets.length) return 0.5;
  const totalSamples = buckets.reduce((sum, bucket) => sum + bucket.sampleCount, 0);
  if (totalSamples <= 0) return 0.5;
  const lowerBounds = buckets.map((bucket) => ({
    sampleCount: bucket.sampleCount,
    score: wilsonLowerBound(bucket.winRate, bucket.sampleCount),
  }));
  const weightedLower = lowerBounds.reduce(
    (sum, bucket) => sum + bucket.score * bucket.sampleCount,
    0,
  ) / totalSamples;
  const spread = Math.max(...buckets.map((bucket) => bucket.winRate))
    - Math.min(...buckets.map((bucket) => bucket.winRate));
  const coverageConfidence = Math.sqrt(buckets.length / (buckets.length + 3));
  return clamp(weightedLower - spread * 0.35 + coverageConfidence * 0.1);
}

function sampleConfidence(row) {
  const sampleCount = effectiveSampleCount(row);
  if (sampleCount <= 0) return 0;
  return Math.sqrt(sampleCount / (sampleCount + SORT_SAMPLE_PRIOR));
}

function backtestGapPenalty(row) {
  const gap = Number(row.backtestGap);
  if (!Number.isFinite(gap) || gap <= 0) return 0;
  return clamp(gap, 0, 0.25);
}

function lossStreakPenalty(row) {
  const losses = Number(row.maxConsecutiveLosses || 0);
  if (!Number.isFinite(losses)) return 0;
  return Math.max(losses - 2, 0) * 0.03;
}

function effectiveSampleCount(row) {
  return Math.max(Number(row.sampleCount || 0), Number(row.validationSampleCount || 0));
}

function windowScorePayload(window) {
  const sampleCount = Number(window?.sampleCount || 0);
  const winRate = numberOrNull(window?.winRate);
  if (sampleCount <= 0 || winRate == null) return null;
  return { sampleCount, winRate };
}

function stableWinRateScore(winRates, fallback) {
  const values = winRates.filter((value) => Number.isFinite(value));
  if (!values.length) return numberOrNull(fallback) ?? 0;
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const reference = numberOrNull(fallback) ?? average;
  const averageDeviation = values.reduce((sum, value) => sum + Math.abs(value - reference), 0) / values.length;
  return clamp(average - averageDeviation);
}

function regimeBuckets(regimeValidation) {
  if (!regimeValidation || typeof regimeValidation !== "object") return [];
  return Object.values(regimeValidation).map((payload) => {
    const winRate = numberOrNull(payload?.winRate ?? payload?.accuracy);
    const sampleCount = Number(payload?.sampleCount ?? payload?.n ?? 0);
    if (winRate == null || sampleCount <= 0) return null;
    return { sampleCount, winRate };
  }).filter(Boolean);
}

function wilsonLowerBound(winRate, sampleCount) {
  const n = Number(sampleCount || 0);
  const p = numberOrNull(winRate);
  if (!Number.isFinite(n) || n <= 0 || p == null) return 0;
  const boundedP = clamp(p);
  const z2 = WILSON_Z * WILSON_Z;
  const denominator = 1 + z2 / n;
  const centre = boundedP + z2 / (2 * n);
  const margin = WILSON_Z * Math.sqrt((boundedP * (1 - boundedP) + z2 / (4 * n)) / n);
  return clamp((centre - margin) / denominator);
}

function clamp(value, min = 0, max = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return min;
  return Math.min(Math.max(number, min), max);
}

function countRowStatus(rows, statuses) {
  return rows.filter((row) => statuses.has(row.status)).length;
}

function sumRowStatus(rows, statuses) {
  return rows.reduce((sum, row) => sum + (statuses.has(row.status) ? row.sampleCount : 0), 0);
}

function issueCount(rows, key) {
  return rows.filter((row) => {
    const value = row[key];
    return value && !["fresh", "complete", "ok"].includes(String(value).toLowerCase());
  }).length;
}

function prefilterSort(left, right) {
  return (right.oosWinRate ?? 0) - (left.oosWinRate ?? 0)
    || (right.backtestWinRate ?? 0) - (left.backtestWinRate ?? 0);
}

function unsettledCandidateCount(report) {
  return Math.max(candidateTotal(report) - settledCandidateTotal(report), 0);
}

function incrementReason(counts, reason) {
  counts.set(reason, (counts.get(reason) || 0) + 1);
}

function numberOrNull(value) {
  if (value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function candidateTotal(report) {
  return Number(report?.allCandidateCount ?? report?.allCandidates?.length ?? 0);
}

function settledCandidateTotal(report) {
  const settledCount = Number(report?.settledCandidateCount);
  if (Number.isFinite(settledCount)) return settledCount;
  if (!Array.isArray(report?.allCandidates)) return 0;
  return report.allCandidates.filter((row) => Number(row.paperLiveSampleCount || row.metrics?.sampleCount || 0) > 0).length;
}

function gapValue(backtestWinRate, winRate) {
  return backtestWinRate == null || winRate == null ? null : backtestWinRate - winRate;
}

function hasBacktestGapRisk(row) {
  return row.backtestGap != null && row.backtestGap >= EVIDENCE_TARGETS.backtestGapWarn;
}

function hasRecentWeakness(row) {
  const recent = row.windows?.recent30 || row.stability?.recent;
  return recent?.winRate != null && Number(recent.winRate) < EVIDENCE_TARGETS.recentWinRateMin;
}

function applyGlobalEvidenceSummary(localSummary, evidenceSummary) {
  if (!evidenceSummary || typeof evidenceSummary !== "object") return localSummary;
  return {
    ...localSummary,
    sampleCount: numberOrFallback(evidenceSummary.sampleCount, localSummary.sampleCount),
    settledCandidateCount: numberOrFallback(
      evidenceSummary.settledCandidateCount,
      localSummary.settledCandidateCount,
    ),
    unsettledCandidateCount: numberOrFallback(
      evidenceSummary.unsettledCandidateCount,
      localSummary.unsettledCandidateCount,
    ),
    settledCoverage: numberOrNull(evidenceSummary.settledCoverage) ?? localSummary.settledCoverage,
    weightedWinRate: numberOrNull(evidenceSummary.weightedWinRate) ?? localSummary.weightedWinRate,
    avgSamplesPerCandidate: numberOrNull(evidenceSummary.avgSamplesPerCandidate) ?? localSummary.avgSamplesPerCandidate,
    sampleRichCandidateCount: numberOrFallback(
      evidenceSummary.sampleRichCandidateCount,
      localSummary.sampleRichCandidateCount,
    ),
    stableCount: numberOrFallback(evidenceSummary.stableCount, localSummary.stableCount),
    collectingCount: numberOrFallback(evidenceSummary.collectingCount, localSummary.collectingCount),
    failedCount: numberOrFallback(evidenceSummary.failedCount, localSummary.failedCount),
    stableSampleCount: numberOrFallback(evidenceSummary.stableSampleCount, localSummary.stableSampleCount),
    collectingSampleCount: numberOrFallback(
      evidenceSummary.collectingSampleCount,
      localSummary.collectingSampleCount,
    ),
    failedSampleCount: numberOrFallback(evidenceSummary.failedSampleCount, localSummary.failedSampleCount),
    modelEvidenceCount: numberOrFallback(evidenceSummary.modelEvidenceCount, localSummary.modelEvidenceCount),
    backtestGapRiskCount: numberOrFallback(evidenceSummary.backtestGapRiskCount, localSummary.backtestGapRiskCount),
    recentWeakCount: numberOrFallback(evidenceSummary.recentWeakCount, localSummary.recentWeakCount),
    dataIssueCount: numberOrFallback(evidenceSummary.dataIssueCount, localSummary.dataIssueCount),
    featureIssueCount: numberOrFallback(evidenceSummary.featureIssueCount, localSummary.featureIssueCount),
  };
}

function numberOrFallback(value, fallback) {
  return numberOrNull(value) ?? fallback;
}
