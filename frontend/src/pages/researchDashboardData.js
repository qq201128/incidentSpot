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

export function settledRows(report) {
  const rows = [
    ...dashboardCandidateSources(report),
    ...(Array.isArray(report?.modelFamilyStatusRows) ? report.modelFamilyStatusRows : []),
  ];
  return rows.map(rowPayload).filter(visibleResearchRow).sort(sampleSort);
}

function dashboardCandidateSources(report) {
  const pooled = [
    ...(Array.isArray(report?.stable) ? report.stable : []),
    ...(Array.isArray(report?.collecting) ? report.collecting : []),
    ...(Array.isArray(report?.failed) ? report.failed : []),
    ...(Array.isArray(report?.candidates) ? report.candidates : []),
  ];
  if (pooled.length) {
    return dedupeCandidates(pooled).slice(0, SETTLED_SOURCE_LIMIT);
  }
  const all = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  return all.slice(0, SETTLED_SOURCE_LIMIT);
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

export function mergeModelFamilyStatusRows(report, models) {
  const statusRows = Array.isArray(models) ? models.map(normalizeModelStatusRow) : [];
  const statusByFamily = modelStatusByFamily(statusRows);
  const allCandidates = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  const mergedCandidates = allCandidates.map((row) => enrichModelCandidate(row, statusByFamily));
  const presentFamilies = new Set(
    mergedCandidates.filter((row) => row?.candidateType === "model").map((row) => row.modelFamily),
  );
  const modelFamilyStatusRows = statusRows.filter((row) => row?.modelFamily && !presentFamilies.has(row.modelFamily));
  return {
    ...report,
    allCandidates: mergedCandidates,
    allCandidateCount: mergedCandidates.length + modelFamilyStatusRows.length,
    modelFamilyStatusRows,
    modelFamilyStatuses: statusRows,
  };
}

export function visibleSettledRows(rows, limit = TOP_ROW_LIMIT) {
  const stableRows = rows.filter((row) => STATUS_KEYS.stable.has(row.status)).slice(0, limit);
  const reserved = new Set(stableRows.map((row) => row.rowKey));
  const slotsAfterStable = Math.max(limit - stableRows.length, 0);
  const modelRows = rows
    .filter((row) => row.type === "model" && !reserved.has(row.rowKey))
    .slice(0, Math.min(MODEL_ROW_RESERVE, slotsAfterStable));
  for (const row of modelRows) reserved.add(row.rowKey);
  const slotsAfterModels = Math.max(slotsAfterStable - modelRows.length, 0);
  const primary = rows.filter((row) => !reserved.has(row.rowKey)).slice(0, slotsAfterModels);
  return [...stableRows, ...primary, ...modelRows].sort(sampleSort);
}

export function researchSummary(report, rows) {
  const settled = settledEvidenceRows(rows);
  const sampleCount = settled.reduce((sum, row) => sum + row.sampleCount, 0);
  const weightedWins = settled.reduce((sum, row) => sum + (row.winRate ?? 0) * row.sampleCount, 0);
  const settledCandidateCount = settled.length;
  const totalCandidates = candidateTotal(report);
  return {
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
  const winRate = paperWinRate ?? (type === "model" ? validationWinRate : null);
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
    sampleCount: Number(row.paperLiveSampleCount || (type === "model" ? 0 : metrics.sampleCount) || 0),
    validationSampleCount,
    validationWinRate,
    winRate,
    backtestWinRate,
    backtestGap: gapValue(backtestWinRate, paperWinRate),
    oosWinRate: numberOrNull(row.oosWinRate),
    profitFactor: numberOrNull(metrics.profitFactor),
    avgReturn: numberOrNull(metrics.avgReturn),
    maxConsecutiveLosses: Number(metrics.maxConsecutiveLosses || 0),
    dataFreshnessStatus: row.dataFreshnessStatus,
    missingFeatureStatus: row.missingFeatureStatus,
    stability: metrics.paperStability || {},
    windows: metrics.paperLiveWindows || {},
  };
}

function visibleResearchRow(row) {
  return row.sampleCount > 0 || row.type === "model";
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
  return row.modelFamily || row.factorName || row.candidateKey || row.strategyKey || EMPTY;
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
    || right.sampleCount - left.sampleCount
    || (right.winRate ?? 0) - (left.winRate ?? 0)
    || (right.profitFactor ?? 0) - (left.profitFactor ?? 0);
}

function statusPriority(row) {
  return STATUS_PRIORITY[row.status] ?? 0;
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
