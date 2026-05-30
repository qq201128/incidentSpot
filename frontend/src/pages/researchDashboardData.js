export const DURATIONS = Object.freeze(["10m", "30m", "60m", "1d"]);
export const SYMBOLS = Object.freeze(["BTCUSDT", "ETHUSDT"]);
export const EMPTY = "—";
export const TOP_ROW_LIMIT = 18;
export const EVIDENCE_TARGETS = Object.freeze({
  backtestGapWarn: 0.1,
  lossStreakLimit: 5,
  profitFactorMin: 1.05,
  recentWinRateMin: 0.58,
  rollingWinRateMin: 0.5,
  sampleRichMin: 30,
  winRateMin: 0.62,
});

const TOP_REASON_LIMIT = 6;
const STATUS_KEYS = Object.freeze({
  collecting: new Set(["paper_collecting", "backtest_candidate"]),
  failed: new Set(["paper_failed", "invalid_data_leakage"]),
  stable: new Set(["paper_stable"]),
});
const STATUS_PRIORITY = Object.freeze({
  paper_stable: 3,
  paper_collecting: 2,
  backtest_candidate: 1,
  paper_failed: 0,
  invalid_data_leakage: 0,
});

export function settledRows(report) {
  const rows = Array.isArray(report?.allCandidates) ? report.allCandidates : [];
  return rows.map(rowPayload).filter((row) => row.sampleCount > 0).sort(sampleSort);
}

export function researchSummary(report, rows) {
  const sampleCount = rows.reduce((sum, row) => sum + row.sampleCount, 0);
  const weightedWins = rows.reduce((sum, row) => sum + (row.winRate ?? 0) * row.sampleCount, 0);
  const settledCandidateCount = rows.length;
  const totalCandidates = candidateTotal(report);
  return {
    reportLoaded: Boolean(report),
    sampleCount,
    settledCandidateCount,
    settledCoverage: totalCandidates > 0 ? settledCandidateCount / totalCandidates : null,
    weightedWinRate: sampleCount > 0 ? weightedWins / sampleCount : null,
    avgSamplesPerCandidate: settledCandidateCount > 0 ? sampleCount / settledCandidateCount : null,
    sampleRichCandidateCount: rows.filter((row) => row.sampleCount >= EVIDENCE_TARGETS.sampleRichMin).length,
    stableCount: countRowStatus(rows, STATUS_KEYS.stable),
    collectingCount: countRowStatus(rows, STATUS_KEYS.collecting),
    failedCount: countRowStatus(rows, STATUS_KEYS.failed),
    stableSampleCount: sumRowStatus(rows, STATUS_KEYS.stable),
    collectingSampleCount: sumRowStatus(rows, STATUS_KEYS.collecting),
    failedSampleCount: sumRowStatus(rows, STATUS_KEYS.failed),
    modelEvidenceCount: rows.filter((row) => row.type === "model").length,
    backtestGapRiskCount: rows.filter(hasBacktestGapRisk).length,
    recentWeakCount: rows.filter(hasRecentWeakness).length,
    dataIssueCount: issueCount(rows, "dataFreshnessStatus"),
    featureIssueCount: issueCount(rows, "missingFeatureStatus"),
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
    .filter((row) => row.sampleCount === 0)
    .sort(prefilterSort)
    .slice(0, 6);
}

export function topReasons(rows, report) {
  const counts = new Map();
  for (const row of rows) {
    if (row.status !== "paper_failed" && row.status !== "invalid_data_leakage") continue;
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

export function statusClass(status) {
  if (status === "paper_stable") return "is-stable";
  if (status === "paper_failed" || status === "invalid_data_leakage") return "is-failed";
  return "is-collecting";
}

export function statusLabel(status) {
  const labels = {
    backtest_candidate: "预筛",
    paper_stable: "稳定",
    paper_collecting: "观察",
    paper_failed: "失败",
    invalid_data_leakage: "泄漏",
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
  const paperWinRate = numberOrNull(row.paperLiveWinRate ?? metrics.winRate);
  const backtestWinRate = numberOrNull(row.backtestWinRate);
  const type = row.candidateType || "factor";
  const name = candidateName(row, type);
  return {
    rowKey: rowKey(row, name),
    candidateKey: row.candidateKey,
    name,
    modelVersion: row.modelVersion,
    type,
    status: row.paperLiveStatus || row.status,
    reason: row.reason,
    sampleCount: Number(row.paperLiveSampleCount || metrics.sampleCount || 0),
    winRate: paperWinRate,
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
