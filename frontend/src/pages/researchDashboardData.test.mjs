import assert from "node:assert/strict";
import {
  mergeModelFamilyStatusRows,
  prefilterRows,
  researchSummary,
  candidateTypeLabel,
  settledRows,
  topReasons,
  visibleSettledRows,
} from "./researchDashboardData.js";
import { reasonLabel } from "./researchDashboardLabels.js";

const report = {
  allCandidateCount: 4,
  allCandidates: [
    {
      candidateKey: "factor_beta",
      candidateType: "factor",
      factorName: "beta",
      paperLiveStatus: "paper_failed",
      paperLiveWinRate: 0.52,
      paperLiveSampleCount: 34,
      backtestWinRate: 0.75,
      reason: "paper_live_win_rate_below_target",
      metrics: {
        sampleCount: 34,
        winRate: 0.52,
        profitFactor: 0.92,
        avgReturn: -0.004,
        maxConsecutiveLosses: 6,
        paperLiveWindows: { recent30: { sampleCount: 30, winRate: 0.5 } },
      },
      dataFreshnessStatus: "fresh",
      missingFeatureStatus: "complete",
    },
    {
      candidateKey: "lstm_v4",
      candidateType: "model",
      modelFamily: "lstm",
      modelVersion: "lstm_v4",
      paperLiveStatus: "paper_stable",
      paperLiveWinRate: 0.68,
      paperLiveSampleCount: 31,
      backtestWinRate: 0.7,
      reason: "stable_paper_live_target_met",
      metrics: {
        sampleCount: 31,
        winRate: 0.68,
        profitFactor: 1.3,
        avgReturn: 0.003,
        maxConsecutiveLosses: 1,
        paperLiveWindows: { recent30: { sampleCount: 30, winRate: 0.67 } },
      },
    },
    {
      candidateKey: "factor_alpha",
      candidateType: "factor",
      factorName: "alpha",
      paperLiveStatus: "paper_collecting",
      paperLiveWinRate: null,
      paperLiveSampleCount: 0,
      backtestWinRate: 0.82,
      oosWinRate: 0.65,
      metrics: { sampleCount: 0 },
    },
    {
      candidateKey: "factor_alpha",
      candidateType: "factor_combo",
      factorName: "combo__alpha__beta",
      paperLiveStatus: "paper_collecting",
      paperLiveWinRate: 0.6,
      paperLiveSampleCount: 3,
      backtestWinRate: 0.61,
      reason: "insufficient_settled_samples",
      metrics: { sampleCount: 3, winRate: 0.6 },
    },
  ],
  predictionFailures: [{ reason: "prediction_failed" }],
};

const rows = settledRows(report);
assert.equal(rows.length, 3);
assert.ok(rows.every((row) => row.sampleCount > 0));
assert.equal(rows[0].name, "lstm · lstm_v4");
assert.equal(rows[0].status, "paper_stable");
assert.equal(rows[1].name, "combo__alpha__beta");
assert.equal(candidateTypeLabel(rows[1]), "组合");
assert.notEqual(rows[1].rowKey, "factor_alpha");
assert.ok(Math.abs(rows[2].backtestGap - 0.23) < 0.000001);

const prefilter = prefilterRows(report);
assert.equal(prefilter.length, 1);
assert.equal(prefilter[0].name, "alpha");
assert.equal(prefilter[0].sampleCount, 0);

const summary = researchSummary(report, rows);
assert.equal(summary.reportLoaded, true);
assert.equal(summary.sampleCount, 68);
assert.equal(summary.settledCandidateCount, 3);
assert.equal(summary.unsettledCandidateCount, 1);
assert.equal(summary.stableCount, 1);
assert.equal(summary.failedCount, 1);
assert.equal(summary.modelEvidenceCount, 1);
assert.equal(summary.backtestGapRiskCount, 1);
assert.equal(summary.recentWeakCount, 1);
assert.equal(summary.sampleRichCandidateCount, 2);

assert.equal(topReasons(rows, report)[0].reason, "paper_live_win_rate_below_target");
assert.equal(reasonLabel("paper_live_win_rate_below_target"), "胜率不足");

assert.equal(researchSummary(null, []).reportLoaded, false);

const observationReport = {
  allCandidateCount: 1,
  allCandidates: [],
  modelFamilyStatusRows: [
    {
      candidateKey: "knn:BTCUSDT:10m",
      candidateType: "model",
      modelFamily: "knn",
      modelVersion: "knn_v1",
      paperLiveStatus: "paper_collecting",
      paperLiveSampleCount: 0,
      validationWinRate: 0.61,
      validationSampleCount: 60,
      reason: "shadow_observation_allowed_without_trade_gate",
      metrics: { sampleCount: 0 },
    },
  ],
};
const observationRows = settledRows(observationReport);
assert.equal(observationRows.length, 1);
assert.equal(observationRows[0].type, "model");
assert.equal(observationRows[0].sampleCount, 0);
assert.equal(observationRows[0].validationSampleCount, 60);
assert.equal(observationRows[0].winRate, 0.61);
const observationSummary = researchSummary(observationReport, observationRows);
assert.equal(observationSummary.sampleCount, 0);
assert.equal(observationSummary.settledCandidateCount, 0);
assert.equal(observationSummary.modelEvidenceCount, 1);
assert.equal(observationSummary.weightedWinRate, null);
assert.equal(reasonLabel("shadow_observation_allowed_without_trade_gate"), "影子观察");
assert.equal(reasonLabel("candidate_win_rate_beats_active_model"), "胜率优于当前模型");

const mergedModelReport = mergeModelFamilyStatusRows(
  {
    allCandidateCount: 1,
    allCandidates: [
      {
        candidateKey: "xgboost_v2",
        candidateType: "model",
        modelFamily: "xgboost",
        modelVersion: "xgboost_v2",
        paperLiveStatus: "paper_failed",
        paperLiveWinRate: 0.5,
        paperLiveSampleCount: 30,
        reason: "paper_live_win_rate_below_target",
        metrics: { sampleCount: 30, winRate: 0.5 },
      },
    ],
  },
  [
    {
      candidateKey: "xgboost_v2",
      candidateType: "model",
      modelFamily: "xgboost",
      modelVersion: "xgboost_v2",
      validationWinRate: 0.64,
      validationSampleCount: 80,
      paperLiveStatus: "paper_collecting",
      metrics: { sampleCount: 0 },
    },
    {
      candidateKey: "knn_v1",
      candidateType: "model",
      modelFamily: "knn",
      modelVersion: "knn_v1",
      validationWinRate: 0.61,
      validationSampleCount: 60,
      paperLiveStatus: "paper_collecting",
      metrics: { sampleCount: 0 },
    },
  ],
);
const mergedModelRows = settledRows(mergedModelReport);
const mergedXgboost = mergedModelRows.find((row) => row.name === "xgboost · xgboost_v2");
assert.equal(mergedModelReport.modelFamilyStatusRows.length, 1);
assert.equal(mergedModelReport.modelFamilyStatusRows[0].modelFamily, "knn");
assert.equal(mergedModelReport.allCandidateCount, 2);
assert.equal(mergedXgboost.sampleCount, 30);
assert.equal(mergedXgboost.validationSampleCount, 80);
assert.equal(mergedXgboost.winRate, 0.5);

const backendModelStatusReport = mergeModelFamilyStatusRows(
  { allCandidates: [] },
  [
    {
      modelFamily: "bayesian",
      status: "passed",
      validationWinRate: 0.57,
      sampleCounts: { validation: 72 },
      paperLiveAdmission: {
        status: "paper_collecting",
        reason: "candidate_win_rate_beats_active_model",
        validationWinRate: 0.57,
      },
    },
  ],
);
const backendModelRows = settledRows(backendModelStatusReport);
assert.equal(backendModelStatusReport.allCandidateCount, 1);
assert.equal(backendModelRows.length, 1);
assert.equal(backendModelRows[0].type, "model");
assert.equal(backendModelRows[0].status, "paper_collecting");
assert.equal(backendModelRows[0].name, "bayesian");
assert.equal(backendModelRows[0].validationSampleCount, 72);
assert.equal(backendModelRows[0].winRate, 0.57);

const manyFactorRows = Array.from({ length: 20 }, (_, index) => ({
  rowKey: `factor-${index}`,
  type: "factor",
  sampleCount: 100 - index,
  winRate: 0.6,
  status: "paper_collecting",
}));
const hiddenModel = {
  rowKey: "model-hidden",
  type: "model",
  sampleCount: 1,
  winRate: 0.4,
  status: "paper_failed",
};
const visible = visibleSettledRows([...manyFactorRows, hiddenModel], 18);
assert.equal(visible.length, 18);
assert.ok(visible.some((row) => row.rowKey === "model-hidden"));

const allModelsVisible = visibleSettledRows([
  ...manyFactorRows,
  ...Array.from({ length: 14 }, (_, index) => ({
    rowKey: `model-${index}`,
    type: "model",
    sampleCount: 0,
    winRate: 0.5,
    status: "paper_collecting",
  })),
], 18);
assert.equal(allModelsVisible.filter((row) => row.type === "model").length, 14);
