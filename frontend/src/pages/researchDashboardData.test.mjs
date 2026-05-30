import assert from "node:assert/strict";
import {
  prefilterRows,
  reasonLabel,
  researchSummary,
  candidateTypeLabel,
  settledRows,
  topReasons,
} from "./researchDashboardData.js";

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
