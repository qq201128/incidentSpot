import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchPaperLiveCandidates,
  runPaperLiveDailyLoop,
} from "../api/factorCombinations";
import { fetchModelFamilyStatus } from "../api/factorLearning";
import { MODEL_FAMILIES } from "../utils/modelFamilies";
import {
  errorMessage,
  mergeModelFamilyStatusRows,
  reportStatus,
  researchSummary,
  settledRows,
} from "./researchDashboardData";
import {
  ResearchHeader,
  SummaryStrip,
} from "./ResearchDashboardSummary";
import {
  ResearchSidePanel,
  SettledSampleMatrix,
} from "./ResearchDashboardEvidence";
import "./ResearchDashboardPage.css";
import "./ResearchDashboardMatrix.css";
import "./ResearchDashboardSidePanel.css";

export default function ResearchDashboardPage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [duration, setDuration] = useState("10m");
  const { loadError, loading, report, runDailyLoop, status } = useResearchReport(symbol, duration);
  const rows = useMemo(() => settledRows(report), [report]);
  const summary = useMemo(() => researchSummary(report, rows), [report, rows]);

  return (
    <main className="research-page layout">
      <ResearchHeader
        duration={duration}
        loading={loading}
        onDurationChange={setDuration}
        onRunDailyLoop={runDailyLoop}
        onSymbolChange={setSymbol}
        status={status}
        symbol={symbol}
      />
      <SummaryStrip summary={summary} />
      <section className="research-main-grid">
        <SettledSampleMatrix
          loadError={loadError}
          loading={loading}
          reportLoaded={summary.reportLoaded}
          rows={rows}
        />
        <ResearchSidePanel report={report} rows={rows} summary={summary} />
      </section>
    </main>
  );
}

function useResearchReport(symbol, duration) {
  const [state, setState] = useState({
    loadError: null,
    loading: false,
    report: null,
    status: "等待读取 paper-live 结算样本",
  });
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);

  const load = useCallback(async (signal) => {
    if (!normalizedSymbol || !duration) {
      setState({ loadError: "交易对或周期无效", loading: false, report: null, status: "交易对或周期无效" });
      return;
    }
    setState((current) => ({ ...current, loadError: null, loading: true, status: "读取结算样本…" }));
    try {
      const [report, models] = await Promise.all([
        fetchPaperLiveCandidates(normalizedSymbol, duration, { signal }),
        fetchModelFamilyCandidates(normalizedSymbol, duration, signal),
      ]);
      if (signal?.aborted) return;
      const merged = mergeModelFamilyStatusRows(report, models);
      setState({ loadError: null, loading: false, report: merged, status: reportStatus(merged) });
    } catch (error) {
      if (signal?.aborted) return;
      const message = errorMessage(error);
      setState((current) => ({ ...current, loadError: message, loading: false, status: `读取失败：${message}` }));
    }
  }, [duration, normalizedSymbol]);

  useEffect(() => {
    const ac = new AbortController();
    void load(ac.signal);
    return () => ac.abort();
  }, [load]);

  const runDailyLoop = useCallback(async () => {
    setState((current) => ({ ...current, loadError: null, loading: true, status: "执行 paper-live 日闭环…" }));
    try {
      const result = await runPaperLiveDailyLoop(normalizedSymbol, duration);
      const first = Array.isArray(result.results) ? result.results[0] : null;
      const models = await fetchModelFamilyCandidates(normalizedSymbol, duration);
      const merged = first?.candidatePool ? mergeModelFamilyStatusRows(first.candidatePool, models) : null;
      setState({
        loadError: null,
        loading: false,
        report: merged,
        status: `日闭环 ${result.status || first?.status || "unknown"}`,
      });
    } catch (error) {
      const message = errorMessage(error);
      setState((current) => ({ ...current, loadError: message, loading: false, status: `日闭环失败：${message}` }));
    }
  }, [duration, normalizedSymbol]);

  return { ...state, runDailyLoop };
}

async function fetchModelFamilyCandidates(symbol, duration, signal) {
  const results = await Promise.allSettled(
    MODEL_FAMILIES.map((family) => fetchModelFamilyStatus(family, symbol, duration, { signal })),
  );
  return results.map((result, index) => {
    const family = MODEL_FAMILIES[index];
    if (result.status === "fulfilled") return modelStatusCandidate(result.value, family, symbol, duration);
    return modelStatusFailureCandidate(family, symbol, duration, errorMessage(result.reason));
  });
}

function modelStatusCandidate(status, family, symbol, duration) {
  const admission = status?.paperLiveAdmission || {};
  return {
    candidateKey: status?.modelVersion || `${family}:${symbol}:${duration}`,
    candidateType: "model",
    strategyKey: status?.strategyKey,
    modelFamily: status?.modelFamily || family,
    modelVersion: status?.modelVersion,
    featureWindow: status?.featureWindow,
    minConfidence: admission.minConfidence ?? status?.selectedConfidenceThreshold,
    validationWinRate: admission.validationWinRate ?? status?.validationWinRate,
    validationSampleCount: status?.sampleCounts?.validation ?? status?.validationGate?.validation?.sampleCount,
    oosWinRate: status?.testWinRate,
    paperLiveWinRate: null,
    paperLiveSampleCount: 0,
    paperLiveStatus: status?.paperLiveStatus || admission.status || "backtest_candidate",
    reason: admission.reason || status?.shadowPredictionBlockedReason || status?.validationFailureReason,
    metrics: { sampleCount: 0 },
  };
}

function modelStatusFailureCandidate(family, symbol, duration, reason) {
  return {
    candidateKey: `${family}:${symbol}:${duration}:status_failed`,
    candidateType: "model",
    modelFamily: family,
    modelVersion: null,
    paperLiveStatus: "model_status_failed",
    reason,
    paperLiveSampleCount: 0,
    paperLiveWinRate: null,
    metrics: { sampleCount: 0 },
  };
}
