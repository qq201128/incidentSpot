import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
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
import "./ResearchDashboardPage.responsive.css";

export default function ResearchDashboardPage() {
  const [searchParams] = useSearchParams();
  const [symbol, setSymbol] = useState(searchParams.get("symbol") || "BTCUSDT");
  const [duration, setDuration] = useState(searchParams.get("duration") || "10m");
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
          duration={duration}
          loadError={loadError}
          loading={loading}
          reportLoaded={summary.reportLoaded}
          rows={rows}
          symbol={symbol}
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
    await loadResearchReport({ duration, setState, signal, symbol: normalizedSymbol });
  }, [duration, normalizedSymbol]);

  useEffect(() => {
    const ac = new AbortController();
    void load(ac.signal);
    return () => ac.abort();
  }, [load]);

  const runDailyLoop = useCallback(async () => {
    await runDailyLoopReport({ duration, setState, symbol: normalizedSymbol });
  }, [duration, normalizedSymbol]);

  return { ...state, runDailyLoop };
}

async function loadResearchReport({ duration, setState, signal, symbol }) {
  if (!symbol || !duration) {
    setState({ loadError: "交易对或周期无效", loading: false, report: null, status: "交易对或周期无效" });
    return;
  }
  setState((current) => ({ ...current, loadError: null, loading: true, status: "读取结算样本…" }));
  try {
    const report = await fetchPaperLiveCandidates(symbol, duration, { signal });
    if (signal?.aborted) return;
    publishReport(setState, report, `${reportStatus(report)} · 模型族状态后台合并中…`);
    void mergeModelFamilyRows({ duration, report, setState, signal, symbol });
  } catch (error) {
    if (signal?.aborted) return;
    publishLoadError(setState, "读取失败", error);
  }
}

async function runDailyLoopReport({ duration, setState, symbol }) {
  setState((current) => ({ ...current, loadError: null, loading: true, status: "执行 paper-live 日闭环…" }));
  try {
    const result = await runPaperLiveDailyLoop(symbol, duration);
    const first = Array.isArray(result.results) ? result.results[0] : null;
    const report = first?.candidatePool || null;
    const status = `日闭环 ${result.status || first?.status || "unknown"}`;
    publishReport(setState, report, report ? `${status} · 模型族状态后台合并中…` : `${status} · 未返回 candidatePool`);
    if (!report) return;
    void mergeModelFamilyRows({ duration, report, setState, symbol });
  } catch (error) {
    publishLoadError(setState, "日闭环失败", error);
  }
}

function publishLoadError(setState, prefix, error) {
  const message = errorMessage(error);
  setState((current) => ({ ...current, loadError: message, loading: false, status: `${prefix}：${message}` }));
}

function publishReport(setState, report, status) {
  const loadError = report ? null : "candidatePool_missing";
  setState({ loadError, loading: false, report, status });
}

async function mergeModelFamilyRows({ duration, report, setState, signal, symbol }) {
  try {
    const models = await fetchModelFamilyCandidates(symbol, duration, signal);
    if (signal?.aborted) return;
    const merged = mergeModelFamilyStatusRows(report, models);
    setState({ loadError: null, loading: false, report: merged, status: reportStatus(merged) });
  } catch (error) {
    if (signal?.aborted) return;
    setState((current) => ({
      ...current,
      loading: false,
      status: `${reportStatus(current.report || report)} · 模型族状态合并失败：${errorMessage(error)}`,
    }));
  }
}

async function fetchModelFamilyCandidates(symbol, duration, signal) {
  const results = await Promise.allSettled(
    MODEL_FAMILIES.map((family) => fetchModelFamilyStatus(family, symbol, duration, { signal })),
  );
  return results.map((result, index) => {
    const family = MODEL_FAMILIES[index];
    if (result.status === "fulfilled") {
      return modelStatusCandidate({ duration, family, status: result.value, symbol });
    }
    return modelStatusFailureCandidate({ duration, family, reason: errorMessage(result.reason), symbol });
  });
}

function modelStatusCandidate({ duration, family, status, symbol }) {
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

function modelStatusFailureCandidate({ duration, family, reason, symbol }) {
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
