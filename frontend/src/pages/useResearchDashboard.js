import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchPaperLiveCandidates,
  fetchPaperLiveLiveSummary,
  runPaperLiveDailyLoop,
  setPaperLiveCandidateLiveTrading,
} from "../api/factorCombinations";
import { fetchResearchModelBundle } from "../api/researchDashboardClient";
import {
  errorMessage,
  mergeModelFamilyStatusRows,
  reportStatus,
} from "./researchDashboardData";
import { peekResearchDashboardCache, storeResearchDashboardCache } from "./researchDashboardCache";

export function useResearchDashboard(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [state, setState] = useState(() => initialState(normalizedSymbol, duration));
  const [liveOverview, setLiveOverview] = useState(null);
  const [liveOverviewError, setLiveOverviewError] = useState(null);

  const load = useCallback(
    async (signal) => {
      await loadResearchReport({ duration, normalizedSymbol, setState, signal });
    },
    [duration, normalizedSymbol],
  );

  const refreshLiveOverview = useCallback(async (signal) => {
    try {
      const overview = await fetchPaperLiveLiveSummary({ signal });
      if (signal?.aborted) return;
      setLiveOverview(overview);
      setLiveOverviewError(null);
    } catch (error) {
      if (signal?.aborted) return;
      setLiveOverviewError(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    const cached = peekResearchDashboardCache(normalizedSymbol, duration);
    if (cached?.report) {
      setState({
        loadError: null,
        liveToggleKey: null,
        loading: false,
        mergingModels: true,
        report: cached.report,
        status: `${reportStatus(cached.report)} · 缓存展示中，后台刷新…`,
      });
    } else {
      setState(initialState(normalizedSymbol, duration));
    }
    const ac = new AbortController();
    void load(ac.signal);
    void refreshLiveOverview(ac.signal);
    return () => ac.abort();
  }, [duration, load, normalizedSymbol, refreshLiveOverview]);

  const runDailyLoop = useCallback(async () => {
    await runDailyLoopReport({ duration, normalizedSymbol, setState });
    void refreshLiveOverview();
  }, [duration, normalizedSymbol, refreshLiveOverview]);

  const toggleCandidateLiveTrading = useCallback(
    async (row, liveTradingEnabled) => {
      await toggleLiveTradingReport({
        duration,
        liveTradingEnabled,
        normalizedSymbol,
        row,
        setState,
      });
      void refreshLiveOverview();
    },
    [duration, normalizedSymbol, refreshLiveOverview],
  );

  return { ...state, liveOverview, liveOverviewError, runDailyLoop, toggleCandidateLiveTrading };
}

function initialState(symbol, duration) {
  const cached = peekResearchDashboardCache(symbol, duration);
  if (cached?.report) {
    return {
      loadError: null,
      loading: false,
      mergingModels: false,
      report: cached.report,
      status: reportStatus(cached.report),
      liveToggleKey: null,
    };
  }
  return {
    loadError: null,
    loading: true,
    mergingModels: false,
    report: null,
    status: "读取结算样本…",
    liveToggleKey: null,
  };
}

async function loadResearchReport({ duration, normalizedSymbol, setState, signal }) {
  if (!normalizedSymbol || !duration) {
    setState({
      loadError: "交易对或周期无效",
      loading: false,
      mergingModels: false,
      report: null,
      status: "交易对或周期无效",
    });
    return;
  }
  setState((current) => ({
    ...current,
    loadError: null,
    loading: !current.report,
    mergingModels: Boolean(current.report),
    status: current.report ? `${reportStatus(current.report)} · 刷新结算样本…` : "读取结算样本…",
  }));
  try {
    const report = await fetchPaperLiveCandidates(normalizedSymbol, duration, { signal });
    if (signal?.aborted) return;
    publishReport(setState, report, `${reportStatus(report)} · 合并模型族状态…`);
    storeResearchDashboardCache(normalizedSymbol, duration, report);
    void mergeModelBundle({ duration, normalizedSymbol, report, setState, signal });
  } catch (error) {
    if (signal?.aborted) return;
    publishLoadError(setState, "读取失败", error);
  }
}

async function runDailyLoopReport({ duration, normalizedSymbol, setState }) {
  setState((current) => ({
    ...current,
    loadError: null,
    loading: true,
    mergingModels: false,
    status: "执行 paper-live 日闭环…",
  }));
  try {
    const result = await runPaperLiveDailyLoop(normalizedSymbol, duration);
    const first = Array.isArray(result.results) ? result.results[0] : null;
    const report = first?.candidatePool || null;
    const status = `日闭环 ${result.status || first?.status || "unknown"}`;
    publishReport(
      setState,
      report,
      report ? `${status} · 合并模型族状态…` : `${status} · 未返回 candidatePool`,
    );
    if (!report) return;
    storeResearchDashboardCache(normalizedSymbol, duration, report);
    void mergeModelBundle({ duration, normalizedSymbol, report, setState });
  } catch (error) {
    publishLoadError(setState, "日闭环失败", error);
  }
}

async function toggleLiveTradingReport({
  duration,
  liveTradingEnabled,
  normalizedSymbol,
  row,
  setState,
}) {
  const candidateKey = row?.candidateKey;
  if (!candidateKey) throw new Error("candidateKey is required");
  setState((current) => ({
    ...current,
    loadError: null,
    liveToggleKey: candidateKey,
    status: `${row.name} · ${liveTradingEnabled ? "开启" : "关闭"}实盘…`,
  }));
  try {
    const result = await setPaperLiveCandidateLiveTrading(
      normalizedSymbol,
      duration,
      candidateKey,
      liveTradingEnabled,
    );
    publishLiveToggleReport({ duration, normalizedSymbol, report: result.report, setState });
  } catch (error) {
    publishLiveToggleError({ error, row, setState });
  }
}

function publishLiveToggleReport({ duration, normalizedSymbol, report, setState }) {
  publishReport(setState, report, `${reportStatus(report)} · 合并模型族状态…`);
  storeResearchDashboardCache(normalizedSymbol, duration, report);
  void mergeModelBundle({ duration, normalizedSymbol, report, setState });
}

function publishLiveToggleError({ error, row, setState }) {
  const message = errorMessage(error);
  setState((current) => ({
    ...current,
    liveToggleKey: null,
    status: `${row?.name || "候选"} · 实盘切换失败：${message}`,
  }));
}

async function mergeModelBundle({ duration, normalizedSymbol, report, setState, signal }) {
  setState((current) => ({ ...current, mergingModels: true }));
  try {
    const bundle = await fetchResearchModelBundle(normalizedSymbol, duration, { signal });
    if (signal?.aborted) return;
    const models = bundleModels(bundle, normalizedSymbol, duration);
    const merged = mergeModelFamilyStatusRows(report, models);
    setState({
      loadError: null,
      liveToggleKey: null,
      loading: false,
      mergingModels: false,
      report: merged,
      status: reportStatus(merged),
    });
    storeResearchDashboardCache(normalizedSymbol, duration, merged);
  } catch (error) {
    if (signal?.aborted) return;
    setState((current) => ({
      ...current,
      liveToggleKey: null,
      loading: false,
      mergingModels: false,
      status: `${reportStatus(current.report || report)} · 模型族合并失败：${errorMessage(error)}`,
    }));
  }
}

function bundleModels(bundle, symbol, duration) {
  return (Array.isArray(bundle?.models) ? bundle.models : []).map((status) =>
    modelStatusCandidate({ status, symbol, duration }),
  );
}

function modelStatusCandidate({ status, symbol, duration }) {
  const admission = status?.paperLiveAdmission || {};
  const family = status?.modelFamily || "unknown";
  return {
    candidateKey: status?.modelVersion || `${family}:${symbol}:${duration}`,
    candidateType: "model",
    strategyKey: status?.strategyKey,
    modelFamily: family,
    modelVersion: status?.modelVersion,
    featureWindow: status?.featureWindow,
    minConfidence: admission.minConfidence ?? status?.selectedConfidenceThreshold,
    validationWinRate: admission.validationWinRate ?? status?.validationWinRate,
    validationSampleCount: status?.validationSampleCount,
    oosWinRate: status?.testWinRate,
    paperLiveWinRate: null,
    paperLiveSampleCount: 0,
    paperLiveStatus: status?.paperLiveStatus || admission.status || "backtest_candidate",
    reason:
      admission.reason ||
      status?.shadowPredictionBlockedReason ||
      status?.validationFailureReason,
    cleanEventFeatures: status?.cleanEventFeatures,
    regimeValidation: status?.regimeValidation,
    shadowPredictionBlockedReason: status?.shadowPredictionBlockedReason,
    metrics: { sampleCount: 0 },
    symbol,
    duration,
  };
}

function publishLoadError(setState, prefix, error) {
  const message = errorMessage(error);
  setState((current) => ({
    ...current,
    loadError: message,
    liveToggleKey: null,
    loading: false,
    mergingModels: false,
    status: `${prefix}：${message}`,
  }));
}

function publishReport(setState, report, status) {
  const loadError = report ? null : "candidatePool_missing";
  setState({
    loadError,
    loading: false,
    mergingModels: false,
    report,
    status,
    liveToggleKey: null,
  });
}
