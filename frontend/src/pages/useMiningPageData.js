import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchMiningOverview } from "../api/miningClient";
import {
  requestFactorLearningRefresh,
  requestModelCandidateSearch,
} from "../api/factorLearning";

const POLL_MS = 3000;
const MODEL_FAMILIES = [
  "lstm",
  "gru",
  "cnn",
  "transformer",
  "random_forest",
  "xgboost",
  "svm",
  "rl_strategy",
  "bayesian",
  "knn",
];

export function useMiningPageData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [overview, setOverview] = useState(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(
    async (signal) => {
      if (!isValidSymbol(normalizedSymbol)) {
        setOverview(null);
        setLoading(false);
        setStatus("请输入有效交易对");
        return;
      }
      try {
        const data = await fetchMiningOverview(normalizedSymbol, duration, { signal });
        if (!signal?.aborted) {
          setOverview(data);
          setStatus("");
          setLoading(false);
        }
      } catch (error) {
        if (isCanceled(error, signal)) return;
        const detail = errorMessage(error);
        const message =
          error?.response?.status === 404
            ? `暂无因子学习记忆，请先执行本地复盘：${detail}`
            : `加载失败：${detail}`;
        setOverview(null);
        setStatus(message);
        setLoading(false);
      }
    },
    [duration, normalizedSymbol],
  );

  const queueRefresh = useCallback(
    async (runAgent) => {
      if (!isValidSymbol(normalizedSymbol)) return;
      setBusy(runAgent ? "agent" : "local");
      setStatus(runAgent ? "联网挖掘排队中…" : "本地复盘排队中…");
      try {
        await requestFactorLearningRefresh(normalizedSymbol, duration, runAgent);
        await load();
      } catch (error) {
        setStatus(`刷新失败：${errorMessage(error)}`);
      } finally {
        setBusy("");
      }
    },
    [duration, load, normalizedSymbol],
  );

  useEffect(() => {
    setLoading(true);
    const ac = new AbortController();
    void load(ac.signal);
    return () => ac.abort();
  }, [load]);

  const needsPoll = useMemo(() => hasActiveTasks(overview), [overview]);

  useEffect(() => {
    if (!needsPoll || !isValidSymbol(normalizedSymbol)) return undefined;
    const ac = new AbortController();
    let timer;
    const poll = async () => {
      try {
        const data = await fetchMiningOverview(normalizedSymbol, duration, { signal: ac.signal });
        if (!ac.signal.aborted) setOverview(data);
        if (!ac.signal.aborted && hasActiveTasks(data)) timer = window.setTimeout(poll, POLL_MS);
      } catch (error) {
        if (!isCanceled(error, ac.signal)) setStatus(`轮询失败：${errorMessage(error)}`);
      }
    };
    timer = window.setTimeout(poll, POLL_MS);
    return () => {
      ac.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [duration, needsPoll, normalizedSymbol]);

  const refreshLocal = useCallback(() => queueRefresh(false), [queueRefresh]);
  const refreshAgent = useCallback(() => queueRefresh(true), [queueRefresh]);

  const searchAllModels = useCallback(async () => {
    if (!isValidSymbol(normalizedSymbol)) return;
    setBusy("search-all");
    try {
      await Promise.all(
        MODEL_FAMILIES.map((family) => requestModelCandidateSearch(family, normalizedSymbol, duration)),
      );
      await load();
    } catch (error) {
      setStatus(`全量搜索失败：${errorMessage(error)}`);
    } finally {
      setBusy("");
    }
  }, [duration, load, normalizedSymbol]);

  const searchModel = useCallback(
    async (family) => {
      if (!isValidSymbol(normalizedSymbol)) return;
      setBusy(`search-${family}`);
      try {
        await requestModelCandidateSearch(family, normalizedSymbol, duration);
        await load();
      } catch (error) {
        setStatus(`${family} 搜索失败：${errorMessage(error)}`);
      } finally {
        setBusy("");
      }
    },
    [duration, load, normalizedSymbol],
  );

  return {
    overview,
    status,
    busy,
    loading,
    refreshLocal,
    refreshAgent,
    searchAllModels,
    searchModel,
    reload: load,
  };
}

function hasActiveTasks(overview) {
  const memory = overview?.memory;
  if (!memory) return false;
  const refresh = memory.refreshTask?.status;
  const agent = memory.llmAgent?.status;
  if (["queued", "running"].includes(refresh) || ["pending", "running"].includes(agent)) return true;
  return (overview?.models || []).some((row) => ["queued", "running"].includes(row.searchStatus));
}

function isValidSymbol(value) {
  return value.length >= 6;
}

function isCanceled(error, signal) {
  return signal?.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}

function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || "unknown_error";
}
