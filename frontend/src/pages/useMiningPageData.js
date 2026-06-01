import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchMiningOverview } from "../api/miningClient";
import {
  DEFAULT_MODEL_SEARCH_RESOURCE,
  requestFactorLearningRefresh,
  requestModelCandidateSearch,
} from "../api/factorLearning";
import { MODEL_FAMILIES } from "../utils/modelFamilies";

const POLL_MS = 3000;
const REFRESH_SETTLE_DELAY_MS = 400;

export function useMiningPageData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [overview, setOverview] = useState(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(
    async (signal, { fresh = false } = {}) => {
      if (!isValidSymbol(normalizedSymbol)) {
        setOverview(null);
        setLoading(false);
        setStatus("请输入有效交易对");
        return;
      }
      try {
        const data = await fetchMiningOverview(normalizedSymbol, duration, { signal, fresh });
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
        await sleep(REFRESH_SETTLE_DELAY_MS);
        await load(undefined, { fresh: true });
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
        const data = await fetchMiningOverview(normalizedSymbol, duration, {
          signal: ac.signal,
          fresh: true,
        });
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
        MODEL_FAMILIES.map((family) =>
          requestModelCandidateSearch(family, normalizedSymbol, duration, "full", DEFAULT_MODEL_SEARCH_RESOURCE),
        ),
      );
      await load(undefined, { fresh: true });
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
        await requestModelCandidateSearch(family, normalizedSymbol, duration, "full", DEFAULT_MODEL_SEARCH_RESOURCE);
        await load(undefined, { fresh: true });
      } catch (error) {
        setStatus(`${family} 搜索失败：${errorMessage(error)}`);
      } finally {
        setBusy("");
      }
    },
    [duration, load, normalizedSymbol],
  );

  const reload = useCallback(() => {
    setLoading(true);
    const ac = new AbortController();
    void load(ac.signal, { fresh: true });
    return () => ac.abort();
  }, [load]);

  return {
    overview,
    status,
    busy,
    loading,
    refreshLocal,
    refreshAgent,
    searchAllModels,
    searchModel,
    reload,
  };
}

function hasActiveTasks(overview) {
  const states = [
    overview?.runStatus?.overall?.state,
    overview?.runStatus?.sections?.worker?.state,
    overview?.runStatus?.sections?.modelSearch?.state,
    ...(overview?.runStatus?.models || []).map((row) => row.state),
  ];
  return states.some((state) => ["queued", "running", "worker_required"].includes(state));
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

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
