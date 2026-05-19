import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorLearningMemory,
  fetchFactorLearningOperators,
  fetchLstmStatus,
  requestFactorLearningRefresh,
} from "../api/factorLearning";

const TASK_POLL_MS = 3000;

export function useFactorLearningData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [memoryState, setMemoryState] = useState({ data: null, status: "" });
  const [operatorState, setOperatorState] = useState({ data: null, status: "" });
  const [lstmState, setLstmState] = useState({ data: null, status: "" });
  const [queueing, setQueueing] = useState(false);
  const loadData = useDataLoader({ duration, normalizedSymbol, setLstmState, setMemoryState, setOperatorState });

  useInitialLoad(loadData);
  useLearningTaskPoll({ data: memoryState.data, duration, setMemoryState, symbol: normalizedSymbol });

  const refresh = useRefreshQueue({ duration, normalizedSymbol, setMemoryState, setQueueing });
  const refreshing = queueing || hasActiveLearningTask(memoryState.data);
  return { memoryState, operatorState, lstmState, refresh, refreshing };
}

function useDataLoader({ duration, normalizedSymbol, setLstmState, setMemoryState, setOperatorState }) {
  return useCallback(async (signal) => {
    if (!isValidSymbol(normalizedSymbol)) {
      setMemoryState({ data: null, status: "请输入有效交易对" });
      return;
    }
    setMemoryState((state) => ({ ...state, status: "读取因子学习记忆…" }));
    await Promise.all([
      loadMemory({ duration, setState: setMemoryState, signal, symbol: normalizedSymbol }),
      loadOperators(signal, setOperatorState),
      loadLstmStatus({ duration, setState: setLstmState, signal, symbol: normalizedSymbol }),
    ]);
  }, [duration, normalizedSymbol, setLstmState, setMemoryState, setOperatorState]);
}

function useInitialLoad(loadData) {
  useEffect(() => {
    const ac = new AbortController();
    void loadData(ac.signal);
    return () => ac.abort();
  }, [loadData]);
}

function useRefreshQueue({ duration, normalizedSymbol, setMemoryState, setQueueing }) {
  return useCallback(async (runAgent) => {
    if (!isValidSymbol(normalizedSymbol)) {
      setMemoryState({ data: null, status: "请输入有效交易对" });
      return;
    }
    setQueueing(true);
    setMemoryState((state) => ({ ...state, status: runAgent ? "联网挖掘排队中…" : "本地复盘排队中…" }));
    try {
      const data = await requestFactorLearningRefresh(normalizedSymbol, duration, runAgent);
      setMemoryState({ data, status: memoryStatus(data) });
    } catch (error) {
      setMemoryState((state) => ({ ...state, status: `刷新失败：${errorMessage(error)}` }));
    } finally {
      setQueueing(false);
    }
  }, [duration, normalizedSymbol, setMemoryState, setQueueing]);
}

function useLearningTaskPoll({ data, duration, setMemoryState, symbol }) {
  const active = hasActiveLearningTask(data);
  useEffect(() => {
    if (!active || !isValidSymbol(symbol)) return undefined;
    const ac = new AbortController();
    let timer;
    const poll = async () => {
      try {
        const next = await fetchFactorLearningMemory(symbol, duration, { signal: ac.signal });
        if (!ac.signal.aborted) setMemoryState({ data: next, status: memoryStatus(next) });
        if (!ac.signal.aborted && hasActiveLearningTask(next)) timer = window.setTimeout(poll, TASK_POLL_MS);
      } catch (error) {
        if (!isCanceled(error, ac.signal)) setMemoryState((state) => ({ ...state, status: `状态轮询失败：${errorMessage(error)}` }));
      }
    };
    timer = window.setTimeout(poll, TASK_POLL_MS);
    return () => {
      ac.abort();
      if (timer) window.clearTimeout(timer);
    };
  }, [active, duration, setMemoryState, symbol]);
}

async function loadMemory({ symbol, duration, signal, setState }) {
  try {
    const data = await fetchFactorLearningMemory(symbol, duration, { signal });
    if (!signal.aborted) setState({ data, status: memoryStatus(data) });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    const detail = errorMessage(error);
    const status = error?.response?.status === 404 ? `暂无因子学习记忆：${detail}` : `读取失败：${detail}`;
    setState({ data: null, status });
  }
}

async function loadOperators(signal, setState) {
  try {
    const data = await fetchFactorLearningOperators({ signal });
    if (!signal.aborted) setState({ data, status: `算子库：${data.total ?? 0} 个` });
  } catch (error) {
    if (!isCanceled(error, signal)) setState({ data: null, status: `算子库失败：${errorMessage(error)}` });
  }
}

async function loadLstmStatus({ symbol, duration, signal, setState }) {
  try {
    const data = await fetchLstmStatus(symbol, duration, { signal });
    if (!signal.aborted) setState({ data, status: lstmStatusText(data) });
  } catch (error) {
    if (!isCanceled(error, signal)) setState({ data: null, status: `LSTM状态失败：${errorMessage(error)}` });
  }
}

export function memoryStatus(data) {
  const updated = data?.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  return `记忆已加载${refreshStatusText(data)}${agentStatusText(data)}${updated}`;
}

function refreshStatusText(data) {
  const task = data?.refreshTask;
  return task?.status ? ` · ${refreshTaskLabel(task, data?.source || {})}` : "";
}

function agentStatusText(data) {
  const status = data?.llmAgent?.status;
  const failed = data?.llmAgent?.error || "查看后台日志";
  if (status === "pending") return " · 联网挖掘已排队";
  if (status === "running") return " · 联网挖掘运行中";
  if (status === "failed") return ` · 联网挖掘失败：${failed}`;
  return data?.llmAgent?.review ? " · 已联网挖掘" : "";
}

export function refreshTaskLabel(task, source) {
  if (!task?.status && source.status !== "queued") return "";
  const action = task?.runAgent ? "复盘+联网挖掘" : "本地复盘";
  const labels = {
    completed: `${action}已完成`,
    failed: `${action}失败：${task?.error || "查看后台日志"}`,
    queued: `${action}排队中`,
    running: `${action}运行中`,
  };
  return labels[task?.status || source.status] || "";
}

export function refreshTaskStatus(task) {
  if (task?.status === "completed") return "done";
  if (task?.status === "failed") return "failed";
  if (task?.status === "queued") return "pending";
  if (task?.status === "running") return "running";
  return "info";
}

function hasActiveLearningTask(data) {
  const refreshStatus = data?.refreshTask?.status;
  const agentStatus = data?.llmAgent?.status;
  return ["queued", "running"].includes(refreshStatus) || ["pending", "running"].includes(agentStatus);
}

function lstmStatusText(data) {
  const label = data?.status || "untrained";
  const version = data?.modelVersion ? ` · ${data.modelVersion}` : "";
  return `LSTM：${label}${version}${lstmReadyStatusText(data)}`;
}

function lstmReadyStatusText(data) {
  if (data?.shadowPredictionReady || data?.shadowPredictionBlockedReason === "combo_snapshot_mismatch") {
    const combo = data?.comboSnapshotReason === "combo_snapshot_mismatch" ? " · 组合变化继续学习" : "";
    return ` · 可模拟下单${combo}`;
  }
  return ` · 阻断：${lstmBlockedReasonLabel(data?.shadowPredictionBlockedReason)}`;
}

function lstmBlockedReasonLabel(reason) {
  const labels = {
    torch_unavailable: "Torch不可用",
    artifacts_incomplete: "模型文件不完整",
    trained_combo_snapshot_missing: "训练组合快照缺失",
    trained_combo_snapshot_incomplete: "训练组合不足Top3",
    current_combo_snapshot_missing: "当前组合排名缺失",
    current_combo_snapshot_incomplete: "当前组合不足Top3",
  };
  return labels[reason] || reason || "未知原因";
}

function isValidSymbol(value) {
  return value.length >= 6;
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}

function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || "unknown_error";
}
