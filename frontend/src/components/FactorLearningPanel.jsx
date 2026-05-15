import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorLearningMemory,
  fetchFactorLearningOperators,
  fetchLstmStatus,
  requestFactorLearningRefresh,
} from "../api/factorLearning";
import FactorAdaptiveLearningPanel from "./FactorAdaptiveLearningPanel";
import FactorLearningCandidateIdeas from "./FactorLearningCandidateIdeas";
import FactorLearningMemoryGrid from "./FactorLearningMemoryGrid";
import FactorLearningOperatorLibrary from "./FactorLearningOperatorLibrary";
import FactorLearningStatusBoxes from "./FactorLearningStatusBoxes";
import "./FactorLearningPanel.css";
import "./FactorLearningCards.css";

const AGENT_RELOAD_DELAY_MS = 5000;

export default function FactorLearningPanel({ symbol, duration }) {
  const learning = useFactorLearningData(symbol, duration);

  return (
    <section className="factor-learning-panel">
      <LearningHeader
        memory={learning.memoryState.data}
        operatorState={learning.operatorState}
        status={learning.memoryState.status}
        refreshing={learning.refreshing}
        onRefreshAgent={() => void learning.refresh(true)}
        onRefreshLocal={() => void learning.refresh(false)}
      />
      <FactorAdaptiveLearningPanel
        learning={learning.memoryState.data?.adaptiveLearning}
        lstm={learning.lstmState.data || learning.memoryState.data?.lstmShadow}
        lstmStatus={learning.lstmState.status}
      />
      <FactorLearningCandidateIdeas memory={learning.memoryState.data} />
      <FactorLearningMemoryGrid memory={learning.memoryState.data} />
      <FactorLearningStatusBoxes
        memory={learning.memoryState.data}
        refreshing={learning.refreshing}
        onRefreshLocal={() => void learning.refresh(false)}
      />
      <FactorLearningOperatorLibrary operators={learning.operatorState.data?.operators || []} />
    </section>
  );
}

function useFactorLearningData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [memoryState, setMemoryState] = useState({ data: null, status: "" });
  const [operatorState, setOperatorState] = useState({ data: null, status: "" });
  const [lstmState, setLstmState] = useState({ data: null, status: "" });
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async (signal) => {
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
  }, [duration, normalizedSymbol]);

  useEffect(() => {
    const ac = new AbortController();
    void loadData(ac.signal);
    return () => ac.abort();
  }, [loadData]);

  const refresh = useCallback(async (runAgent) => {
    if (!isValidSymbol(normalizedSymbol)) {
      setMemoryState({ data: null, status: "请输入有效交易对" });
      return;
    }
    setRefreshing(true);
    setMemoryState((state) => ({ ...state, status: runAgent ? "联网挖掘中…" : "本地复盘中…" }));
    try {
      await refreshLearningMemory({ duration, loadData, normalizedSymbol, runAgent, setMemoryState });
    } catch (error) {
      setMemoryState((state) => ({ ...state, status: `刷新失败：${error.message}` }));
    } finally {
      setRefreshing(false);
    }
  }, [duration, loadData, normalizedSymbol]);

  return { memoryState, operatorState, lstmState, refresh, refreshing };
}

async function refreshLearningMemory({ normalizedSymbol, duration, runAgent, loadData, setMemoryState }) {
  const data = await requestFactorLearningRefresh(normalizedSymbol, duration, runAgent);
  setMemoryState({ data, status: memoryStatus(data) });
  if (data.agentQueued) {
    window.setTimeout(() => void loadData(new AbortController().signal), AGENT_RELOAD_DELAY_MS);
  }
}

async function loadMemory({ symbol, duration, signal, setState }) {
  try {
    const data = await fetchFactorLearningMemory(symbol, duration, { signal });
    if (!signal.aborted) setState({ data, status: memoryStatus(data) });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    const detail = error?.response?.data?.detail || error.message;
    const status = error?.response?.status === 404 ? `暂无因子学习记忆：${detail}` : `读取失败：${detail}`;
    setState({ data: null, status });
  }
}

async function loadOperators(signal, setState) {
  try {
    const data = await fetchFactorLearningOperators({ signal });
    if (!signal.aborted) setState({ data, status: `算子库：${data.total ?? 0} 个` });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setState({ data: null, status: `算子库失败：${error?.response?.data?.detail || error.message}` });
  }
}

async function loadLstmStatus({ symbol, duration, signal, setState }) {
  try {
    const data = await fetchLstmStatus(symbol, duration, { signal });
    if (!signal.aborted) setState({ data, status: lstmStatusText(data) });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setState({ data: null, status: `LSTM状态失败：${error?.response?.data?.detail || error.message}` });
  }
}

function LearningHeader(props) {
  const agent = props.memory?.llmAgent;
  const source = props.memory?.source || {};
  const agentStatus = agent?.status || (agent?.review ? "done" : "idle");
  return (
    <div className="factor-learning-head">
      <div className="factor-learning-head-main">
        <span className="section-kicker">因子学习 / 自动挖掘</span>
        <h2>{agent?.model || "Kimi 因子挖掘 Agent"}</h2>
        <div className="factor-learning-status-line">
          <HeaderStatus status={agentStatus} text={agentStatusLabel(agentStatus)} />
          <HeaderStatus text={props.status} />
          <HeaderStatus text={props.operatorState.status} />
        </div>
      </div>
      <div className="factor-learning-actions">
        <Metric label="结算样本" value={source.settledPredictionCount ?? "—"} />
        <Metric label="亏损模式" value={source.lossPatternCount ?? "—"} />
        <button
          type="button"
          className="factor-learning-action-secondary"
          disabled={props.refreshing}
          onClick={props.onRefreshLocal}
        >
          本地复盘
        </button>
        <button
          type="button"
          className="factor-learning-action-primary"
          disabled={props.refreshing}
          onClick={props.onRefreshAgent}
        >
          联网挖掘
        </button>
      </div>
    </div>
  );
}

function HeaderStatus({ status = "info", text }) {
  if (!text) return null;
  return <span className={`factor-learning-status is-${status}`}>{text}</span>;
}

function Metric({ label, value }) {
  return (
    <span className="factor-learning-metric">
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function agentStatusLabel(status) {
  const labels = {
    done: "Agent 已完成",
    failed: "Agent 失败",
    idle: "等待挖掘",
    pending: "Agent 排队中",
    running: "Agent 运行中",
  };
  return labels[status] || status;
}

function memoryStatus(data) {
  const updated = data?.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  const status = data?.llmAgent?.status;
  const failed = data?.llmAgent?.error || "查看后台日志";
  const agent = status === "pending" ? " · 联网挖掘已排队"
    : status === "failed" ? ` · 联网挖掘失败：${failed}` : data?.llmAgent?.review ? " · 已联网挖掘" : "";
  return `记忆已加载${agent}${updated}`;
}

function lstmStatusText(data) {
  const label = data?.status || "untrained";
  const version = data?.modelVersion ? ` · ${data.modelVersion}` : "";
  const ready = data?.shadowPredictionReady ? " · 可模拟下单" : ` · 阻断：${lstmBlockedReasonLabel(data?.shadowPredictionBlockedReason)}`;
  return `LSTM：${label}${version}${ready}`;
}

function lstmBlockedReasonLabel(reason) {
  const labels = {
    torch_unavailable: "Torch不可用",
    artifacts_incomplete: "模型文件不完整",
    trained_combo_snapshot_missing: "训练组合快照缺失",
    trained_combo_snapshot_incomplete: "训练组合不足Top3",
    current_combo_snapshot_missing: "当前组合排名缺失",
    current_combo_snapshot_incomplete: "当前组合不足Top3",
    combo_snapshot_mismatch: "组合排名已变化",
  };
  return labels[reason] || reason || "未知原因";
}

function isValidSymbol(value) {
  return value.length >= 6;
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}
