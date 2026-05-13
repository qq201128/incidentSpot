import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorLearningMemory,
  fetchFactorLearningOperators,
  requestFactorLearningRefresh,
} from "../api/factorLearning";
import FactorAdaptiveLearningPanel from "./FactorAdaptiveLearningPanel";
import FactorLearningCandidateIdeas from "./FactorLearningCandidateIdeas";
import FactorLearningMemoryGrid from "./FactorLearningMemoryGrid";
import FactorLearningMetricsHelp from "./FactorLearningMetricsHelp";
import FactorLearningOperatorLibrary from "./FactorLearningOperatorLibrary";
import FactorLearningStatusBoxes from "./FactorLearningStatusBoxes";
import "./FactorLearningPanel.css";

const AGENT_RELOAD_DELAY_MS = 5000;

export default function FactorLearningPanel({ symbol, duration }) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [memoryState, setMemoryState] = useState({ data: null, status: "" });
  const [operatorState, setOperatorState] = useState({ data: null, status: "" });
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
      const data = await requestFactorLearningRefresh(normalizedSymbol, duration, runAgent);
      setMemoryState({ data, status: memoryStatus(data) });
      if (data.agentQueued) {
        window.setTimeout(() => void loadData(new AbortController().signal), AGENT_RELOAD_DELAY_MS);
      }
    } catch (error) {
      setMemoryState((state) => ({ ...state, status: `刷新失败：${error.message}` }));
    } finally {
      setRefreshing(false);
    }
  }, [duration, loadData, normalizedSymbol]);

  return (
    <section className="factor-learning-panel">
      <LearningHeader
        memory={memoryState.data}
        operatorState={operatorState}
        status={memoryState.status}
        refreshing={refreshing}
        onRefreshAgent={() => void refresh(true)}
        onRefreshLocal={() => void refresh(false)}
      />
      <FactorAdaptiveLearningPanel learning={memoryState.data?.adaptiveLearning} />
      <FactorLearningMetricsHelp />
      <FactorLearningCandidateIdeas memory={memoryState.data} />
      <FactorLearningMemoryGrid memory={memoryState.data} />
      <FactorLearningStatusBoxes memory={memoryState.data} />
      <FactorLearningOperatorLibrary operators={operatorState.data?.operators || []} />
    </section>
  );
}

async function loadMemory({ symbol, duration, signal, setState }) {
  try {
    const data = await fetchFactorLearningMemory(symbol, duration, { signal });
    if (!signal.aborted) setState({ data, status: memoryStatus(data) });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    const status = error?.response?.status === 404 ? "暂无因子学习记忆" : `读取失败：${error.message}`;
    setState({ data: null, status });
  }
}

async function loadOperators(signal, setState) {
  try {
    const data = await fetchFactorLearningOperators({ signal });
    if (!signal.aborted) setState({ data, status: `算子库：${data.total ?? 0} 个` });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setState({ data: null, status: `算子库失败：${error.message}` });
  }
}

function LearningHeader(props) {
  const agent = props.memory?.llmAgent;
  const source = props.memory?.source || {};
  return (
    <div className="factor-learning-head">
      <div>
        <span className="section-kicker">因子学习 / 自动挖掘</span>
        <h2>{agent?.model || "Kimi 因子挖掘 Agent"}</h2>
        <p>{props.status}</p>
        <p>{props.operatorState.status}</p>
      </div>
      <div className="factor-learning-actions">
        <Metric label="结算样本" value={source.settledPredictionCount ?? "—"} />
        <Metric label="亏损模式" value={source.lossPatternCount ?? "—"} />
        <button type="button" disabled={props.refreshing} onClick={props.onRefreshLocal}>
          本地复盘
        </button>
        <button type="button" disabled={props.refreshing} onClick={props.onRefreshAgent}>
          联网挖掘
        </button>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <span className="factor-learning-metric">
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function memoryStatus(data) {
  const updated = data?.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  const status = data?.llmAgent?.status;
  const failed = data?.llmAgent?.error || "查看后台日志";
  const agent = status === "pending" ? " · 联网挖掘已排队"
    : status === "failed" ? ` · 联网挖掘失败：${failed}` : data?.llmAgent?.review ? " · 已联网挖掘" : "";
  return `记忆已加载${agent}${updated}`;
}

function isValidSymbol(value) {
  return value.length >= 6;
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}
