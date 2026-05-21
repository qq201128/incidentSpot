import FactorAdaptiveLearningPanel from "./FactorAdaptiveLearningPanel";
import FactorLearningCandidateIdeas from "./FactorLearningCandidateIdeas";
import FactorLearningMemoryGrid from "./FactorLearningMemoryGrid";
import FactorLearningOperatorLibrary from "./FactorLearningOperatorLibrary";
import FactorLearningStatusBoxes from "./FactorLearningStatusBoxes";
import {
  refreshTaskLabel,
  refreshTaskStatus,
  useFactorLearningData,
} from "./useFactorLearningData";
import "./FactorLearningPanel.css";
import "./FactorLearningCards.css";

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
        onSearchCandidates={(family) => void learning.startLstmSearch(family)}
        searchStatus={learning.lstmSearchState}
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

function LearningHeader(props) {
  const agent = props.memory?.llmAgent;
  const source = props.memory?.source || {};
  const refreshTask = props.memory?.refreshTask;
  const minedLibrary = props.memory?.minedFactorLibrary || {};
  const agentLibrary = props.memory?.agentMinedFactorLibrary || {};
  const agentStatus = agent?.status || (agent?.review ? "done" : "idle");
  return (
    <div className="factor-learning-head">
      <div className="factor-learning-head-main">
        <span className="section-kicker">因子学习 / 自动挖掘</span>
        <h2>{agent?.model || "Kimi 因子挖掘 Agent"}</h2>
        <div className="factor-learning-status-line">
          <HeaderStatus status={refreshTaskStatus(refreshTask)} text={refreshTaskLabel(refreshTask, source)} />
          <HeaderStatus status={agentStatus} text={agentStatusLabel(agentStatus)} />
          <HeaderStatus text={props.status} />
          <HeaderStatus text={props.operatorState.status} />
          <HeaderStatus text={rankingRefreshSourceLabel(source.rankingRefreshSource)} />
        </div>
      </div>
      <div className="factor-learning-actions">
        <Metric label="结算样本" value={source.settledPredictionCount ?? "—"} />
        <Metric label="亏损模式" value={source.lossPatternCount ?? "—"} />
        <Metric label="组合回灌" value={minedLibrary.total ?? "—"} />
        <Metric label="Agent入库" value={agentLibrary.total ?? "—"} />
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
    completed: "Agent 已完成",
    done: "Agent 已完成",
    failed: "Agent 失败",
    idle: "等待挖掘",
    pending: "Agent 排队中",
    running: "Agent 运行中",
  };
  return labels[status] || status;
}

function rankingRefreshSourceLabel(value) {
  const labels = {
    cache: "复盘来源：缓存",
    rebuilt_cache: "复盘来源：重建缓存",
    provided: "复盘来源：组合重算",
  };
  return labels[value] || (value ? `复盘来源：${value}` : "");
}
