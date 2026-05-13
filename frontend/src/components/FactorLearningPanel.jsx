import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorLearningMemory,
  fetchFactorLearningOperators,
  requestFactorLearningRefresh,
} from "../api/factorLearning";
import {
  categoryLabel,
  columnLabels,
  factorLabel,
  learningPatternLabel,
  operatorLabel,
  operatorTraceLabel,
} from "../utils/factorLearningLabels";
import FactorLearningMetricsHelp from "./FactorLearningMetricsHelp";
import FactorLearningStatusBoxes from "./FactorLearningStatusBoxes";
import "./FactorLearningPanel.css";

const TOP_WEIGHT_LIMIT = 10;
const OPERATOR_PREVIEW_LIMIT = 64;
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
      loadMemory(normalizedSymbol, duration, signal, setMemoryState),
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
      <FactorLearningMetricsHelp />
      <CandidateIdeas memory={memoryState.data} />
      <LearningMemoryGrid memory={memoryState.data} />
      <FactorLearningStatusBoxes memory={memoryState.data} />
      <OperatorLibrary operators={operatorState.data?.operators || []} />
    </section>
  );
}

async function loadMemory(symbol, duration, signal, setState) {
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
        <Metric
          label="结算样本"
          title="已结算、并参与亏损模式学习的预测条数。"
          value={source.settledPredictionCount ?? "—"}
        />
        <Metric
          label="亏损模式"
          title="当前识别出的亏损特征模式数量。"
          value={source.lossPatternCount ?? "—"}
        />
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

function CandidateIdeas({ memory }) {
  const ideas = candidateIdeas(memory);
  return (
    <div
      className="factor-learning-section"
      title="Agent 或规划给出的下一步可挖因子设想（公式提示、理由、所需列与校验项）。"
    >
      <div className="factor-learning-title">
        <h3>候选挖掘因子</h3>
        <span>{ideas.length} 项</span>
      </div>
      <div className="factor-learning-candidates">
        {ideas.map((idea, index) => (
          <article key={`${idea.nameHint || "idea"}-${index}`} className="factor-learning-candidate">
            <div className="factor-learning-candidate-top">
              <strong>{candidateIdeaTitle(idea, index)}</strong>
              <span>{operatorTraceLabel(idea.operatorTrace).join(" · ") || "无算子链"}</span>
            </div>
            <code>{idea.formulaHint || "—"}</code>
            <p>{idea.rationaleZh || idea.rationale || "—"}</p>
            <TagList items={columnLabels(idea.requiredColumns)} empty="无列要求" />
            <TagList items={idea.validationChecks} empty="无验证项" muted />
          </article>
        ))}
        {!ideas.length ? <p className="factor-learning-empty">暂无候选因子</p> : null}
      </div>
    </div>
  );
}

function LearningMemoryGrid({ memory }) {
  return (
    <div className="factor-learning-grid">
      <PatternBox
        title="成功模式"
        sectionHint="历史表现达标的因子按因子族/算子归类。右侧整数为支持度（落入该类的因子个数，非百分比）。"
        items={successPatterns(memory)}
        valueKey="label"
        metaKey="support"
      />
      <PatternBox
        title="禁区"
        sectionHint="高相关因子簇；右侧为簇内平均绝对 Spearman 相关系数（越高越冗余）。"
        items={forbiddenRegions(memory)}
        valueKey="region"
        metaKey="avgAbsCorrelation"
      />
      <PatternBox
        title="亏损特征"
        sectionHint="与亏损显著相关的因子方向/阈值。右侧为命中该规则时的亏损占比。"
        items={lossPatterns(memory)}
        valueKey="feature"
        metaKey="lossRate"
      />
      <WeightsBox
        weights={memory?.weights || {}}
        sectionHint="组合打分相对权重（约 100%）；命中亏损特征的因子会被降权。列表最多 10 条。"
      />
    </div>
  );
}

function PatternBox({ title, sectionHint, items, valueKey, metaKey }) {
  return (
    <section className="factor-learning-box" title={sectionHint}>
      <div className="factor-learning-title compact">
        <h3>{title}</h3>
        <span>{items.length}</span>
      </div>
      <ul>
        {items.slice(0, 8).map((item, index) => (
          <li key={`${title}-${index}`}>
            <strong>{learningPatternLabel(item, valueKey)}</strong>
            <span>{metaLabel(metaKey, item[metaKey])}</span>
          </li>
        ))}
      </ul>
      {!items.length ? <p className="factor-learning-empty small">暂无数据</p> : null}
    </section>
  );
}

function WeightsBox({ weights, sectionHint }) {
  const rows = Object.entries(weights)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, TOP_WEIGHT_LIMIT);
  return (
    <section className="factor-learning-box" title={sectionHint}>
      <div className="factor-learning-title compact">
        <h3>自动权重</h3>
        <span>{Object.keys(weights).length}</span>
      </div>
      <ul>
        {rows.map(([name, value]) => (
          <li key={name}>
            <strong>{factorLabel(name)}</strong>
            <span>{formatPct(value, 1)}</span>
          </li>
        ))}
      </ul>
      {!rows.length ? <p className="factor-learning-empty small">暂无权重</p> : null}
    </section>
  );
}

function OperatorLibrary({ operators }) {
  const grouped = groupOperators(operators);
  return (
    <div
      className="factor-learning-operators"
      title="按因子族分组的算子（对行情列做变换）。悬停单个标签可查看签名与用途。"
    >
      <div className="factor-learning-title">
        <h3>运算符库</h3>
        <span>{operators.length} 个</span>
      </div>
      <div className="factor-learning-operator-grid">
        {grouped.slice(0, OPERATOR_PREVIEW_LIMIT).map(([category, items]) => (
          <section key={category} className="factor-learning-operator-family">
            <h4>{categoryLabel(category)}</h4>
            <div>
              {items.map((item) => (
                <span key={item.name} title={`${item.signature} · ${item.purpose}`}>
                  {operatorLabel(item.name)}
                </span>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value, title: hint }) {
  return (
    <span className="factor-learning-metric" title={hint}>
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function TagList({ items, empty, muted = false }) {
  const rows = Array.isArray(items) ? items.filter(Boolean) : [];
  return (
    <div className={`factor-learning-tags${muted ? " muted" : ""}`}>
      {rows.length ? rows.slice(0, 6).map((item) => <span key={item}>{item}</span>) : <span>{empty}</span>}
    </div>
  );
}

function candidateIdeas(memory) {
  return memory?.llmAgent?.review?.factorMiningPlan?.candidateFactorIdeas || [];
}

function candidateIdeaTitle(idea, index) {
  if (idea.displayNameZh) return idea.displayNameZh;
  if (idea.nameHint) return factorLabel(idea.nameHint);
  return `候选因子 ${index + 1}`;
}

function successPatterns(memory) {
  return memory?.factorMining?.successPatterns || [];
}

function forbiddenRegions(memory) {
  return memory?.factorMining?.forbiddenRegions || [];
}

function lossPatterns(memory) {
  return memory?.lossMemory?.patterns || [];
}

function groupOperators(operators) {
  const grouped = new Map();
  for (const operator of operators) {
    const key = operator.category || "other";
    grouped.set(key, [...(grouped.get(key) || []), operator]);
  }
  return Array.from(grouped.entries());
}

function metaLabel(key, value) {
  if (key === "lossRate" || key === "avgAbsCorrelation") return formatPct(value, 1);
  return value ?? "—";
}

function memoryStatus(data) {
  const updated = data?.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  const status = data?.llmAgent?.status;
  const failed = data?.llmAgent?.error || "查看后台日志";
  const agent = status === "pending" ? " · 联网挖掘已排队"
    : status === "failed" ? ` · 联网挖掘失败：${failed}` : data?.llmAgent?.review ? " · 已联网挖掘" : "";
  return `记忆已加载${agent}${updated}`;
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function isValidSymbol(value) {
  return value.length >= 6;
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}
