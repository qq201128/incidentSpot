import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorLearningMemory,
  fetchFactorLearningOperators,
  requestFactorLearningRefresh,
} from "../api/factorLearning";
import "./FactorLearningPanel.css";

const TOP_WEIGHT_LIMIT = 10;
const OPERATOR_PREVIEW_LIMIT = 64;

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
    } catch (error) {
      setMemoryState((state) => ({ ...state, status: `刷新失败：${error.message}` }));
    } finally {
      setRefreshing(false);
    }
  }, [duration, normalizedSymbol]);

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
      <CandidateIdeas memory={memoryState.data} />
      <LearningMemoryGrid memory={memoryState.data} />
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

function CandidateIdeas({ memory }) {
  const ideas = candidateIdeas(memory);
  return (
    <div className="factor-learning-section">
      <div className="factor-learning-title">
        <h3>候选挖掘因子</h3>
        <span>{ideas.length} 项</span>
      </div>
      <div className="factor-learning-candidates">
        {ideas.map((idea, index) => (
          <article key={`${idea.nameHint || "idea"}-${index}`} className="factor-learning-candidate">
            <div className="factor-learning-candidate-top">
              <strong>{idea.nameHint || `candidate_${index + 1}`}</strong>
              <span>{operatorTrace(idea).join(" · ") || "operator_trace_empty"}</span>
            </div>
            <code>{idea.formulaHint || "—"}</code>
            <p>{idea.rationale || "—"}</p>
            <TagList items={idea.requiredColumns} empty="无列要求" />
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
      <PatternBox title="成功模式" items={successPatterns(memory)} valueKey="label" metaKey="support" />
      <PatternBox title="禁区" items={forbiddenRegions(memory)} valueKey="region" metaKey="avgAbsCorrelation" />
      <PatternBox title="亏损特征" items={lossPatterns(memory)} valueKey="feature" metaKey="lossRate" />
      <WeightsBox weights={memory?.weights || {}} />
    </div>
  );
}

function PatternBox({ title, items, valueKey, metaKey }) {
  return (
    <section className="factor-learning-box">
      <div className="factor-learning-title compact">
        <h3>{title}</h3>
        <span>{items.length}</span>
      </div>
      <ul>
        {items.slice(0, 8).map((item, index) => (
          <li key={`${title}-${index}`}>
            <strong>{item[valueKey] || item.pattern || "—"}</strong>
            <span>{metaLabel(metaKey, item[metaKey])}</span>
          </li>
        ))}
      </ul>
      {!items.length ? <p className="factor-learning-empty small">暂无数据</p> : null}
    </section>
  );
}

function WeightsBox({ weights }) {
  const rows = Object.entries(weights)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, TOP_WEIGHT_LIMIT);
  return (
    <section className="factor-learning-box">
      <div className="factor-learning-title compact">
        <h3>自动权重</h3>
        <span>{Object.keys(weights).length}</span>
      </div>
      <ul>
        {rows.map(([name, value]) => (
          <li key={name}>
            <strong>{name}</strong>
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
    <div className="factor-learning-operators">
      <div className="factor-learning-title">
        <h3>运算符库</h3>
        <span>{operators.length} 个</span>
      </div>
      <div className="factor-learning-operator-grid">
        {grouped.slice(0, OPERATOR_PREVIEW_LIMIT).map(([category, items]) => (
          <section key={category} className="factor-learning-operator-family">
            <h4>{category}</h4>
            <div>
              {items.map((item) => (
                <span key={item.name} title={`${item.signature} · ${item.purpose}`}>
                  {item.name}
                </span>
              ))}
            </div>
          </section>
        ))}
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

function successPatterns(memory) {
  return memory?.factorMining?.successPatterns || [];
}

function forbiddenRegions(memory) {
  return memory?.factorMining?.forbiddenRegions || [];
}

function lossPatterns(memory) {
  return memory?.lossMemory?.patterns || [];
}

function operatorTrace(idea) {
  return Array.isArray(idea.operatorTrace) ? idea.operatorTrace.filter(Boolean) : [];
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
  const agent = data?.llmAgent?.review ? " · 已联网挖掘" : "";
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
