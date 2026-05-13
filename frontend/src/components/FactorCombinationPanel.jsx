import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorCombinationRanking,
  fetchFactorCombinationSignals,
  requestFactorCombinationRefresh,
} from "../api/factorCombinations";
import FactorComboPositionsPanel from "./FactorComboPositionsPanel";
import "./FactorCombinationPanel.css";

const TOP_PER_DURATION = 3;
const SIGNAL_LIMIT = 12;
const REFRESH_RELOAD_DELAY_MS = 3000;
const RANKING_PREVIEW_LIMIT = 16;
const DURATION_LABELS = { "10m": "10m", "30m": "30m", "60m": "60m", "1d": "1d" };
/** Column order when showing one timeframe per column */
const DURATION_COLUMN_ORDER = ["10m", "30m", "60m", "1d"];

export default function FactorCombinationPanel({ symbol, duration }) {
  const combo = useFactorCombinationData(symbol, duration);
  return <ComboPanelView {...combo} duration={duration} />;
}

function useFactorCombinationData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [rankingState, setRankingState] = useState({ items: [], status: "", updatedAt: null });
  const [signalState, setSignalState] = useState({ items: [], status: "", missing: [], failures: [] });
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async (signal) => {
    if (!isValidSymbol(normalizedSymbol)) {
      setInvalidStates(setRankingState, setSignalState);
      return;
    }
    setRankingState((state) => ({ ...state, status: "加载多因子组合缓存…" }));
    setSignalState((state) => ({ ...state, status: "刷新周期信号…" }));
    await Promise.all([
      loadRankingState(normalizedSymbol, duration, signal, setRankingState),
      loadSignalState(normalizedSymbol, signal, setSignalState),
    ]);
  }, [duration, normalizedSymbol]);

  useEffect(() => {
    const ac = new AbortController();
    void loadData(ac.signal);
    return () => ac.abort();
  }, [loadData]);

  const refresh = useCallback(async (targetDuration) => {
    if (!isValidSymbol(normalizedSymbol)) return setInvalidRanking(setRankingState);
    setRefreshing(true);
    try {
      await requestFactorCombinationRefresh(normalizedSymbol, targetDuration);
      setRankingState((state) => ({ ...state, status: refreshStatus(targetDuration) }));
      window.setTimeout(() => void loadData(new AbortController().signal), REFRESH_RELOAD_DELAY_MS);
    } catch (error) {
      setRankingState((state) => ({ ...state, status: `刷新失败：${error.message}` }));
    } finally {
      setRefreshing(false);
    }
  }, [loadData, normalizedSymbol]);

  return {
    symbol: normalizedSymbol,
    rankingState,
    signalState,
    refreshing,
    onRefreshCurrent: () => void refresh(duration), onRefreshAll: () => void refresh(undefined),
  };
}

async function loadRankingState(symbol, duration, signal, setState) {
  try {
    const data = await fetchFactorCombinationRanking(symbol, duration, { signal });
    if (signal.aborted) return;
    const items = Array.isArray(data.ranking) ? data.ranking : [];
    setState({ items, status: rankingStatus(data, symbol, duration), updatedAt: data.updatedAt });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setState({ items: [], status: `组合排名失败：${error.message}`, updatedAt: null });
  }
}

async function loadSignalState(symbol, signal, setState) {
  try {
    const data = await fetchFactorCombinationSignals(
      symbol,
      SIGNAL_LIMIT,
      { signal, topPerDuration: TOP_PER_DURATION },
    );
    if (signal.aborted) return;
    const items = Array.isArray(data.signals) ? data.signals : [];
    const missing = Array.isArray(data.missingDurations) ? data.missingDurations : [];
    const failures = Array.isArray(data.signalFailures) ? data.signalFailures : [];
    setState({ items, missing, failures, status: signalStatus(items, missing, failures) });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setState({ items: [], missing: [], failures: [], status: `周期信号失败：${error.message}` });
  }
}

function ComboPanelView(props) {
  const signals = props.signalState.items;
  const [selectedKey, setSelectedKey] = useState("");
  const selectedSignal = useMemo(
    () => signals.find((signal) => signalKey(signal) === selectedKey) || null,
    [selectedKey, signals],
  );
  useEffect(() => {
    if (selectedKey && !selectedSignal) setSelectedKey("");
  }, [selectedKey, selectedSignal]);
  return (
    <section className="factor-combo-panel">
      <ComboPanelHeader {...props} />
      {signals.length ? (
        <SignalGrid
          onSelect={(signal) => setSelectedKey(signalKey(signal))}
          selectedKey={selectedKey}
          signals={signals}
        />
      ) : null}
      {selectedSignal ? <FactorComboPositionsPanel signal={selectedSignal} symbol={props.symbol} /> : null}
      <ComboRankingTable ranking={props.rankingState.items} />
    </section>
  );
}

function ComboPanelHeader({
  symbol,
  duration,
  rankingState,
  signalState,
  refreshing,
  onRefreshCurrent,
  onRefreshAll,
}) {
  return (
    <div className="factor-combo-head">
      <div>
        <span className="section-kicker">多因子组合</span>
        <h2>{symbol} · {DURATION_LABELS[duration] || duration}</h2>
        <p>{rankingState.status}</p>
        <p>{signalState.status}</p>
      </div>
      <div className="factor-combo-actions">
        <button type="button" disabled={refreshing} onClick={onRefreshCurrent}>
          当前周期
        </button>
        <button type="button" disabled={refreshing} onClick={onRefreshAll}>
          全部周期
        </button>
      </div>
    </div>
  );
}

function groupSignalsIntoDurationColumns(signals) {
  const byDuration = new Map();
  for (const signal of signals) {
    const d = signal.duration || "—";
    if (!byDuration.has(d)) byDuration.set(d, []);
    byDuration.get(d).push(signal);
  }
  for (const list of byDuration.values()) {
    list.sort((a, b) => (a.comboRank ?? 0) - (b.comboRank ?? 0));
  }
  const ordered = DURATION_COLUMN_ORDER.filter((d) => byDuration.has(d));
  const rest = [...byDuration.keys()]
    .filter((d) => !DURATION_COLUMN_ORDER.includes(d))
    .sort((a, b) => String(a).localeCompare(String(b)));
  return [...ordered, ...rest].map((duration) => ({
    duration,
    signals: byDuration.get(duration),
  }));
}

function SignalGrid({ onSelect, selectedKey, signals }) {
  const columns = useMemo(() => groupSignalsIntoDurationColumns(signals), [signals]);
  return (
    <div className="factor-combo-signals">
      {columns.map(({ duration, signals: columnSignals }) => (
        <div key={duration} className="factor-combo-signal-column" aria-label={`${DURATION_LABELS[duration] || duration} 周期`}>
          {columnSignals.map((signal) => (
            <SignalCard
              key={signalKey(signal)}
              onSelect={onSelect}
              selected={selectedKey === signalKey(signal)}
              signal={signal}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

function SignalCard({ onSelect, selected, signal }) {
  return (
    <button
      type="button"
      className={`factor-combo-signal ${directionClass(signal.direction)}${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(signal)}
    >
      <div className="factor-combo-signal-top">
        <span>{signal.duration} · Top{signal.comboRank || "—"}</span>
        <strong>{directionText(signal.direction)}</strong>
      </div>
      <h3 title={signal.factorDisplayName}>{signal.factorDisplayName || signal.factorName}</h3>
      <p>{memberText(signal.members)}</p>
      <div className="factor-combo-signal-metrics">
        <Metric label="胜率" value={formatPct(signal.historicalWinRate, 1)} />
        <Metric label="盈亏比" value={formatNum(signal.historicalProfitFactor, 2)} />
        <Metric label="置信" value={formatPct(signal.confidence, 1)} />
        <Metric label="模拟" value={signal.qualityPassed ? "候选" : "阻断"} />
      </div>
    </button>
  );
}

function ComboRankingTable({ ranking }) {
  return (
    <div className="factor-combo-ranking">
      <div className="factor-combo-ranking-title">
        <h3>胜率最高组合</h3>
        <span>{ranking.length} 项</span>
      </div>
      <div className="factor-combo-table-wrap">
        <table className="factors-table factor-combo-table">
          <thead>
            <tr>
              <th>#</th>
              <th>组合因子</th>
              <th>成员</th>
              <th>胜率</th>
              <th>贡献</th>
              <th>夏普</th>
              <th>IR</th>
              <th>盈亏比</th>
            </tr>
          </thead>
          <tbody>{ranking.slice(0, RANKING_PREVIEW_LIMIT).map(renderRankingRow)}</tbody>
        </table>
        {!ranking.length ? <p className="factor-combo-empty">暂无组合排名</p> : null}
      </div>
    </div>
  );
}

function renderRankingRow(row, index) {
  return (
    <tr key={row.factorName || index}>
      <td>{index + 1}</td>
      <td>
        <strong className="factors-name-cn">{row.factorDisplayName || row.factorName}</strong>
        <code className="factors-code">{row.factorName}</code>
      </td>
      <td>{memberText(row.members)}</td>
      <td>{formatPct(row.winRate, 1)}</td>
      <td>{formatPct(row.contribution, 1)}</td>
      <td>{formatNum(row.sharpe, 2)}</td>
      <td>{formatNum(row.ir, 4)}</td>
      <td>{formatNum(row.profitFactor, 2)}</td>
    </tr>
  );
}

function Metric({ label, value }) {
  return (
    <span>
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function rankingStatus(data, symbol, duration) {
  if (data.source === "none") {
    return `暂无组合排名缓存（${symbol} / ${duration}）`;
  }
  const updated = data.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  return `组合排名：${data.total ?? 0} 项（${symbol} / ${duration}${updated}）`;
}

function signalStatus(items, missing, failures) {
  const suffix = missing.length ? ` · 缺少 ${missing.join(", ")}` : "";
  const failed = failures.length ? ` · 失败 ${failures.length}` : "";
  return `周期 Top${TOP_PER_DURATION} 模拟：${items.length} 个${suffix}${failed}`;
}

function refreshStatus(duration) {
  return duration ? `已排队重算 ${duration} 多因子组合` : "已排队重算全部周期多因子组合";
}

function setInvalidStates(setRankingState, setSignalState) {
  setInvalidRanking(setRankingState);
  setSignalState({ items: [], status: "请输入有效交易对", missing: [], failures: [] });
}

function setInvalidRanking(setRankingState) {
  setRankingState({ items: [], status: "请输入有效交易对", updatedAt: null });
}

function memberText(members) {
  if (!Array.isArray(members) || !members.length) return "—";
  return members.map((member) => member.displayName || member.name).join(" + ");
}

function directionText(direction) {
  return direction === "down" ? "做空" : "做多";
}

function directionClass(direction) {
  return direction === "down" ? "is-down" : "is-up";
}

function signalKey(signal) {
  return `${signal.duration}-${signal.comboRank || 0}-${signal.factorName || ""}`;
}
function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function isValidSymbol(symbol) {
  return symbol.length >= 6;
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}
