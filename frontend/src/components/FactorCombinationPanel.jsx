import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFactorCombinationRanking,
  fetchFactorCombinationSignals,
  requestFactorCombinationRefresh,
} from "../api/factorCombinations";
import FactorComboPositionsPanel from "./FactorComboPositionsPanel";
import FactorCombinationRankingTable from "./FactorCombinationRankingTable";
import FactorCombinationSignalGrid, { signalKey } from "./FactorCombinationSignalGrid";
import "./FactorCombinationPanel.css";

const TOP_PER_DURATION = 3;
const SIGNAL_LIMIT = 12;
const REFRESH_RELOAD_DELAY_MS = 3000;
const DURATION_LABELS = { "10m": "10m", "30m": "30m", "60m": "60m", "1d": "1d" };
const SIGNAL_IDLE_STATUS = "等待加载周期信号…";

export default function FactorCombinationPanel({ symbol, duration }) {
  const combo = useFactorCombinationData(symbol, duration);
  return <ComboPanelView {...combo} duration={duration} />;
}

function useFactorCombinationData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [rankingState, setRankingState] = useState(initialRankingState);
  const [baseSummary, setBaseSummary] = useState("");
  const [signalState, setSignalState] = useState(initialSignalState);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingSignals, setLoadingSignals] = useState(false);
  const signalAbortRef = useRef(null);
  const loadRanking = useRankingLoader({ duration, setBaseSummary, setRankingState, symbol: normalizedSymbol });

  useRankingEffect(loadRanking);
  useResetSignalState({ setLoadingSignals, setSignalState, signalAbortRef, symbol: normalizedSymbol });
  useSignalAbortCleanup(signalAbortRef);
  const loadSignals = useSignalLoader({
    symbol: normalizedSymbol,
    signalAbortRef,
    setLoadingSignals,
    setSignalState,
  });
  useAutoSignalLoad(loadSignals);
  const refresh = useRefreshHandler({
    loadRanking,
    loadSignals,
    setRankingState,
    setRefreshing,
    symbol: normalizedSymbol,
  });

  return {
    symbol: normalizedSymbol,
    baseSummary,
    rankingState,
    signalState,
    refreshing,
    loadingSignals,
    onLoadSignals: () => void loadSignals(),
    onRefreshCurrent: () => void refresh(duration), onRefreshAll: () => void refresh(undefined),
  };
}

function useRankingEffect(loadRanking) {
  useEffect(() => {
    const ac = new AbortController();
    void loadRanking(ac.signal);
    return () => ac.abort();
  }, [loadRanking]);
}

function useResetSignalState({ setLoadingSignals, setSignalState, signalAbortRef, symbol }) {
  useEffect(() => {
    signalAbortRef.current?.abort();
    setLoadingSignals(false);
    setSignalState(isValidSymbol(symbol) ? initialSignalState() : invalidSignalState());
  }, [setLoadingSignals, setSignalState, signalAbortRef, symbol]);
}

function useSignalAbortCleanup(signalAbortRef) {
  useEffect(() => () => signalAbortRef.current?.abort(), [signalAbortRef]);
}

function useAutoSignalLoad(loadSignals) {
  useEffect(() => {
    void loadSignals();
  }, [loadSignals]);
}

function useRankingLoader({ duration, setBaseSummary, setRankingState, symbol }) {
  return useCallback(async (signal) => {
    if (!isValidSymbol(symbol)) {
      setBaseSummary("");
      setInvalidRanking(setRankingState);
      return;
    }
    setRankingState((state) => ({ ...state, status: "加载多因子组合缓存…" }));
    await loadRankingState({ duration, setBaseSummary, setState: setRankingState, signal, symbol });
  }, [duration, setBaseSummary, setRankingState, symbol]);
}

function useRefreshHandler({ loadRanking, loadSignals, setRankingState, setRefreshing, symbol }) {
  return useCallback(async (targetDuration) => {
    if (!isValidSymbol(symbol)) return setInvalidRanking(setRankingState);
    setRefreshing(true);
    try {
      await requestFactorCombinationRefresh(symbol, targetDuration);
      setRankingState((state) => ({ ...state, status: refreshStatus(targetDuration) }));
      window.setTimeout(() => void loadRanking(new AbortController().signal), REFRESH_RELOAD_DELAY_MS);
      window.setTimeout(() => void loadSignals(), REFRESH_RELOAD_DELAY_MS);
    } catch (error) {
      setRankingState((state) => ({ ...state, status: `刷新失败：${errorMessage(error)}` }));
    } finally {
      setRefreshing(false);
    }
  }, [loadRanking, loadSignals, setRankingState, setRefreshing, symbol]);
}

function useSignalLoader({ symbol, signalAbortRef, setLoadingSignals, setSignalState }) {
  return useCallback(async () => {
    if (!isValidSymbol(symbol)) return setSignalState(invalidSignalState());
    signalAbortRef.current?.abort();
    const ac = new AbortController();
    signalAbortRef.current = ac;
    setLoadingSignals(true);
    setSignalState((state) => ({ ...state, status: "读取数据层周期信号…" }));
    try {
      await loadSignalState(symbol, ac.signal, setSignalState);
    } finally {
      if (!ac.signal.aborted) setLoadingSignals(false);
    }
  }, [setLoadingSignals, setSignalState, signalAbortRef, symbol]);
}

async function loadRankingState({ symbol, duration, signal, setBaseSummary, setState }) {
  try {
    const data = await fetchFactorCombinationRanking(symbol, duration, { signal });
    if (signal.aborted) return;
    const items = Array.isArray(data.ranking) ? data.ranking : [];
    const highWinrateItems = Array.isArray(data.highWinrateRanking) ? data.highWinrateRanking : [];
    setBaseSummary(baseFactorSummary(data));
    setState({
      items,
      highWinrateItems,
      highWinrateSummary: data.highWinrateSummary || null,
      highWinrateStatus: data.highWinrateStatus || null,
      dataCoverage: data.dataCoverage || null,
      status: rankingStatus(data, symbol, duration),
      updatedAt: data.updatedAt,
    });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setBaseSummary("");
    setState(errorRankingState(error));
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
    setState({ items, missing, failures, status: signalStatus(items, missing, failures, data.signalCacheStatus) });
  } catch (error) {
    if (isCanceled(error, signal)) return;
    setState({ items: [], missing: [], failures: [], status: `周期信号失败：${errorMessage(error)}` });
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
        <FactorCombinationSignalGrid
          onSelect={(signal) => setSelectedKey(signalKey(signal))}
          selectedKey={selectedKey}
          signals={signals}
        />
      ) : null}
      {selectedSignal ? <FactorComboPositionsPanel signal={selectedSignal} symbol={props.symbol} /> : null}
      <FactorCombinationRankingTable
        highWinrateRanking={props.rankingState.highWinrateItems}
        highWinrateSummary={props.rankingState.highWinrateSummary}
        ranking={props.rankingState.items}
      />
    </section>
  );
}

function ComboPanelHeader({
  symbol,
  duration,
  baseSummary,
  rankingState,
  signalState,
  refreshing,
  loadingSignals,
  onRefreshCurrent,
  onRefreshAll,
  onLoadSignals,
}) {
  return (
    <div className="factor-combo-head">
      <div>
        <span className="section-kicker">多因子组合</span>
        <h2>{symbol} · {DURATION_LABELS[duration] || duration}</h2>
        <p>{rankingState.status}</p>
        {baseSummary ? <p>{baseSummary}</p> : null}
        <StatusLine data={rankingState.highWinrateStatus} />
        <CoverageLine data={rankingState.dataCoverage} />
        <p>{signalState.status}</p>
      </div>
      <div className="factor-combo-actions">
        <button type="button" disabled={loadingSignals} onClick={onLoadSignals}>
          {loadingSignals ? "加载中…" : "重载周期信号"}
        </button>
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

function rankingStatus(data, symbol, duration) {
  if (data.source === "none") {
    return `暂无组合排名缓存（${symbol} / ${duration}）`;
  }
  if (data.source === "stale_cache") {
    const reason = data.cacheStatus?.reason ? ` · ${cacheReasonText(data.cacheStatus.reason)}` : "";
    const updated = data.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
    return `组合排名旧缓存：${data.total ?? 0} 项（${symbol} / ${duration}${updated}${reason}）`;
  }
  const updated = data.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  return `组合排名：${data.total ?? 0} 项（${symbol} / ${duration}${updated}）`;
}

function baseFactorSummary(data) {
  if (data.source === "none") return "";
  const total = data.baseFactorCount ?? 0;
  const mined = data.minedFactorUsedCount ?? 0;
  const agent = data.agentMinedFactorUsedCount ?? 0;
  const source = data.minedFactorSourceCount ?? 0;
  return `候选基础因子：${total} 个 · 挖掘因子参与 ${mined} 个 · Agent 因子 ${agent} 个 · 挖掘库来源 ${source} 条`;
}

function signalStatus(items, missing, failures, cacheStatus) {
  if (cacheStatus?.reason === "signal_cache_missing") {
    return cacheStatus.message || "周期信号缓存不存在";
  }
  const suffix = missing.length ? ` · 缺少 ${missing.join(", ")}` : "";
  const failed = failures.length ? ` · 失败 ${failures.length}` : "";
  const cache = cacheStatus?.usable === false ? " · 数据层缓存需刷新" : " · 数据层缓存";
  return `周期 Top${TOP_PER_DURATION} 实盘模拟：${items.length} 个${suffix}${failed}${cache}`;
}

function cacheReasonText(reason) {
  const texts = {
    legacy_without_fingerprint: "缺少数据指纹",
    market_data_changed: "行情数据已变化",
    market_data_missing: "行情数据缺失",
  };
  return texts[reason] || reason;
}

function StatusLine({ data }) {
  if (!data) return null;
  const metrics = data.metrics || {};
  const settled = data.settledSampleCount ?? data.sampleCount ?? metrics.sampleCount ?? 0;
  const required = data.requiredSampleCount ?? data.thresholds?.requiredSampleCount ?? data.thresholds?.activeSampleCount ?? "—";
  const winRate = formatPct(data.winRate ?? metrics.winRate, 1);
  return (
    <p>
      High-winrate：{data.status || "unknown"} · settled {settled}/{required} · winRate {winRate} · {data.reason || "no_reason"}
    </p>
  );
}

function CoverageLine({ data }) {
  if (!data?.mainRange) return null;
  const range = data.mainRange;
  const missing = Array.isArray(data.missingFeatureSources) ? data.missingFeatureSources : [];
  const missingNames = [...new Set(missing.map((row) => row.table).filter(Boolean))];
  const missingText = missingNames.length ? ` · 缺失特征源 ${missingNames.join(", ")}` : "";
  return (
    <p>
      数据覆盖：10m K线 {range.rowCount ?? 0} 根 · {formatDate(range.minTimeUtc)} 至 {formatDate(range.maxTimeUtc)}{missingText}
    </p>
  );
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function refreshStatus(duration) {
  return duration ? `已排队重算 ${duration} 多因子组合` : "已排队重算全部周期多因子组合";
}

function setInvalidRanking(setRankingState) {
  setRankingState({ ...initialRankingState(), status: "请输入有效交易对" });
}

function initialRankingState() {
  return {
    items: [],
    highWinrateItems: [],
    highWinrateSummary: null,
    highWinrateStatus: null,
    dataCoverage: null,
    status: "",
    updatedAt: null,
  };
}

function errorRankingState(error) {
  return { ...initialRankingState(), status: `组合排名失败：${errorMessage(error)}` };
}

function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || "unknown_error";
}

function initialSignalState() {
  return { items: [], status: SIGNAL_IDLE_STATUS, missing: [], failures: [] };
}

function invalidSignalState() {
  return { items: [], status: "请输入有效交易对", missing: [], failures: [] };
}

function isValidSymbol(symbol) {
  return symbol.length >= 6;
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}
