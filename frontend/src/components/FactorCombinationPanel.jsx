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
const SIGNAL_IDLE_STATUS = "周期信号未加载";

export default function FactorCombinationPanel({ symbol, duration }) {
  const combo = useFactorCombinationData(symbol, duration);
  return <ComboPanelView {...combo} duration={duration} />;
}

function useFactorCombinationData(symbol, duration) {
  const normalizedSymbol = useMemo(() => symbol.trim().toUpperCase(), [symbol]);
  const [rankingState, setRankingState] = useState({ items: [], status: "", updatedAt: null });
  const [signalState, setSignalState] = useState(initialSignalState);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingSignals, setLoadingSignals] = useState(false);
  const signalAbortRef = useRef(null);
  const loadRanking = useRankingLoader(normalizedSymbol, duration, setRankingState);

  useEffect(() => {
    const ac = new AbortController();
    void loadRanking(ac.signal);
    return () => ac.abort();
  }, [loadRanking]);

  useEffect(() => {
    signalAbortRef.current?.abort();
    setLoadingSignals(false);
    setSignalState(isValidSymbol(normalizedSymbol) ? initialSignalState() : invalidSignalState());
  }, [normalizedSymbol]);

  useEffect(() => () => signalAbortRef.current?.abort(), []);
  const loadSignals = useSignalLoader({
    symbol: normalizedSymbol,
    signalAbortRef,
    setLoadingSignals,
    setSignalState,
  });

  const refresh = useCallback(async (targetDuration) => {
    if (!isValidSymbol(normalizedSymbol)) return setInvalidRanking(setRankingState);
    setRefreshing(true);
    try {
      await requestFactorCombinationRefresh(normalizedSymbol, targetDuration);
      setRankingState((state) => ({ ...state, status: refreshStatus(targetDuration) }));
      window.setTimeout(() => void loadRanking(new AbortController().signal), REFRESH_RELOAD_DELAY_MS);
    } catch (error) {
      setRankingState((state) => ({ ...state, status: `刷新失败：${error.message}` }));
    } finally {
      setRefreshing(false);
    }
  }, [loadRanking, normalizedSymbol]);

  return {
    symbol: normalizedSymbol,
    rankingState,
    signalState,
    refreshing,
    loadingSignals,
    onLoadSignals: () => void loadSignals(),
    onRefreshCurrent: () => void refresh(duration), onRefreshAll: () => void refresh(undefined),
  };
}

function useRankingLoader(symbol, duration, setRankingState) {
  return useCallback(async (signal) => {
    if (!isValidSymbol(symbol)) {
      setInvalidRanking(setRankingState);
      return;
    }
    setRankingState((state) => ({ ...state, status: "加载多因子组合缓存…" }));
    await loadRankingState({ duration, setState: setRankingState, signal, symbol });
  }, [duration, setRankingState, symbol]);
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

async function loadRankingState({ symbol, duration, signal, setState }) {
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
    setState({ items, missing, failures, status: signalStatus(items, missing, failures, data.signalCacheStatus) });
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
        <FactorCombinationSignalGrid
          onSelect={(signal) => setSelectedKey(signalKey(signal))}
          selectedKey={selectedKey}
          signals={signals}
        />
      ) : null}
      {selectedSignal ? <FactorComboPositionsPanel signal={selectedSignal} symbol={props.symbol} /> : null}
      <FactorCombinationRankingTable ranking={props.rankingState.items} />
    </section>
  );
}

function ComboPanelHeader({
  symbol,
  duration,
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
        <p>{signalState.status}</p>
      </div>
      <div className="factor-combo-actions">
        <button type="button" disabled={loadingSignals} onClick={onLoadSignals}>
          加载周期信号
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
  const updated = data.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  return `组合排名：${data.total ?? 0} 项（${symbol} / ${duration}${updated}）`;
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

function refreshStatus(duration) {
  return duration ? `已排队重算 ${duration} 多因子组合` : "已排队重算全部周期多因子组合";
}

function setInvalidRanking(setRankingState) {
  setRankingState({ items: [], status: "请输入有效交易对", updatedAt: null });
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
