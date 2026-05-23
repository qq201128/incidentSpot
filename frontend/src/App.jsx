import { useCallback, useEffect, useState } from "react";
import { fetchWorkbenchSummary } from "./api/workbenchClient";
import {
  createQuickTrade,
  deleteAllEvents,
  deleteEventsByStrategy,
  settleEvent,
} from "./api/client";
import AppNavigation from "./components/AppNavigation";
import FactorLearningPage from "./pages/FactorLearningPage";
import FactorsPage from "./pages/FactorsPage";
import TradingWorkbench from "./components/TradingWorkbench";
import "./components/EventContractPanel.css";
import { strategyLabel } from "./utils/strategyLabels";
import { useLatestPrediction } from "./hooks/useLatestPrediction";
import { useTradingMarket } from "./hooks/useTradingMarket";

const EVENTS_POLL_MS = 5000;

export default function App() {
  const [appView, setAppView] = useState("trade");
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setIntervalValue] = useState("10m");
  const [status, setStatus] = useState("就绪");
  const [summary, setSummary] = useState(null);
  const [summaryLatencyMs, setSummaryLatencyMs] = useState(null);
  const [recordsReloadKey, setRecordsReloadKey] = useState(0);
  const market = useTradingMarket(symbol, interval, setStatus);
  const { latestPrediction, getFreshPrediction } = useLatestPrediction(symbol, setStatus, interval);
  const reloadWorkbench = useReloadWorkbench(
    symbol,
    interval,
    setSummary,
    setSummaryLatencyMs,
    setStatus,
    setRecordsReloadKey,
  );

  useEffect(() => {
    let timer;
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      await reloadWorkbench();
      if (!stopped) timer = window.setTimeout(tick, EVENTS_POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [reloadWorkbench]);

  const handleClearAllEvents = useCallback(async () => {
    await deleteAllEvents();
    await reloadWorkbench({ rethrow: true });
    setStatus("已清除全部事件");
  }, [reloadWorkbench]);

  const handleClearStrategyEvents = useCallback(
    async (strategyKey) => {
      await deleteEventsByStrategy(strategyKey);
      await reloadWorkbench({ rethrow: true });
      setStatus(`已清除「${strategyLabel(strategyKey)}」执行事件`);
    },
    [reloadWorkbench],
  );

  const handleQuickTrade = useCallback(
    async (payload) => {
      const result = await createQuickTrade(payload);
      setStatus(quickTradeStatus(result));
      void reloadWorkbench();
    },
    [reloadWorkbench],
  );

  const handleSettle = useCallback(
    async (eventId) => {
      await settleEvent(eventId);
      await reloadWorkbench({ rethrow: true });
      setStatus("事件已结算");
    },
    [reloadWorkbench],
  );

  if (appView === "factors") {
    return pageFrame(appView, setAppView, <FactorsPage />);
  }
  if (appView === "learning") {
    return pageFrame(appView, setAppView, <FactorLearningPage />);
  }
  return pageFrame(
    appView,
    setAppView,
    <TradingWorkbench
      {...market}
      interval={interval}
      latestPrediction={latestPrediction}
      onClearAllEvents={handleClearAllEvents}
      onClearStrategyEvents={handleClearStrategyEvents}
      onIntervalChange={setIntervalValue}
      onPredict={getFreshPrediction}
      onQuickTrade={handleQuickTrade}
      onSettle={handleSettle}
      onSymbolChange={setSymbol}
      recordsReloadKey={recordsReloadKey}
      status={status}
      summary={summary}
      summaryLatencyMs={summaryLatencyMs}
      symbol={symbol}
    />,
  );
}

function useReloadWorkbench(symbol, interval, setSummary, setLatency, setStatus, setRecordsReloadKey) {
  return useCallback(async (options = {}) => {
    try {
      const { data, latencyMs } = await fetchWorkbenchSummary(symbol, interval);
      setSummary(data);
      setLatency(latencyMs);
      setRecordsReloadKey((value) => value + 1);
    } catch (err) {
      console.error("工作台摘要加载失败", err);
      setStatus(`工作台摘要加载失败：${err.message}`);
      if (options.rethrow) throw err;
    }
  }, [symbol, interval, setSummary, setLatency, setStatus, setRecordsReloadKey]);
}

function pageFrame(appView, setAppView, children) {
  return (
    <>
      <AppNavigation appView={appView} onViewChange={setAppView} />
      {children}
    </>
  );
}

function quickTradeStatus(result) {
  const simulated = result.simulated || result.externalStatus === "SIMULATED";
  const label = simulated ? "模拟事件已创建（未调用 Binance）" : "真实事件已创建";
  const external = !simulated && result.externalOrderId ? ` / Binance #${result.externalOrderId}` : "";
  return `${label}：#${result.eventId}${external} @ ${Number(result.strikeValue).toFixed(2)}`;
}
