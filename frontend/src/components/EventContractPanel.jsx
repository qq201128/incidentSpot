import { useCallback, useEffect, useMemo, useState } from "react";
import { eventDurationMinutesFromWindow } from "../utils/eventDuration";
import { settledExpectedProfitUsdt } from "../utils/eventSettlement";
import AutoStrategyControls from "./AutoStrategyControls";
import EnsembleJudgePanel from "./EnsembleJudgePanel";
import TradeControls from "./TradeControls";
const STORAGE_LIVE_TRADING = "eventContract:liveTradingEnabled";
const STORAGE_PANEL_TAB = "eventContract:rightPanelTab";
const FIXED_PAYOUT_RATE = 0.8;

const PANEL_TABS = /** @type {const} */ (["strategies", "judge", "trade"]);

export default function EventContractPanel({
  symbol,
  chartInterval = "10m",
  currentPrice,
  events,
  onQuickTrade,
  onPredict,
  latestPrediction,
  onClearAllEvents,
}) {
  const [durationMinutes, setDurationMinutes] = useState(() =>
    intervalToTradeMinutes(chartInterval),
  );
  const [amount, setAmount] = useState(5);
  const [prediction, setPrediction] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [predictInfo, setPredictInfo] = useState("");
  const [localOpenPositionPending, setLocalOpenPositionPending] = useState(false);
  const [liveTradingEnabled, setLiveTradingEnabled] = useState(
    () => localStorage.getItem(STORAGE_LIVE_TRADING) === "1",
  );
  const [clearAllLoading, setClearAllLoading] = useState(false);
  const [panelTab, setPanelTab] = useState(() => _initialPanelTab());
  const [strategyReloadKey, setStrategyReloadKey] = useState(0);

  const aiHistorySuccess = useMemo(() => _aiHistorySuccessByStrategy(events, symbol), [events, symbol]);
  const dbHasOpenPosition = useMemo(
    () => events.some((event) => event.symbol === symbol && event.status === "OPEN" && event.orderSide),
    [events, symbol],
  );
  const hasOpenPosition = dbHasOpenPosition || localOpenPositionPending;

  useEffect(() => {
    localStorage.setItem(STORAGE_LIVE_TRADING, liveTradingEnabled ? "1" : "0");
  }, [liveTradingEnabled]);

  useEffect(() => {
    localStorage.setItem(STORAGE_PANEL_TAB, panelTab);
  }, [panelTab]);

  useEffect(() => {
    setDurationMinutes(intervalToTradeMinutes(chartInterval));
  }, [chartInterval]);

  useEffect(() => {
    setPrediction(null);
    setPredictError("");
    setPredictInfo("");
    setPredictLoading(false);
    setLocalOpenPositionPending(false);
  }, [symbol, durationMinutes]);

  useEffect(() => {
    if (dbHasOpenPosition) setLocalOpenPositionPending(false);
  }, [dbHasOpenPosition]);

  useEffect(() => {
    const duration = predictDurationKey(durationMinutes);
    if (latestPrediction?.symbol === symbol && latestPrediction?.duration === duration) {
      setPrediction(latestPrediction);
    }
  }, [durationMinutes, latestPrediction, symbol]);

  const runAiPredictAndTrade = useCallback(async () => {
    if (!onPredict || !onQuickTrade) return "error";
    if (hasOpenPosition) return _blockTrade(setPredictInfo);
    setPredictLoading(true);
    setPredictError("");
    setPredictInfo("");
    try {
      const result = await onPredict(symbol, predictDurationKey(durationMinutes));
      setPrediction(result);
      const blocked = predictionBlockReason(result);
      if (blocked) {
        setPredictInfo(blocked);
        return "blocked";
      }
      await onQuickTrade(ruleQuickTradePayload({
        amount,
        currentPrice,
        durationMinutes,
        liveTradingEnabled,
        result,
        symbol,
      }));
      setLocalOpenPositionPending(true);
      return "placed";
    } catch (err) {
      setPredictError(String(err?.response?.data?.detail || err?.message || "规则计算或下单失败"));
      return "error";
    } finally {
      setPredictLoading(false);
    }
  }, [amount, currentPrice, durationMinutes, hasOpenPosition, liveTradingEnabled, onPredict, onQuickTrade, symbol]);

  async function handleTrade(direction) {
    if (hasOpenPosition) {
      setPredictInfo("已有进行中持仓，等待上一笔结束后再下单");
      return;
    }
    setPredictError("");
    try {
      await onQuickTrade(manualQuickTradePayload({
        amount,
        currentPrice,
        direction,
        durationMinutes,
        liveTradingEnabled,
        symbol,
      }));
      setLocalOpenPositionPending(true);
    } catch (err) {
      setPredictError(String(err?.response?.data?.detail || err?.message || "下单失败"));
    }
  }

  async function handleClearAllEventsClick() {
    if (!onClearAllEvents || clearAllLoading) return;
    setClearAllLoading(true);
    try {
      await onClearAllEvents();
      setLocalOpenPositionPending(false);
    } finally {
      setClearAllLoading(false);
    }
  }

  async function handlePredictClick() {
    if (!onPredict || !onQuickTrade) return;
    await runAiPredictAndTrade();
  }

  return (
    <section className="panel panel-with-tabs">
      <div className="panel-head-tabs">
        <h2 className="panel-title">事件合约</h2>
        <div className="panel-tab-bar" role="tablist" aria-label="事件合约分区">
          <button
            type="button"
            role="tab"
            className={`panel-tab ${panelTab === "strategies" ? "is-active" : ""}`}
            aria-selected={panelTab === "strategies"}
            id="panel-tab-strategies"
            onClick={() => setPanelTab("strategies")}
          >
            自动策略
          </button>
          <button
            type="button"
            role="tab"
            className={`panel-tab ${panelTab === "judge" ? "is-active" : ""}`}
            aria-selected={panelTab === "judge"}
            id="panel-tab-judge"
            onClick={() => setPanelTab("judge")}
          >
            综合裁判
          </button>
          <button
            type="button"
            role="tab"
            className={`panel-tab ${panelTab === "trade" ? "is-active" : ""}`}
            aria-selected={panelTab === "trade"}
            id="panel-tab-trade"
            onClick={() => setPanelTab("trade")}
          >
            交易
          </button>
        </div>
      </div>

      {panelTab === "strategies" && (
        <div className="panel-tab-panel" role="tabpanel" aria-labelledby="panel-tab-strategies">
          <AutomationCard
            amount={amount}
            clearAllLoading={clearAllLoading}
            reloadKey={strategyReloadKey}
            onClearAllEvents={onClearAllEvents}
            onClearAllEventsClick={handleClearAllEventsClick}
            symbol={symbol}
          />
        </div>
      )}
      {panelTab === "judge" && (
        <div
          className="panel-tab-panel panel-tab-panel--judge"
          role="tabpanel"
          aria-labelledby="panel-tab-judge"
        >
          <EnsembleJudgePanel
            symbol={symbol}
            duration={predictDurationKey(durationMinutes)}
            onConfirmed={() => setStrategyReloadKey((value) => value + 1)}
          />
        </div>
      )}
      {panelTab === "trade" && (
        <div
          className="panel-tab-panel panel-tab-panel--trade"
          role="tabpanel"
          aria-labelledby="panel-tab-trade"
        >
          <TradeControls
            symbol={symbol}
            currentPrice={currentPrice}
            durationMinutes={durationMinutes}
            amount={amount}
            prediction={prediction}
            predictLoading={predictLoading}
            predictInfo={predictInfo}
            predictError={predictError}
            aiHistorySuccess={aiHistorySuccess}
            liveTradingEnabled={liveTradingEnabled}
            onLiveTradingChange={setLiveTradingEnabled}
            onAmountChange={setAmount}
            onDurationChange={setDurationMinutes}
            onPredictClick={handlePredictClick}
            onTrade={handleTrade}
          />
        </div>
      )}
    </section>
  );
}

function _initialPanelTab() {
  const raw = localStorage.getItem(STORAGE_PANEL_TAB);
  if (raw === "events") return "trade";
  return PANEL_TABS.includes(/** @type {any} */ (raw)) ? raw : "trade";
}

function AutomationCard({ amount, clearAllLoading, reloadKey, onClearAllEvents, onClearAllEventsClick, symbol }) {
  return (
    <div className="card automation-toggles">
      <p className="toggle-hint trade-mode-hint automation-intro">
        每条策略右侧可单独切换「模拟 / 实盘」；同一策略下各结算周期共用该开关。仅对已点亮的周期自动下单；多策略可并行。
      </p>
      <AutoStrategyControls symbol={symbol} amount={amount} reloadKey={reloadKey} />
      <p className="toggle-hint trade-mode-hint">
        同一策略在不同周期上的持仓相互独立。
      </p>
      <div className="clear-all-row">
        <button
          type="button"
          className="clear-all-events-btn"
          onClick={() => void onClearAllEventsClick()}
          disabled={!onClearAllEvents || clearAllLoading}
        >
          {clearAllLoading ? "清除中..." : "清除全部事件"}
        </button>
        <p className="toggle-hint clear-all-hint">立即删除库中全部事件与关联订单、结算记录并重载列表。</p>
      </div>
    </div>
  );
}

function predictionBlockReason(result) {
  const bestProb = aiBestSideProbability(result.probabilityUp);
  const threshold = aiTradeConfidenceThreshold(result);
  if (result.productionTarget?.passed !== true) return "规则回测未达标，已跳过创建事件与下单";
  if (bestProb !== null && bestProb < threshold) {
    return `最高方向置信仅 ${Math.round(bestProb * 100)}%，已跳过创建事件与下单（需 ≥${Math.round(threshold * 100)}%）`;
  }
  if (result.highWinrateGateEnabled && result.highWinrateGatePassed !== true) {
    return "高胜率门控未通过，已跳过创建事件与下单";
  }
  return _qualityBlockReason(result);
}

function _qualityBlockReason(result) {
  if (result.highWinrateGateEnabled || result.tradeQualityPassed === true) return "";
  const score = Math.round(Number(result.tradeQualityScore || 0) * 100);
  const scoreMin = Math.round(Number(result.tradeQualityScoreMin || 0) * 100);
  return `质量分仅 ${score}%，已跳过创建事件与下单（需 ≥${scoreMin}%）`;
}

function ruleQuickTradePayload(input) {
  const sideUp = input.result.direction === "up";
  const prob = sideUp ? Number(input.result.probabilityUp) : 1 - Number(input.result.probabilityUp);
  return {
    liveTradingEnabled: input.liveTradingEnabled,
    event: ruleEventPayload(input, sideUp, prob),
    order: orderPayload(input.amount, sideUp),
  };
}

function ruleEventPayload(input, sideUp, probForSide) {
  const confPct = (Math.round(probForSide * 1000) / 10).toFixed(1);
  return {
    strategyKey: input.result.strategyKey,
    symbol: input.symbol,
    title: `${input.symbol} 规则${durationLabel(input.durationMinutes)} ${sideUp ? "看涨" : "看跌"} 置信${confPct}%`,
    eventInterval: resolveEventIntervalByDuration(input.durationMinutes),
    ruleType: "ABOVE",
    strikeValue: Number(input.currentPrice || 0),
    upperBound: null,
    endTime: endTimeIso(input.durationMinutes),
    aiProbabilityUp: input.result.probabilityUp,
    aiPredictedDirection: input.result.direction,
    aiQualityScore: input.result.tradeQualityScore,
    aiQualityPassed: input.result.tradeQualityPassed,
    aiHighWinrateGate: input.result.highWinrateGate,
    aiHighWinrateRule: input.result.highWinrateRule,
    aiHighWinratePassed: input.result.highWinrateGatePassed,
    aiHighWinrateValue: input.result.highWinrateGateValue,
  };
}

function manualQuickTradePayload(input) {
  const sideUp = input.direction === "UP";
  const strike = Number(input.currentPrice || 0);
  return {
    liveTradingEnabled: input.liveTradingEnabled,
    event: {
      symbol: input.symbol,
      title: `${input.symbol} ${durationLabel(input.durationMinutes)}后是否高于 ${strike.toFixed(2)}`,
      eventInterval: resolveEventIntervalByDuration(input.durationMinutes),
      ruleType: "ABOVE",
      strikeValue: strike,
      upperBound: null,
      endTime: endTimeIso(input.durationMinutes),
    },
    order: orderPayload(input.amount, sideUp),
  };
}

function orderPayload(amount, sideUp) {
  return { side: sideUp ? "BUY" : "SELL", qty: Number(amount), price: FIXED_PAYOUT_RATE };
}

function _blockTrade(setPredictInfo) {
  setPredictInfo("已有进行中持仓，等待上一笔结束后再规则计算下单"); return "blocked";
}
function _aiHistorySuccessByStrategy(events, symbol) {
  const settled = events.filter((event) => _isSettledAiEvent(event, symbol));
  const overallHits = settled.filter((event) => Number(event.aiPredictionCorrect) === 1).length;
  let overallPnl = 0;
  for (const event of settled) {
    const pnl = settledExpectedProfitUsdt(event);
    if (pnl != null) overallPnl += pnl;
  }
  const overall = {
    total: settled.length,
    hits: overallHits,
    rate: settled.length > 0 ? overallHits / settled.length : null,
    pnlU: overallPnl,
  };
  const UNKNOWN_DURATION = -1;
  const byKey = new Map();
  for (const event of settled) {
    const strategyKey = event.strategyKey || "manual";
    const durationMinutes = eventDurationMinutesFromWindow(event) ?? UNKNOWN_DURATION;
    const key = `${strategyKey}\t${durationMinutes}`;
    let bucket = byKey.get(key);
    if (!bucket) {
      bucket = { total: 0, hits: 0, pnlU: 0 };
      byKey.set(key, bucket);
    }
    bucket.total += 1;
    if (Number(event.aiPredictionCorrect) === 1) bucket.hits += 1;
    const pnl = settledExpectedProfitUsdt(event);
    if (pnl != null) bucket.pnlU += pnl;
  }
  const byStrategy = [...byKey.entries()]
    .map(([compoundKey, { total, hits, pnlU }]) => {
      const tab = compoundKey.lastIndexOf("\t");
      const strategyKey = tab >= 0 ? compoundKey.slice(0, tab) : compoundKey;
      const durStr = tab >= 0 ? compoundKey.slice(tab + 1) : "";
      const durationMinutes = Number(durStr);
      return {
        strategyKey,
        durationMinutes: Number.isFinite(durationMinutes) ? durationMinutes : UNKNOWN_DURATION,
        total,
        hits,
        pnlU,
        rate: total > 0 ? hits / total : null,
      };
    })
    .sort(
      (a, b) =>
        (a.durationMinutes === UNKNOWN_DURATION ? 1e9 : a.durationMinutes) -
          (b.durationMinutes === UNKNOWN_DURATION ? 1e9 : b.durationMinutes) ||
        b.pnlU - a.pnlU ||
        a.strategyKey.localeCompare(b.strategyKey),
    );
  return { overall, byStrategy };
}

function _isSettledAiEvent(event, symbol) {
  return event.symbol === symbol && event.status === "SETTLED" && event.aiPredictedDirection && event.aiPredictionCorrect != null;
}

function aiBestSideProbability(probabilityUp) {
  const p = Number(probabilityUp);
  if (!Number.isFinite(p)) return null;
  return Math.max(p, 1 - p);
}

function aiTradeConfidenceThreshold(prediction) {
  const value = Number(prediction?.tradeConfidenceThreshold);
  if (!Number.isFinite(value)) throw new Error("prediction response missing tradeConfidenceThreshold");
  return value;
}

function predictDurationKey(minutes) {
  if (minutes === 10) return "10m";
  if (minutes === 30) return "30m";
  if (minutes === 60) return "60m";
  if (minutes === 1440) return "1d";
  return "10m";
}

/** 与指数 K 线周期选项一致 */
function intervalToTradeMinutes(interval) {
  switch (interval) {
    case "30m":
      return 30;
    case "60m":
      return 60;
    case "1d":
      return 1440;
    default:
      return 10;
  }
}

function resolveEventIntervalByDuration(minutes) {
  if (minutes <= 10) return "10m";
  if (minutes <= 30) return "30m";
  if (minutes <= 60) return "60m";
  return "1d";
}

function durationLabel(minutes) { return minutes === 1440 ? "1天" : `${minutes}分钟`; }

function endTimeIso(durationMinutes) { return new Date(Date.now() + durationMinutes * 60 * 1000).toISOString(); }
