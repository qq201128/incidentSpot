import { useEffect, useState } from "react";
import { DRAWING_TOOLS, useKlineChartUi } from "../hooks/useKlineChartUi";
import { KLINE_INTERVAL_OPTIONS } from "../utils/klineIntervals";
import { durationKeyFromChartInterval } from "../utils/tradeDuration";
import EventContractPanel from "./EventContractPanel";
import EventRecordsTable from "./EventRecordsTable";
import KlineChart from "./KlineChart";
import OrderBook from "./OrderBook";
import RecentTrades from "./RecentTrades";
import WorkbenchStatusBar from "./WorkbenchStatusBar";

export default function TradingWorkbench(props) {
  const chartUi = useKlineChartUi();
  const [ensembleReloadKey, setEnsembleReloadKey] = useState(0);
  const [recordsPage, setRecordsPage] = useState(1);
  const [compactRecordsPage, setCompactRecordsPage] = useState(1);

  useEffect(() => {
    setRecordsPage(1);
    setCompactRecordsPage(1);
  }, [props.symbol]);

  return (
    <main className="terminal-layout">
      <WorkbenchHeader status={props.status} summary={props.summary} />
      <div className="terminal-grid">
        <section className="market terminal-card">
          <MarketToolbar
            activePanel={chartUi.activePanel}
            chartUi={chartUi}
            interval={props.interval}
            onIntervalChange={props.onIntervalChange}
            onSymbolChange={props.onSymbolChange}
            symbol={props.symbol}
          />
          <div className="chart-stage">
            <ChartToolRail
              activeTool={chartUi.drawingTool}
              drawingMode={chartUi.drawingMode}
              drawingsLocked={chartUi.drawingsLocked}
              onSelectTool={(toolId) => chartUi.selectDrawingTool(toolId)}
            />
            <KlineChart
              clearDrawingsToken={chartUi.clearDrawingsToken}
              data={props.chartData}
              drawingTool={chartUi.drawingMode ? chartUi.drawingTool : "cursor"}
              drawingsLocked={chartUi.drawingsLocked}
              fitToken={chartUi.fitToken}
              indicators={chartUi.indicators}
              latest={props.chartLatestData}
              settings={chartUi.settings}
            />
          </div>
          <ChartFooter
            chartUi={chartUi}
            currentPrice={props.currentPrice}
            interval={props.interval}
            latencyMs={props.summaryLatencyMs}
            onIntervalChange={props.onIntervalChange}
          />
        </section>
        <aside className="market-orderbook" aria-label="合约盘口与成交">
          <OrderBook symbol={props.symbol} lastTrade={props.aggTrades?.[0] ?? null} />
          <RecentTrades symbol={props.symbol} trades={props.aggTrades} />
        </aside>
        <section className="trade-column terminal-card">
          <EventContractPanel
            symbol={props.symbol}
            chartInterval={props.interval}
            currentPrice={props.currentPrice}
            hasOpenPosition={Boolean(props.summary?.hasOpenPosition)}
            onQuickTrade={props.onQuickTrade}
            onPredict={props.onPredict}
            latestPrediction={props.latestPrediction}
            onClearAllEvents={props.onClearAllEvents}
            onEnsembleRefreshed={() => setEnsembleReloadKey((value) => value + 1)}
          />
        </section>
        <div className="records-column">
          <EventRecordsTable
            symbol={props.symbol}
            page={recordsPage}
            onPageChange={setRecordsPage}
            reloadKey={props.recordsReloadKey}
            ensembleDuration={durationKeyFromChartInterval(props.interval)}
            ensembleReloadKey={ensembleReloadKey}
          />
        </div>
        <div className="records-side">
          <EventRecordsTable
            symbol={props.symbol}
            compact
            page={compactRecordsPage}
            onPageChange={setCompactRecordsPage}
            reloadKey={props.recordsReloadKey}
            ensembleReloadKey={ensembleReloadKey}
          />
        </div>
      </div>
      <WorkbenchStatusBar latencyMs={props.summaryLatencyMs} summary={props.summary} />
    </main>
  );
}

function WorkbenchHeader({ status, summary }) {
  const [open, setOpen] = useState(false);
  const eventCount = summary?.eventTotal;
  const openCount = summary?.eventCounts?.OPEN;
  return (
    <header className="topbar">
      <div>
        <span className="eyebrow">事件合约方向预测工作台</span>
        <h1>Binance 指数事件合约控制台</h1>
      </div>
      <div className="status-pill-wrap">
        <div className="status-pill">
          <span className="status-dot" />
          <span>运行状态 · {status || "等待系统状态"}</span>
          <button type="button" className={open ? "is-active" : ""} onClick={() => setOpen((value) => !value)}>
            状态详情
          </button>
        </div>
        {open ? (
          <div className="status-detail-popover" role="status">
            <p>当前状态：{status || "—"}</p>
            <p>事件总数：{eventCount ?? "—"}</p>
            <p>进行中：{openCount ?? "—"}</p>
            <p>交易对：{summary?.symbol ?? "—"}</p>
            <p>规则周期：{summary?.duration ?? summary?.interval ?? "—"}</p>
          </div>
        ) : null}
      </div>
    </header>
  );
}

function MarketToolbar({ activePanel, chartUi, interval, onIntervalChange, onSymbolChange, symbol }) {
  return (
    <div className="section-head">
      <h2>指数K线</h2>
      <div className="toolbar">
        <label>
          交易对
          <select value={symbol} onChange={(e) => onSymbolChange(e.target.value)}>
            <option value="BTCUSDT">BTCUSDT</option>
            <option value="ETHUSDT">ETHUSDT</option>
          </select>
        </label>
        <label>
          周期
          <select value={interval} onChange={(e) => onIntervalChange(e.target.value)}>
            {KLINE_INTERVAL_OPTIONS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          数据源
          <select value="Binance Index" readOnly>
            <option>Binance Index</option>
          </select>
        </label>
        <div className="toolbar-chart-actions">
          <button
            type="button"
            className={activePanel === "indicators" ? "is-active" : ""}
            onClick={() => chartUi.togglePanel("indicators")}
          >
            指标
          </button>
          <button
            type="button"
            className={chartUi.drawingMode ? "is-active" : ""}
            onClick={chartUi.toggleDrawingMode}
          >
            画线
          </button>
          <button
            type="button"
            className={activePanel === "settings" ? "is-active" : ""}
            onClick={() => chartUi.togglePanel("settings")}
          >
            设置
          </button>
        </div>
      </div>
      {activePanel === "indicators" ? (
        <ChartIndicatorPanel indicators={chartUi.indicators} onToggle={chartUi.toggleIndicator} />
      ) : null}
      {activePanel === "settings" ? (
        <ChartSettingsPanel chartUi={chartUi} />
      ) : null}
    </div>
  );
}

function ChartIndicatorPanel({ indicators, onToggle }) {
  return (
    <div className="chart-panel chart-panel-indicators" role="region" aria-label="指标设置">
      <label>
        <input type="checkbox" checked={indicators.ma7} onChange={() => onToggle("ma7")} />
        MA7
      </label>
      <label>
        <input type="checkbox" checked={indicators.ma20} onChange={() => onToggle("ma20")} />
        MA20
      </label>
      <label>
        <input type="checkbox" checked={indicators.ma60} onChange={() => onToggle("ma60")} />
        MA60
      </label>
      <p className="chart-panel-hint">勾选后立即叠加到 K 线主图</p>
    </div>
  );
}

function ChartSettingsPanel({ chartUi }) {
  const { settings } = chartUi;
  return (
    <div className="chart-panel chart-panel-settings" role="region" aria-label="图表设置">
      <label>
        <input
          type="checkbox"
          checked={settings.showGrid !== false}
          onChange={(event) => chartUi.patchSettings({ showGrid: event.target.checked })}
        />
        显示网格
      </label>
      <label>
        <input
          type="checkbox"
          checked={Boolean(settings.crosshairMagnet)}
          onChange={(event) => chartUi.patchSettings({ crosshairMagnet: event.target.checked })}
        />
        十字线吸附 K 线
      </label>
      <label>
        <input
          type="checkbox"
          checked={Boolean(settings.autoScroll)}
          onChange={(event) => chartUi.patchSettings({ autoScroll: event.target.checked })}
        />
        新 K 线自动滚到最新
      </label>
      <button type="button" className="chart-panel-action" onClick={chartUi.requestFit}>
        适应窗口
      </button>
    </div>
  );
}

function ChartToolRail({ activeTool, drawingMode, drawingsLocked, onSelectTool }) {
  return (
    <div className={`chart-toolrail${drawingMode ? " is-drawing-active" : ""}`} aria-label="画图工具">
      {DRAWING_TOOLS.map((tool) => (
        <button
          key={tool.id}
          type="button"
          className={`${activeTool === tool.id ? "is-active" : ""}${tool.id === "lock" && drawingsLocked ? " is-locked" : ""}`}
          title={tool.label}
          aria-label={tool.label}
          onClick={() => onSelectTool(tool.id)}
        >
          {tool.icon}
        </button>
      ))}
    </div>
  );
}

function ChartFooter({ chartUi, currentPrice, interval, latencyMs, onIntervalChange }) {
  const latency = Number.isFinite(latencyMs) ? `${Math.round(latencyMs)}ms` : "—";
  const clockLabel = useClockLabel();
  const { settings } = chartUi;

  return (
    <>
      <div className="chart-periods">
        {KLINE_INTERVAL_OPTIONS.map((item) => (
          <button
            key={item.value}
            type="button"
            className={interval === item.value ? "is-active" : ""}
            onClick={() => onIntervalChange(item.value)}
          >
            {item.label}
          </button>
        ))}
        <span>{clockLabel}</span>
        <button
          type="button"
          className={settings.priceScaleMode === "percent" ? "is-active" : ""}
          title="百分比坐标"
          onClick={() => chartUi.togglePriceScaleMode("percent")}
        >
          %
        </button>
        <button
          type="button"
          className={settings.priceScaleMode === "log" ? "is-active" : ""}
          title="对数坐标"
          onClick={() => chartUi.togglePriceScaleMode("log")}
        >
          log
        </button>
        <button
          type="button"
          className={settings.autoScroll ? "is-active" : ""}
          title="自动跟随最新 K 线"
          onClick={() => chartUi.patchSettings({ autoScroll: !settings.autoScroll })}
        >
          自动
        </button>
      </div>
      <div className="market-metrics">
        <div>
          指数价: <strong>{formatPrice(currentPrice)}</strong>
        </div>
        <div>
          入场价口径: <strong>Binance Index</strong>
        </div>
        <div>
          K线延迟: <strong className="latency-hot">{latency}</strong>
        </div>
        <div>
          <strong className="value-down">失败将直接暴露</strong>
        </div>
      </div>
    </>
  );
}

function panelProps(props) {
  return {
    chartInterval: props.interval,
    currentPrice: props.currentPrice,
    hasOpenPosition: Boolean(props.summary?.hasOpenPosition),
    latestPrediction: props.latestPrediction,
    onClearAllEvents: props.onClearAllEvents,
    onPredict: props.onPredict,
    onQuickTrade: props.onQuickTrade,
    symbol: props.symbol,
  };
}

function formatPrice(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—";
}

function useClockLabel() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return `${now.toLocaleTimeString("zh-CN", { hour12: false })} (UTC+8)`;
}
