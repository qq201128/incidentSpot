import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FactorsPage from "./pages/FactorsPage";
import KlineChart from "./components/KlineChart";
import OrderBook from "./components/OrderBook";
import RecentTrades from "./components/RecentTrades";
import EventContractPanel from "./components/EventContractPanel";
import "./components/EventContractPanel.css";
import {
  createQuickTrade,
  deleteAllEvents,
  deleteEventsByStrategy,
  fetchAggTrades,
  fetchEvents,
  fetchIndexKlines,
  fetchLastPrice,
  openIndexKlineSocket,
  settleEvent,
} from "./api/client";
import { strategyLabel } from "./utils/strategyLabels";
import { useLatestPrediction } from "./hooks/useLatestPrediction";
import { mergeKlineCandle, normalizeChartCandle } from "./utils/klineCandles";

export default function App() {
  const [appView, setAppView] = useState("trade");
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setIntervalValue] = useState("10m");
  const [history, setHistory] = useState([]);
  const [latest, setLatest] = useState(null);
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState("就绪");
  const [tickerPrice, setTickerPrice] = useState(0);
  const [lastKlineAt, setLastKlineAt] = useState(0);
  const [priceTick, setPriceTick] = useState(0);
  const [aggTrades, setAggTrades] = useState(null);
  const lastKlineAtRef = useRef(0);
  const chartWsGraceStartRef = useRef(0);
  const { latestPrediction, getFreshPrediction } = useLatestPrediction(symbol, setStatus, interval);

  const chartData = useMemo(() => history.map(normalizeChartCandle), [history]);
  const currentPrice = useMemo(() => {
    const klineClose = latest?.close != null ? Number(latest.close) : null;
    const klineFresh = lastKlineAt > 0 && Date.now() - lastKlineAt < 12_000;
    if (klineClose != null && Number.isFinite(klineClose) && klineFresh) {
      return klineClose;
    }
    if (tickerPrice > 0) return tickerPrice;
    if (klineClose != null && Number.isFinite(klineClose)) return klineClose;
    if (history.length) return Number(history[history.length - 1].close);
    return 0;
  }, [latest, history, tickerPrice, lastKlineAt, priceTick]);

  const chartLatestData = useMemo(() => {
    if (!latest) return null;
    const base = normalizeChartCandle(latest);
    if (latest.isClosed === true) return base;
    const p = currentPrice;
    if (!(p > 0) || !Number.isFinite(p)) return base;
    const h = Number(latest.high);
    const l = Number(latest.low);
    return {
      ...base,
      close: p,
      high: Number.isFinite(h) ? Math.max(h, p) : p,
      low: Number.isFinite(l) ? Math.min(l, p) : p,
    };
  }, [latest, currentPrice]);

  useEffect(() => {
    const id = window.setInterval(() => {
      setPriceTick((n) => n + 1);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    async function load() {
      setStatus("正在加载指数K线...");
      const rows = await fetchIndexKlines(symbol, interval, 500);
      setHistory(rows);
      if (rows.length) setLatest(rows[rows.length - 1]);
      setStatus(`指数K线已加载：${rows.length} 条`);
    }
    load().catch((err) => setStatus(`指数K线加载失败：${err.message}`));
  }, [symbol, interval]);

  useEffect(() => {
    let timer;
    let stopped = false;
    const tick = () => {
      if (stopped) return;
      fetchLastPrice(symbol)
        .then((p) => {
          if (!stopped) setTickerPrice(p);
        })
        .catch(() => {});
    };
    tick();
    timer = window.setInterval(tick, 2000);
    return () => {
      stopped = true;
      if (timer) clearInterval(timer);
    };
  }, [symbol]);

  useEffect(() => {
    let stopped = false;
    const poll = () => {
      fetchAggTrades(symbol, 40)
        .then((rows) => {
          if (!stopped) setAggTrades(Array.isArray(rows) ? rows : []);
        })
        .catch(() => {});
    };
    poll();
    const timer = window.setInterval(poll, 2000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [symbol]);

  /** 指数 WS 无推送时，用 REST indexPriceKlines 刷新图表。 */
  useEffect(() => {
    let timer;
    let stopped = false;
    chartWsGraceStartRef.current = Date.now();
    lastKlineAtRef.current = 0;

    const poll = async () => {
      if (stopped) return;
      const now = Date.now();
      const sinceKline = lastKlineAtRef.current === 0 ? now - chartWsGraceStartRef.current : now - lastKlineAtRef.current;
      const quietMs = lastKlineAtRef.current === 0 ? 6000 : 8000;
      if (sinceKline < quietMs) return;
      try {
        const rows = await fetchIndexKlines(symbol, interval, 500);
        if (stopped || !Array.isArray(rows) || !rows.length) return;
        setHistory(rows);
        setLatest(rows[rows.length - 1]);
      } catch {
        // ignore
      }
    };

    timer = window.setInterval(poll, 5000);
    return () => {
      stopped = true;
      if (timer) clearInterval(timer);
    };
  }, [symbol, interval]);

  useEffect(() => {
    fetchEvents()
      .then(setEvents)
      .catch((err) => setStatus(`事件加载失败：${err.message}`));
  }, []);

  useEffect(() => {
    let timer;
    let stopped = false;

    const tick = async () => {
      if (stopped) return;
      try {
        const rows = await fetchEvents();
        if (!stopped) {
          setEvents(rows);
        }
      } catch {
        // Keep UI resilient; status already reflects realtime channel state.
      }
    };

    timer = window.setInterval(tick, 3000);
    return () => {
      stopped = true;
      if (timer) clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let ws;
    let retryCount = 0;
    let retryTimer;
    let stopped = false;
    setLatest(null);
    setLastKlineAt(0);
    lastKlineAtRef.current = 0;
    chartWsGraceStartRef.current = Date.now();

    const connect = () => {
      if (stopped) return;
      ws = openIndexKlineSocket(symbol, interval, (payload) => {
        setLatest((prev) => mergeKlineCandle(prev, payload));
        const t = Date.now();
        lastKlineAtRef.current = t;
        setLastKlineAt(t);
        setHistory((prev) => {
          if (!prev.length) return [payload];
          const idx = prev.findIndex((x) => x.openTime === payload.openTime);
          if (idx === -1) return [...prev.slice(-499), payload];
          const copy = [...prev];
          copy[idx] = mergeKlineCandle(copy[idx], payload);
          return copy;
        });
      });

      ws.onopen = async () => {
        retryCount = 0;
        setStatus("指数K线实时连接已建立");
        try {
          const rows = await fetchIndexKlines(symbol, interval, 500);
          setHistory(rows);
          if (rows.length) setLatest(rows[rows.length - 1]);
        } catch {
          setStatus("指数K线重连刷新失败");
        }
      };
      ws.onerror = () => setStatus("指数K线实时连接异常");
      ws.onclose = () => {
        if (stopped) return;
        const wait = Math.min(1000 * 2 ** retryCount, 10000);
        retryCount += 1;
        setStatus(`${Math.floor(wait / 1000)} 秒后重连指数K线`);
        retryTimer = setTimeout(connect, wait);
      };
    };

    connect();
    return () => {
      stopped = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (ws) ws.close();
    };
  }, [symbol, interval]);

  const reloadEvents = useCallback(async () => {
    const rows = await fetchEvents();
    setEvents(rows);
  }, []);

  const handleClearAllEvents = useCallback(async () => {
    await deleteAllEvents();
    await reloadEvents();
    setStatus("已清除全部事件");
  }, [reloadEvents]);

  const handleClearStrategyEvents = useCallback(
    async (strategyKey) => {
      await deleteEventsByStrategy(strategyKey);
      await reloadEvents();
      setStatus(`已清除「${strategyLabel(strategyKey)}」策略的事件`);
    },
    [reloadEvents],
  );

  async function handleQuickTrade(payload) {
    const quickTradeResult = await createQuickTrade(payload);
    const simulated = quickTradeResult.simulated || quickTradeResult.externalStatus === "SIMULATED";
    const statusLabel = simulated ? "模拟事件已创建" : "真实事件已创建";
    const externalText = !simulated && quickTradeResult.externalOrderId
      ? ` / Binance #${quickTradeResult.externalOrderId}`
      : "";
    setStatus(
      `${statusLabel}：#${quickTradeResult.eventId}${externalText} @ ${Number(
        quickTradeResult.strikeValue,
      ).toFixed(2)}`,
    );
    // Keep action response fast; refresh list in background.
    void reloadEvents();
  }

  async function handleSettle(eventId) {
    await settleEvent(eventId);
    await reloadEvents();
    setStatus("事件已结算");
  }

  const appNav = (
    <nav className="app-nav" aria-label="主导航">
      <button
        type="button"
        className={`app-nav-link${appView === "trade" ? " app-nav-link-active" : ""}`}
        onClick={() => setAppView("trade")}
        aria-current={appView === "trade" ? "page" : undefined}
      >
        工作台
      </button>
      <button
        type="button"
        className={`app-nav-link${appView === "factors" ? " app-nav-link-active" : ""}`}
        onClick={() => setAppView("factors")}
        aria-current={appView === "factors" ? "page" : undefined}
      >
        因子库
      </button>
    </nav>
  );

  if (appView === "factors") {
    return (
      <>
        {appNav}
        <FactorsPage />
      </>
    );
  }

  return (
    <>
      {appNav}
      <main className="layout">
      <header className="topbar">
        <div>
          <span className="eyebrow">规则事件交易工作台</span>
          <h1>Binance 指数K线与实时规则控制台</h1>
        </div>
        <p className="status-pill">{status}</p>
      </header>
      <div className="dashboard-grid">
        <div className="workspace-column">
          <section className="market">
            <div className="section-head">
              <div>
                <span className="section-kicker">行情</span>
                <h2>指数K线</h2>
              </div>
              <div className="toolbar">
                <label>交易对</label>
                <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
                <label>周期</label>
                <select value={interval} onChange={(e) => setIntervalValue(e.target.value)}>
                  <option value="10m">10分钟</option>
                  <option value="30m">30分钟</option>
                  <option value="60m">60分钟</option>
                  <option value="1d">1天</option>
                </select>
              </div>
            </div>
            <div className="market-body">
              <KlineChart data={chartData} latest={chartLatestData} />
              <aside className="market-orderbook" aria-label="合约盘口与成交">
                <OrderBook symbol={symbol} lastTrade={aggTrades?.[0] ?? null} />
                <RecentTrades symbol={symbol} trades={aggTrades} />
              </aside>
            </div>
          </section>
        </div>

        <div className="trade-column">
          <EventContractPanel
            symbol={symbol}
            chartInterval={interval}
            currentPrice={currentPrice}
            events={events}
            onQuickTrade={handleQuickTrade}
            onSettle={handleSettle}
            onPredict={getFreshPrediction}
            latestPrediction={latestPrediction}
            onClearAllEvents={handleClearAllEvents}
            onClearStrategyEvents={handleClearStrategyEvents}
          />
        </div>
      </div>
    </main>
    </>
  );
}
