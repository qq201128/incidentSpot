import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchAggTrades,
  fetchIndexKlines,
  fetchLastPrice,
  openIndexKlineSocket,
} from "../api/client";
import { mergeKlineCandle, normalizeChartCandle } from "../utils/klineCandles";

const PRICE_POLL_MS = 2000;
const TRADES_POLL_MS = 3000;

export function useTradingMarket(symbol, interval, setStatus) {
  const [history, setHistory] = useState([]);
  const [latest, setLatest] = useState(null);
  const [tickerPrice, setTickerPrice] = useState(0);
  const [lastKlineAt, setLastKlineAt] = useState(0);
  const [priceTick, setPriceTick] = useState(0);
  const [aggTrades, setAggTrades] = useState(null);
  const lastKlineAtRef = useRef(0);
  const chartWsGraceStartRef = useRef(0);

  useMarketClock(setPriceTick);
  useInitialKlines(symbol, interval, setHistory, setLatest, setStatus);
  usePricePolling(symbol, setTickerPrice, setStatus);
  useTradePolling(symbol, setAggTrades, setStatus);
  useRestKlineRefresh(symbol, interval, setHistory, setLatest, setStatus, lastKlineAtRef, chartWsGraceStartRef);
  useIndexKlineSocket(symbol, interval, setHistory, setLatest, setLastKlineAt, setStatus, lastKlineAtRef, chartWsGraceStartRef);

  const chartData = useMemo(() => history.map(normalizeChartCandle), [history]);
  const currentPrice = useCurrentPrice(latest, history, tickerPrice, lastKlineAt, priceTick);
  const chartLatestData = useChartLatest(latest, currentPrice);

  return { aggTrades, chartData, chartLatestData, currentPrice, lastKlineAt };
}

function useMarketClock(setPriceTick) {
  useEffect(() => {
    const id = window.setInterval(() => setPriceTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [setPriceTick]);
}

function useInitialKlines(symbol, interval, setHistory, setLatest, setStatus) {
  useEffect(() => {
    async function load() {
      setStatus("正在加载指数K线...");
      const rows = await fetchIndexKlines(symbol, interval, 500);
      setHistory(rows);
      if (rows.length) setLatest(rows[rows.length - 1]);
      setStatus(`指数K线已加载：${rows.length} 条`);
    }
    load().catch((err) => setStatus(`指数K线加载失败：${err.message}`));
  }, [symbol, interval, setHistory, setLatest, setStatus]);
}

function usePricePolling(symbol, setTickerPrice, setStatus) {
  useEffect(() => {
    let timer;
    let stopped = false;
    const tick = async () => {
      try {
        const p = await fetchLastPrice(symbol);
        if (!stopped) setTickerPrice(p);
      } catch (err) {
        console.error("最新指数价加载失败", err);
        if (!stopped) setStatus(`最新指数价加载失败：${err.message}`);
      } finally {
        if (!stopped) timer = window.setTimeout(tick, PRICE_POLL_MS);
      }
    };
    void tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [symbol, setTickerPrice, setStatus]);
}

function useTradePolling(symbol, setAggTrades, setStatus) {
  useEffect(() => {
    let timer;
    let stopped = false;
    const poll = async () => {
      try {
        const rows = await fetchAggTrades(symbol, 40);
        if (!stopped) setAggTrades(Array.isArray(rows) ? rows : []);
      } catch (err) {
        console.error("近期成交加载失败", err);
        if (!stopped) setStatus(`近期成交加载失败：${err.message}`);
      } finally {
        if (!stopped) timer = window.setTimeout(poll, TRADES_POLL_MS);
      }
    };
    void poll();
    return () => {
      stopped = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [symbol, setAggTrades, setStatus]);
}

function useRestKlineRefresh(symbol, interval, setHistory, setLatest, setStatus, lastKlineAtRef, graceRef) {
  useEffect(() => {
    let timer;
    let stopped = false;
    graceRef.current = Date.now();
    lastKlineAtRef.current = 0;
    const poll = async () => {
      if (stopped || !shouldRefreshKlines(lastKlineAtRef.current, graceRef.current)) return;
      try {
        const rows = await fetchIndexKlines(symbol, interval, 500);
        if (stopped || !Array.isArray(rows) || !rows.length) return;
        setHistory(rows);
        setLatest(rows[rows.length - 1]);
      } catch (err) {
        console.error("指数K线 REST 刷新失败", err);
        if (!stopped) setStatus(`指数K线 REST 刷新失败：${err.message}`);
      }
    };
    timer = window.setInterval(poll, 5000);
    return () => {
      stopped = true;
      if (timer) clearInterval(timer);
    };
  }, [symbol, interval, setHistory, setLatest, setStatus, lastKlineAtRef, graceRef]);
}

function useIndexKlineSocket(symbol, interval, setHistory, setLatest, setLastKlineAt, setStatus, lastKlineAtRef, graceRef) {
  useEffect(() => {
    const state = { retryCount: 0, retryTimer: null, stopped: false, ws: null };
    setLatest(null);
    setLastKlineAt(0);
    lastKlineAtRef.current = 0;
    graceRef.current = Date.now();
    connectIndexSocket(state, { symbol, interval, setHistory, setLatest, setLastKlineAt, setStatus, lastKlineAtRef });
    return () => {
      state.stopped = true;
      if (state.retryTimer) clearTimeout(state.retryTimer);
      if (state.ws) state.ws.close();
    };
  }, [symbol, interval, setHistory, setLatest, setLastKlineAt, setStatus, lastKlineAtRef, graceRef]);
}

function connectIndexSocket(state, deps) {
  if (state.stopped) return;
  try {
    state.ws = openIndexKlineSocket(deps.symbol, deps.interval, (payload) => handleKlinePayload(payload, deps));
  } catch (err) {
    console.error("指数K线 WebSocket 创建失败", err);
    deps.setStatus(`指数K线 WebSocket 创建失败：${err.message}`);
    return;
  }
  state.ws.onopen = () => handleSocketOpen(state, deps);
  state.ws.onerror = (err) => {
    console.error("指数K线实时连接异常", err);
    deps.setStatus("指数K线实时连接异常");
  };
  state.ws.onclose = () => scheduleSocketRetry(state, deps);
}

function handleKlinePayload(payload, deps) {
  deps.setLatest((prev) => mergeKlineCandle(prev, payload));
  const t = Date.now();
  deps.lastKlineAtRef.current = t;
  deps.setLastKlineAt(t);
  deps.setHistory((prev) => mergeHistoryPayload(prev, payload));
}

async function handleSocketOpen(state, deps) {
  state.retryCount = 0;
  deps.setStatus("指数K线实时连接已建立");
  try {
    const rows = await fetchIndexKlines(deps.symbol, deps.interval, 500);
    deps.setHistory(rows);
    if (rows.length) deps.setLatest(rows[rows.length - 1]);
  } catch (err) {
    console.error("指数K线重连刷新失败", err);
    deps.setStatus(`指数K线重连刷新失败：${err.message}`);
  }
}

function scheduleSocketRetry(state, deps) {
  if (state.stopped) return;
  const wait = Math.min(1000 * 2 ** state.retryCount, 10000);
  state.retryCount += 1;
  deps.setStatus(`${Math.floor(wait / 1000)} 秒后重连指数K线`);
  state.retryTimer = setTimeout(() => connectIndexSocket(state, deps), wait);
}

function mergeHistoryPayload(prev, payload) {
  if (!prev.length) return [payload];
  const idx = prev.findIndex((x) => x.openTime === payload.openTime);
  if (idx === -1) return [...prev.slice(-499), payload];
  const copy = [...prev];
  copy[idx] = mergeKlineCandle(copy[idx], payload);
  return copy;
}

function shouldRefreshKlines(lastKlineAt, graceStartedAt) {
  const now = Date.now();
  const sinceKline = lastKlineAt === 0 ? now - graceStartedAt : now - lastKlineAt;
  return sinceKline >= (lastKlineAt === 0 ? 6000 : 8000);
}

function useCurrentPrice(latest, history, tickerPrice, lastKlineAt, priceTick) {
  return useMemo(() => {
    const klineClose = latest?.close != null ? Number(latest.close) : null;
    const klineFresh = lastKlineAt > 0 && Date.now() - lastKlineAt < 12_000;
    if (klineClose != null && Number.isFinite(klineClose) && klineFresh) return klineClose;
    if (tickerPrice > 0) return tickerPrice;
    if (klineClose != null && Number.isFinite(klineClose)) return klineClose;
    if (history.length) return Number(history[history.length - 1].close);
    return 0;
  }, [latest, history, tickerPrice, lastKlineAt, priceTick]);
}

function useChartLatest(latest, currentPrice) {
  return useMemo(() => {
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
}
