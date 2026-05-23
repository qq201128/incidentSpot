import axios from "axios";

/**
 * 浏览器里请求的 API 根地址。
 * - `npm run dev`：默认走 Vite 同源代理（vite.config.js），避免跨域；勿再依赖 VITE_API_BASE_URL 指公网 :8000。
 * - 开发时要直连远端后端：设 VITE_DIRECT_API=1，并配置 VITE_API_BASE_URL。
 * - `npm run build`：使用 .env.production 或构建环境的 VITE_API_BASE_URL。
 */
const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const MARKET_REQUEST_TIMEOUT_MS = 12_000;
const LOCAL_REQUEST_TIMEOUT_MS = 8_000;

const DIRECT_IN_DEV =
  import.meta.env.VITE_DIRECT_API === "1" ||
  import.meta.env.VITE_DIRECT_API === "true";

const SAME_ORIGIN_API =
  import.meta.env.VITE_SAME_ORIGIN_API === "1" ||
  import.meta.env.VITE_SAME_ORIGIN_API === "true" ||
  (import.meta.env.DEV && !DIRECT_IN_DEV);

function httpOriginToWsOrigin(httpUrl) {
  try {
    const u = new URL(httpUrl);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.origin;
  } catch (err) {
    throw new Error(`无法从 API 地址解析 WebSocket 地址：${httpUrl}`, { cause: err });
  }
}

function wsBaseFromBrowserOrigin() {
  if (typeof window === "undefined") {
    return "ws://127.0.0.1:8000";
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}

const BASE_URL = SAME_ORIGIN_API
  ? ""
  : import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_API_BASE;

const WS_BASE_URL = SAME_ORIGIN_API
  ? wsBaseFromBrowserOrigin()
  : import.meta.env.VITE_WS_BASE_URL?.trim() || httpOriginToWsOrigin(BASE_URL || DEFAULT_API_BASE);
export const API_BASE_URL = BASE_URL;
const ALLOWED_INTERVALS = new Set(["10m", "30m", "60m", "1d"]);

function normalizeInterval(interval) {
  const value = typeof interval === "string" ? interval : String(interval || "");
  return ALLOWED_INTERVALS.has(value) ? value : "30m";
}

export async function fetchHistory(symbol, interval, limit = 500, { live = false } = {}) {
  const safeInterval = normalizeInterval(interval);
  const params = { symbol, interval: safeInterval, limit };
  if (live) params.live = true;
  const { data } = await axios.get(`${BASE_URL}/api/klines`, { params });
  return data;
}

/** 官方指数价（与 /api/index-price、事件口径一致）。 */
export async function fetchLatestPrice(symbol) {
  const { data } = await axios.get(`${BASE_URL}/api/last-price`, {
    params: { symbol },
    timeout: MARKET_REQUEST_TIMEOUT_MS,
  });
  const p = Number(data?.indexPrice);
  if (!Number.isFinite(p) || p <= 0) {
    throw new Error("index price unavailable");
  }
  return p;
}

/** 轮询用：同 fetchLatestPrice，指数价 premiumIndex。 */
export async function fetchLastPrice(symbol) {
  return fetchLatestPrice(symbol);
}

/** Binance index-price K线（/fapi/v1/indexPriceKlines），与本地合约价 K 线分离、不写库。 */
export async function fetchIndexKlines(symbol, interval, limit = 500) {
  const safeInterval = normalizeInterval(interval);
  const { data } = await axios.get(`${BASE_URL}/api/index-klines`, {
    params: { symbol, interval: safeInterval, limit },
    timeout: MARKET_REQUEST_TIMEOUT_MS,
  });
  return data;
}

/** 官方指数价 + 标记价（/fapi/v1/premiumIndex）。 */
export async function fetchIndexPrice(symbol) {
  const { data } = await axios.get(`${BASE_URL}/api/index-price`, {
    params: { symbol },
  });
  return data;
}

/** USD-M 订单簿档位（REST /fapi/v1/depth 经后端代理）。 */
export async function fetchOrderbookDepth(symbol, limit = 20) {
  const { data } = await axios.get(`${BASE_URL}/api/depth`, {
    params: { symbol, limit },
    timeout: MARKET_REQUEST_TIMEOUT_MS,
  });
  return data;
}

/** USD-M 聚合成交（最新在前），用于最新成交面板 */
export async function fetchAggTrades(symbol, limit = 40) {
  const { data } = await axios.get(`${BASE_URL}/api/agg-trades`, {
    params: { symbol, limit },
    timeout: MARKET_REQUEST_TIMEOUT_MS,
  });
  return data;
}

export async function createEvent(payload) {
  const { data } = await axios.post(`${BASE_URL}/api/events`, payload);
  return data;
}

export async function createOrder(eventId, payload) {
  const { data } = await axios.post(`${BASE_URL}/api/events/${eventId}/orders`, payload);
  return data;
}

export async function createQuickTrade(payload) {
  const { data } = await axios.post(`${BASE_URL}/api/events/quick-trade`, payload);
  return data;
}

export async function settleEvent(eventId) {
  const { data } = await axios.post(`${BASE_URL}/api/events/${eventId}/settle`);
  return data;
}

export async function fetchEventsPage({ symbol, page = 1, pageSize = 8, view = "events" } = {}) {
  const { data } = await axios.get(`${BASE_URL}/api/events`, {
    params: { symbol, page, pageSize, view },
    timeout: LOCAL_REQUEST_TIMEOUT_MS,
  });
  return data;
}

export async function deleteAllEvents() {
  const { data } = await axios.delete(`${BASE_URL}/api/events`);
  return data;
}

/** @param {string} strategyKey */
export async function deleteEventsByStrategy(strategyKey) {
  const { data } = await axios.delete(`${BASE_URL}/api/events`, {
    params: { strategyKey },
  });
  return data;
}

export async function updateAutoTradeSettings(payload) {
  const { data } = await axios.put(`${BASE_URL}/api/auto-trade/settings`, payload);
  return data;
}

export async function fetchAutoTradeStrategies() {
  const { data } = await axios.get(`${BASE_URL}/api/auto-trade/strategies`);
  return data;
}

export async function updateAutoTradeStrategy(strategyKey, payload) {
  const { data } = await axios.put(
    `${BASE_URL}/api/auto-trade/strategies/${encodeURIComponent(strategyKey)}`,
    payload,
  );
  return data;
}

export async function fetchEnsembleStatus(symbol, duration = "10m") {
  const { data } = await axios.get(`${BASE_URL}/api/ensemble/status`, {
    params: { symbol, duration },
  });
  return data;
}

export async function fetchEnsembleRanking(symbol, duration = "10m") {
  const { data } = await axios.get(`${BASE_URL}/api/ensemble/ranking`, {
    params: { symbol, duration },
  });
  return data;
}

export async function refreshEnsemble(symbol, duration = "10m") {
  const { data } = await axios.post(`${BASE_URL}/api/ensemble/refresh`, null, {
    params: { symbol, duration },
  });
  return data;
}

export async function confirmEnsembleStage(symbol, duration, stage) {
  const { data } = await axios.post(`${BASE_URL}/api/ensemble/confirm-stage`, {
    symbol,
    duration,
    stage,
  });
  return data;
}

export async function predictDirection(symbol, duration = "10m", limit = 2000, strategyKey) {
  const params = { symbol, duration, limit };
  if (strategyKey) params.strategyKey = strategyKey;
  const { data } = await axios.post(`${BASE_URL}/api/predict`, null, {
    params,
  });
  return data;
}

/** 因子库列表（分页、搜索、单因子/组合因子切换）。 */
export async function fetchFactorsList({
  category,
  kind = "single",
  q,
  page = 1,
  pageSize = 20,
} = {}) {
  const params = { kind, page, page_size: pageSize };
  if (category) params.category = category;
  if (q) params.q = q;
  const { data } = await axios.get(`${BASE_URL}/api/factors/list`, { params });
  return data;
}

/** 单个因子元数据（可选 symbol/duration 以附带排名缓存指标）。 */
export async function fetchFactorDetail(factorName, symbol, duration) {
  const params = {};
  if (symbol) params.symbol = symbol;
  if (duration) params.duration = duration;
  const { data } = await axios.get(
    `${BASE_URL}/api/factors/detail/${encodeURIComponent(factorName)}`,
    { params },
  );
  return data;
}

/** 单因子回测指标。 */
export async function fetchFactorBacktest(factorName, symbol, duration = "10m") {
  const { data } = await axios.get(
    `${BASE_URL}/api/factors/backtest/${encodeURIComponent(factorName)}`,
    {
      params: { symbol, duration },
    },
  );
  return data;
}

/** 读取后台缓存的全因子 IR 排名（轻量）；无缓存时 ranking 为空。 */
export async function fetchFactorRanking(symbol, duration = "10m", category, options = {}) {
  const params = { symbol, duration };
  if (category) params.category = category;
  const { data } = await axios.get(`${BASE_URL}/api/factors/ranking`, {
    params,
    signal: options.signal,
    timeout: options.timeoutMs ?? 30_000,
  });
  return data;
}

/** 排队后台重算排名并写入缓存（不阻塞完成）。 */
export async function requestFactorRankingRefresh(symbol, duration) {
  const params = { symbol };
  if (duration) params.duration = duration;
  const { data } = await axios.post(`${BASE_URL}/api/factors/ranking/refresh`, null, {
    params,
    timeout: 15_000,
  });
  return data;
}

export async function fetchLatestPrediction(symbol, duration = "10m", strategyKey) {
  const params = { symbol, duration };
  if (strategyKey) params.strategyKey = strategyKey;
  const { data } = await axios.get(`${BASE_URL}/api/predict/latest`, {
    params,
  });
  return data;
}

export function openPredictionSocket(symbol, duration, onMessage, strategyKey) {
  const strategyParam = strategyKey ? `&strategyKey=${encodeURIComponent(strategyKey)}` : "";
  const ws = new WebSocket(
    `${WS_BASE_URL}/ws/predictions?symbol=${symbol.toLowerCase()}&duration=${duration}${strategyParam}`
  );

  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    onMessage(payload);
  };

  return ws;
}

export function openKlineSocket(symbol, interval, onMessage) {
  const safeInterval = normalizeInterval(interval);
  const ws = new WebSocket(
    `${WS_BASE_URL}/ws/klines?symbol=${symbol.toLowerCase()}&interval=${safeInterval}`
  );

  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "kline") {
      onMessage(payload.data);
    }
  };

  return ws;
}

/** 指数价 K 线 WebSocket（后端用 Binance markPrice@1s 的 index price 合成）。 */
export function openIndexKlineSocket(symbol, interval, onMessage) {
  const safeInterval = normalizeInterval(interval);
  const ws = new WebSocket(
    `${WS_BASE_URL}/ws/index-klines?symbol=${symbol.toLowerCase()}&interval=${safeInterval}`
  );

  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "kline") {
      onMessage(payload.data);
    }
  };

  return ws;
}
