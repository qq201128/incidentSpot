import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function fetchFactorPageOverview(symbol, duration = "10m", category, options = {}) {
  const params = { symbol, duration };
  if (category) params.category = category;
  const { data } = await axios.get(`${BASE_URL}/api/factors/overview`, {
    params,
    signal: options.signal,
    timeout: options.timeoutMs ?? 30_000,
  });
  return data;
}

export async function fetchFactorPeriodScores(factorName, symbol, options = {}) {
  const { data } = await axios.get(
    `${BASE_URL}/api/factors/detail/${encodeURIComponent(factorName)}/scores`,
    {
      params: { symbol },
      signal: options.signal,
      timeout: options.timeoutMs ?? 30_000,
    },
  );
  return data;
}

export async function fetchFactorPageBundle(
  symbol,
  duration = "10m",
  { category, kind = "single", q, page = 1, pageSize = 20 } = {},
  options = {},
) {
  const params = { symbol, duration, kind, page, page_size: pageSize };
  if (category) params.category = category;
  if (q) params.q = q;
  const { data } = await axios.get(`${BASE_URL}/api/factors/page`, {
    params,
    signal: options.signal,
    timeout: options.timeoutMs ?? 45_000,
  });
  return data;
}

export async function fetchFactorAlerts(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${BASE_URL}/api/factors/alerts`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? 20_000,
  });
  return data;
}
