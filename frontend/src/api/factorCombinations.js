import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;
const REFRESH_TIMEOUT_MS = 15_000;

export async function fetchFactorCombinationRanking(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factors/combinations/ranking`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function fetchFactorCombinationSignals(symbol, limit, options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factors/combinations/signals`, {
    params: { symbol, limit },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function fetchFactorCombinationPositions(symbol, duration, factorName, options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factors/combinations/positions`, {
    params: { symbol, duration, factorName, limit: options.limit ?? 80 },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function requestFactorCombinationRefresh(symbol, duration, config = {}) {
  const params = { symbol, ..._configParams(config) };
  if (duration) {
    params.duration = duration;
  }
  const { data } = await axios.post(`${API_BASE_URL}/api/factors/combinations/refresh`, null, {
    params,
    timeout: REFRESH_TIMEOUT_MS,
  });
  return data;
}

function _configParams(config) {
  const params = {};
  if (config.baseFactorLimit) {
    params.baseFactorLimit = config.baseFactorLimit;
  }
  if (config.comboSizes) {
    params.comboSizes = config.comboSizes;
  }
  if (config.resultLimit) {
    params.resultLimit = config.resultLimit;
  }
  return params;
}
