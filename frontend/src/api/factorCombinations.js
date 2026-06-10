import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;
const PAPER_LIVE_CANDIDATES_TIMEOUT_MS = 60_000;
const REFRESH_TIMEOUT_MS = 15_000;
const DAILY_LOOP_TIMEOUT_MS = 120_000;

export async function fetchFactorCombinationRanking(symbol, duration = "10m", options = {}) {
  const params = { symbol, duration };
  if (options.q) params.q = options.q;
  if (options.page) params.page = options.page;
  if (options.pageSize) params.pageSize = options.pageSize;
  const { data } = await axios.get(`${API_BASE_URL}/api/factors/combinations/ranking`, {
    params,
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function fetchFactorCombinationSignals(symbol, limit, options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factors/combinations/signals`, {
    params: { symbol, limit, topPerDuration: options.topPerDuration },
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

export async function fetchPaperLiveCandidates(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factors/combinations/paper-live/candidates`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? PAPER_LIVE_CANDIDATES_TIMEOUT_MS,
  });
  return data;
}

export async function runPaperLiveDailyLoop(symbol, duration = "10m") {
  const { data } = await axios.post(`${API_BASE_URL}/api/factors/combinations/paper-live/daily-loop`, null, {
    params: { symbol, duration },
    timeout: DAILY_LOOP_TIMEOUT_MS,
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
  if (config.profile) {
    params.profile = config.profile;
  }
  if (config.baseFactorLimit) {
    params.baseFactorLimit = config.baseFactorLimit;
  }
  if (config.comboSizes) {
    params.comboSizes = config.comboSizes;
  }
  if (config.resultLimit) {
    params.resultLimit = config.resultLimit;
  }
  if (config.incremental) {
    params.incremental = true;
  }
  if (config.batchSize) {
    params.batchSize = config.batchSize;
  }
  return params;
}
