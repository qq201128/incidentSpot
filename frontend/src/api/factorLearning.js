import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;
const REFRESH_TIMEOUT_MS = 90_000;

export async function fetchFactorLearningMemory(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factor-learning/memory`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function fetchFactorLearningOperators(options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/factor-learning/operators`, {
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function fetchLstmStatus(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/lstm/status`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function requestFactorLearningRefresh(symbol, duration = "10m", runAgent = true) {
  const { data } = await axios.post(`${API_BASE_URL}/api/factor-learning/refresh`, null, {
    params: { symbol, duration, runAgent },
    timeout: REFRESH_TIMEOUT_MS,
  });
  return data;
}
