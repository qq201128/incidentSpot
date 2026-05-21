import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;
const REFRESH_QUEUE_TIMEOUT_MS = 15_000;

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
  return fetchModelStatus("lstm", symbol, duration, options);
}

export async function fetchModelStatus(family, symbol, duration = "10m", options = {}) {
  return fetchModelFamilyStatus(family, symbol, duration, options);
}

export async function fetchModelFamilyStatus(family, symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/models/${family}/status`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}

export async function requestLstmCandidateSearch(symbol, duration = "10m", profile = "full") {
  return requestModelCandidateSearch("lstm", symbol, duration, profile);
}

export async function requestModelCandidateSearch(family, symbol, duration = "10m", profile = "full", parallelWorkers = 10) {
  const { data } = await axios.post(`${API_BASE_URL}/api/models/${family}/candidate-search`, null, {
    params: { symbol, duration, profile, parallelWorkers },
    timeout: REFRESH_QUEUE_TIMEOUT_MS,
  });
  return data;
}

export async function requestFactorLearningRefresh(symbol, duration = "10m", runAgent = true) {
  const { data } = await axios.post(`${API_BASE_URL}/api/factor-learning/refresh`, null, {
    params: { symbol, duration, runAgent },
    timeout: REFRESH_QUEUE_TIMEOUT_MS,
  });
  return data;
}
