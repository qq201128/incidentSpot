import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;
const REFRESH_QUEUE_TIMEOUT_MS = 15_000;
const RETRAIN_QUEUE_TIMEOUT_MS = 45_000;
export const DEFAULT_MODEL_SEARCH_RESOURCE = Object.freeze({
  internalThreads: 1,
  parallelWorkers: 1,
  xgboostProcessWorkers: 1,
});

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

export async function requestLstmCandidateSearch(symbol, duration = "10m", profile = "fast", options = {}) {
  return requestModelCandidateSearch("lstm", symbol, duration, profile, options);
}

export async function requestModelCandidateSearch(
  family,
  symbol,
  duration = "10m",
  profile = "fast",
  options = {},
) {
  const resource = modelSearchResourceParams(options);
  const { data } = await axios.post(`${API_BASE_URL}/api/models/${family}/candidate-search`, null, {
    params: {
      symbol,
      duration,
      profile,
      resetHistory: Boolean(options.resetHistory),
      ...resource,
    },
    timeout: REFRESH_QUEUE_TIMEOUT_MS,
  });
  return data;
}

export async function requestModelRetrainAll(options = {}) {
  const resource = modelSearchResourceParams(options);
  const { data } = await axios.post(`${API_BASE_URL}/api/models/search/retrain-all`, null, {
    params: {
      symbols: options.symbols,
      durations: options.durations,
      families: options.families,
      profile: options.profile || "full",
      resetHistory: options.resetHistory !== false,
      ...resource,
    },
    timeout: RETRAIN_QUEUE_TIMEOUT_MS,
  });
  return data;
}

function modelSearchResourceParams(options) {
  return {
    internalThreads: positiveIntOrDefault(options.internalThreads, DEFAULT_MODEL_SEARCH_RESOURCE.internalThreads),
    parallelWorkers: positiveIntOrDefault(options.parallelWorkers, DEFAULT_MODEL_SEARCH_RESOURCE.parallelWorkers),
    xgboostProcessWorkers: positiveIntOrDefault(
      options.xgboostProcessWorkers,
      DEFAULT_MODEL_SEARCH_RESOURCE.xgboostProcessWorkers,
    ),
  };
}

function positiveIntOrDefault(value, fallback) {
  const selected = Number(value ?? fallback);
  if (!Number.isInteger(selected) || selected <= 0) {
    throw new Error(`model search resource value must be a positive integer: ${value}`);
  }
  return selected;
}

export async function requestFactorLearningRefresh(symbol, duration = "10m", runAgent = true) {
  const { data } = await axios.post(`${API_BASE_URL}/api/factor-learning/refresh`, null, {
    params: { symbol, duration, runAgent },
    timeout: REFRESH_QUEUE_TIMEOUT_MS,
  });
  return data;
}
