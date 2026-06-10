import axios from "axios";
import { API_BASE_URL } from "./client";

const LOCAL_REQUEST_TIMEOUT_MS = 8_000;
const EVENTS_PAGE_DEDUPE_MS = 250;
const eventsPageInflight = new Map();

export async function createEvent(payload) {
  const { data } = await axios.post(`${API_BASE_URL}/api/events`, payload);
  return data;
}

export async function createOrder(eventId, payload) {
  const { data } = await axios.post(`${API_BASE_URL}/api/events/${eventId}/orders`, payload);
  return data;
}

export async function createQuickTrade(payload) {
  const { data } = await axios.post(`${API_BASE_URL}/api/events/quick-trade`, payload);
  return data;
}

export async function settleEvent(eventId) {
  const { data } = await axios.post(`${API_BASE_URL}/api/events/${eventId}/settle`);
  return data;
}

export async function fetchEventsPage(options = {}) {
  return dedupedEventsPageRequest(eventPageParams(options));
}

export async function deleteAllEvents() {
  const { data } = await axios.delete(`${API_BASE_URL}/api/events`);
  return data;
}

export async function deleteEventsByStrategy(strategyKey) {
  const { data } = await axios.delete(`${API_BASE_URL}/api/events`, {
    params: { strategyKey },
  });
  return data;
}

async function dedupedEventsPageRequest(params) {
  const key = stableParamsKey(params);
  const now = performance.now();
  const existing = eventsPageInflight.get(key);
  if (existing && now - existing.startedAt <= EVENTS_PAGE_DEDUPE_MS) {
    return existing.promise;
  }
  const promise = requestEventsPage(params);
  eventsPageInflight.set(key, { promise, startedAt: now });
  try {
    return await promise;
  } finally {
    const current = eventsPageInflight.get(key);
    if (current?.promise === promise) eventsPageInflight.delete(key);
  }
}

async function requestEventsPage(params) {
  const { data } = await axios.get(`${API_BASE_URL}/api/events`, {
    params,
    timeout: LOCAL_REQUEST_TIMEOUT_MS,
  });
  return data;
}

function eventPageParams({ symbol, strategyKey, durationMinutes, page = 1, pageSize = 8, q } = {}) {
  const params = { page, pageSize };
  if (symbol && String(symbol).trim()) params.symbol = symbol;
  if (q) params.q = q;
  if (strategyKey) params.strategyKey = strategyKey;
  if (durationMinutes != null && Number.isFinite(Number(durationMinutes))) {
    params.durationMinutes = durationMinutes;
  }
  return params;
}

function stableParamsKey(params) {
  return Object.keys(params)
    .sort()
    .map((key) => `${key}=${String(params[key])}`)
    .join("&");
}
