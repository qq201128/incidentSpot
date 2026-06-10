import axios from "axios";
import { API_BASE_URL } from "./client";

const TIMEOUT_MS = 30_000;

export async function fetchEventFinalDecisionLatest(symbol, duration, options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/event-final-decisions/latest`, {
    params: { symbol, duration },
    timeout: TIMEOUT_MS,
    signal: options.signal,
  });
  return data?.latest ?? null;
}

export async function fetchEventFinalDecisionSummary(symbol, duration, options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/event-final-decisions/summary`, {
    params: { symbol, duration },
    timeout: TIMEOUT_MS,
    signal: options.signal,
  });
  return data;
}
