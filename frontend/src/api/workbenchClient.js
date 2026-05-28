import axios from "axios";
import { API_BASE_URL } from "./client";

const SUMMARY_TIMEOUT_MS = 30_000;

export async function fetchWorkbenchSummary(symbol, duration) {
  const startedAt = performance.now();
  const { data } = await axios.get(`${API_BASE_URL}/api/workbench/summary`, {
    params: { symbol, duration },
    timeout: SUMMARY_TIMEOUT_MS,
  });
  return { data, latencyMs: performance.now() - startedAt };
}

export async function fetchEventGovernance(symbol, duration) {
  const startedAt = performance.now();
  const { data } = await axios.get(`${API_BASE_URL}/api/workbench/event-governance`, {
    params: { symbol, duration },
    timeout: SUMMARY_TIMEOUT_MS,
  });
  return { data, latencyMs: performance.now() - startedAt };
}

/** 规则命中率：各结算周期因子数量（轻量，用于 Tab） */
export async function fetchAiHistoryMeta(symbol, { signal } = {}) {
  const startedAt = performance.now();
  const { data } = await axios.get(`${API_BASE_URL}/api/workbench/ai-history-success/meta`, {
    params: { symbol },
    timeout: SUMMARY_TIMEOUT_MS,
    signal,
  });
  return { data, latencyMs: performance.now() - startedAt };
}

/** 规则命中率：当前周期分页列表 + KPI */
export async function fetchAiHistorySuccess(
  symbol,
  { durationMinutes, page = 1, pageSize = 10, signal } = {},
) {
  const startedAt = performance.now();
  const { data } = await axios.get(`${API_BASE_URL}/api/workbench/ai-history-success`, {
    params: { symbol, durationMinutes, page, pageSize },
    timeout: SUMMARY_TIMEOUT_MS,
    signal,
  });
  return { data, latencyMs: performance.now() - startedAt };
}
