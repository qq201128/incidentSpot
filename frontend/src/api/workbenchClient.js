import axios from "axios";
import { API_BASE_URL } from "./client";

const SUMMARY_TIMEOUT_MS = 8_000;

export async function fetchWorkbenchSummary(symbol, duration, limit = 20) {
  const startedAt = performance.now();
  const { data } = await axios.get(`${API_BASE_URL}/api/workbench/summary`, {
    params: { symbol, duration, limit },
    timeout: SUMMARY_TIMEOUT_MS,
  });
  return { data, latencyMs: performance.now() - startedAt };
}
