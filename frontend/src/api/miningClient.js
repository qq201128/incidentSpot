import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;

export async function fetchMiningOverview(symbol, duration = "10m", options = {}) {
  const params = { symbol, duration };
  if (options.fresh) params.fresh = true;
  const { data } = await axios.get(`${API_BASE_URL}/api/mining/overview`, {
    params,
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}
