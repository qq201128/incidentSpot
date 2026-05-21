import axios from "axios";
import { API_BASE_URL } from "./client";

const DEFAULT_TIMEOUT_MS = 30_000;

export async function fetchMiningOverview(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/mining/overview`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  });
  return data;
}
