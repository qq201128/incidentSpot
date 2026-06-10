import axios from "axios";
import { API_BASE_URL } from "./client";

const BUNDLE_TIMEOUT_MS = 45_000;

export async function fetchResearchModelBundle(symbol, duration = "10m", options = {}) {
  const { data } = await axios.get(`${API_BASE_URL}/api/models/research-bundle`, {
    params: { symbol, duration },
    signal: options.signal,
    timeout: options.timeoutMs ?? BUNDLE_TIMEOUT_MS,
  });
  return data;
}
