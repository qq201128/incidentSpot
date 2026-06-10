import axios from "axios";
import { API_BASE_URL } from "./client";

const LOCAL_REQUEST_TIMEOUT_MS = 8_000;

export async function updateAutoTradeSettings(payload) {
  const { data } = await axios.put(`${API_BASE_URL}/api/auto-trade/settings`, payload);
  return data;
}

export async function fetchAutoTradeStrategies() {
  const { data } = await axios.get(`${API_BASE_URL}/api/auto-trade/strategies`);
  return data;
}

export async function fetchSimulationSlots(symbol, duration = "10m") {
  const { data } = await axios.get(`${API_BASE_URL}/api/auto-trade/simulation-slots`, {
    params: { symbol, duration },
    timeout: LOCAL_REQUEST_TIMEOUT_MS,
  });
  return data;
}

export async function updateAutoTradeStrategy(strategyKey, payload) {
  const { data } = await axios.put(
    `${API_BASE_URL}/api/auto-trade/strategies/${encodeURIComponent(strategyKey)}`,
    payload,
  );
  return data;
}
