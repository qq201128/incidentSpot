import axios from "axios";
import { API_BASE_URL } from "./client";

export async function fetchAutoTradeStrategy(strategyKey, { symbol, duration }) {
  const { data } = await axios.get(
    `${API_BASE_URL}/api/auto-trade/strategies/${encodeURIComponent(strategyKey)}`,
    { params: { symbol, duration } },
  );
  return data;
}
