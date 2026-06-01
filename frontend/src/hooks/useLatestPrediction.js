import { useCallback, useEffect, useState } from "react";
import { fetchLatestPrediction, openPredictionSocket } from "../api/client";

const FRESH_PREDICTION_MS = 30000;

/** @param {string} predictionDuration API 周期：10m | 30m | 60m | 1d（与图表 / 事件合约一致） */
export function useLatestPrediction(symbol, onStatus, predictionDuration = "10m") {
  const [latestPrediction, setLatestPrediction] = useState(null);

  useEffect(() => {
    let ws;
    let retryTimer;
    let retryCount = 0;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      ws = openPredictionSocket(symbol, predictionDuration, setLatestPrediction);
      ws.onopen = () => {
        retryCount = 0;
        onStatus("预测实时连接已建立");
      };
      ws.onerror = (err) => {
        if (stopped) return;
        // 浏览器常在 onclose 之前误报 onerror；勿用其覆盖首页其它加载状态
        console.error("预测 WebSocket 异常", err);
      };
      ws.onclose = () => {
        if (stopped) return;
        const wait = Math.min(1000 * 2 ** retryCount, 10000);
        retryCount += 1;
        onStatus(`${Math.floor(wait / 1000)} 秒后重连预测流`);
        retryTimer = setTimeout(connect, wait);
      };
    };

    connect();
    return () => {
      stopped = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (ws) ws.close();
    };
  }, [onStatus, symbol, predictionDuration]);

  const getFreshPrediction = useCallback(async (targetSymbol, duration) => {
    if (_isFreshPrediction(latestPrediction, targetSymbol, duration)) return latestPrediction;
    const latest = await fetchLatestPrediction(targetSymbol, duration);
    if (!_isFreshPrediction(latest, targetSymbol, duration)) {
      throw new Error("后台规则信号尚未产生 2.5 秒内的新结果");
    }
    return latest;
  }, [latestPrediction]);

  return { latestPrediction, getFreshPrediction };
}

function _isFreshPrediction(prediction, symbol, duration) {
  if (!prediction) return false;
  if (prediction.symbol !== symbol.toUpperCase() || prediction.duration !== duration) return false;
  const generatedAt = Date.parse(prediction.generatedAt);
  if (Number.isFinite(generatedAt)) {
    return Date.now() - generatedAt <= FRESH_PREDICTION_MS;
  }
  const ageMs = Number(prediction.ageMs);
  return Number.isFinite(ageMs) && ageMs <= FRESH_PREDICTION_MS;
}
