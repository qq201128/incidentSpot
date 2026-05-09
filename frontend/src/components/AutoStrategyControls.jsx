import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchAutoTradeStrategies, updateAutoTradeStrategy } from "../api/client";

export default function AutoStrategyControls({
  symbol,
  duration,
  durationMinutes,
  amount,
  liveTradingEnabled,
}) {
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [updatingKey, setUpdatingKey] = useState("");
  const [error, setError] = useState("");

  const enabledKeys = useMemo(
    () => strategies.filter((item) => item.enabled).map((item) => item.strategyKey).join("|"),
    [strategies],
  );

  useEffect(() => {
    let stopped = false;
    fetchAutoTradeStrategies()
      .then((data) => {
        if (!stopped) setStrategies(Array.isArray(data?.strategies) ? data.strategies : []);
      })
      .catch((err) => {
        if (!stopped) setError(_errorMessage(err, "读取策略配置失败"));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, []);

  useEffect(() => {
    if (!enabledKeys) return;
    let stopped = false;
    const enabled = strategies.filter((item) => item.enabled);
    Promise.all(enabled.map((item) => _updateStrategy(item, true)))
      .then((rows) => {
        if (!stopped) _mergeStrategyRows(setStrategies, rows);
      })
      .catch((err) => {
        if (!stopped) setError(_errorMessage(err, "同步策略配置失败"));
      });
    return () => {
      stopped = true;
    };
  }, [amount, duration, durationMinutes, enabledKeys, liveTradingEnabled, symbol]);

  const toggleStrategy = useCallback(
    async (strategy) => {
      setUpdatingKey(strategy.strategyKey);
      setError("");
      try {
        const updated = await _updateStrategy(strategy, !strategy.enabled);
        _mergeStrategyRows(setStrategies, [updated]);
      } catch (err) {
        setError(_errorMessage(err, "更新策略配置失败"));
      } finally {
        setUpdatingKey("");
      }
    },
    [amount, duration, durationMinutes, liveTradingEnabled, symbol],
  );

  function _updateStrategy(strategy, enabled) {
    return updateAutoTradeStrategy(strategy.strategyKey, {
      strategyKey: strategy.strategyKey,
      enabled,
      liveTradingEnabled,
      symbol,
      duration,
      durationMinutes,
      qty: Number(amount),
    });
  }

  if (loading) {
    return <div className="strategy-empty">正在读取策略配置...</div>;
  }

  return (
    <div className="strategy-control-list">
      {strategies.map((strategy) => (
        <div key={strategy.strategyKey} className="strategy-control-row">
          <div>
            <strong>{strategy.name || strategy.strategyKey}</strong>
            <span>{strategy.description}</span>
            <StrategyBacktestSummary summary={strategy.backtestSummary} />
            {strategy.tradable === false && <span className="strategy-disabled">{strategy.disabledReason}</span>}
          </div>
          <button
            type="button"
            className={strategy.enabled ? "strategy-toggle on" : "strategy-toggle off"}
            disabled={updatingKey === strategy.strategyKey || strategy.tradable === false}
            onClick={() => void toggleStrategy(strategy)}
          >
            {_toggleLabel(strategy, updatingKey)}
          </button>
        </div>
      ))}
      {!strategies.length && <div className="strategy-empty">暂无可用策略</div>}
      {!!error && <div className="predict-error">{error}</div>}
    </div>
  );
}

function _mergeStrategyRows(setStrategies, rows) {
  setStrategies((prev) =>
    prev.map((item) => {
      const updated = rows.find((row) => row.strategyKey === item.strategyKey);
      return updated ? { ...item, ...updated } : item;
    }),
  );
}

function StrategyBacktestSummary({ summary }) {
  if (!summary) return null;
  const winRate = `${(Number(summary.winRate) * 100).toFixed(2)}%`;
  const minDay = `${(Number(summary.minDailyWinRate) * 100).toFixed(2)}%`;
  return (
    <span className="strategy-metric">
      回测 {summary.trades} 单 / {summary.wins} 胜 / 胜率 {winRate} / 最低单日 {minDay}
    </span>
  );
}

function _toggleLabel(strategy, updatingKey) {
  if (strategy.tradable === false) return "待接入";
  if (updatingKey === strategy.strategyKey) return "同步中";
  return strategy.enabled ? "开启" : "关闭";
}

function _errorMessage(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}
