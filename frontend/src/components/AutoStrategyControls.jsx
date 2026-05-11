import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchAutoTradeStrategies, updateAutoTradeStrategy } from "../api/client";

export default function AutoStrategyControls({ symbol, amount, liveTradingEnabled }) {
  const [strategies, setStrategies] = useState([]);
  const strategiesRef = useRef([]);
  const [loading, setLoading] = useState(true);
  const [updatingKey, setUpdatingKey] = useState("");
  const [error, setError] = useState("");

  const groups = useMemo(() => _groupStrategies(strategies), [strategies]);

  const enabledKeys = useMemo(
    () =>
      strategies
        .filter((item) => item.enabled)
        .map((item) => `${item.strategyKey}\t${item.duration}`)
        .sort()
        .join("|"),
    [strategies],
  );

  function buildSlotUpdate(slot, enabled) {
    return updateAutoTradeStrategy(slot.strategyKey, {
      strategyKey: slot.strategyKey,
      enabled,
      liveTradingEnabled,
      symbol,
      duration: slot.duration,
      durationMinutes: slot.durationMinutes,
      qty: Number(amount),
    });
  }

  useEffect(() => {
    strategiesRef.current = strategies;
  }, [strategies]);

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

  /** 交易对 / 数量 / 模拟开关变更时同步到已开启的周期槽位 */
  useEffect(() => {
    if (!enabledKeys) return;
    const enabled = strategiesRef.current.filter((item) => item.enabled);
    if (!enabled.length) return;
    let stopped = false;
    Promise.all(enabled.map((item) => buildSlotUpdate(item, true)))
      .then((rows) => {
        if (!stopped) _mergeStrategyRows(setStrategies, rows);
      })
      .catch((err) => {
        if (!stopped) setError(_errorMessage(err, "同步策略配置失败"));
      });
    return () => {
      stopped = true;
    };
  }, [amount, enabledKeys, liveTradingEnabled, symbol]);

  const toggleSlot = useCallback(
    async (slot) => {
      if (slot.tradable === false) return;
      const key = `${slot.strategyKey}:${slot.duration}`;
      setUpdatingKey(key);
      setError("");
      try {
        const updated = await buildSlotUpdate(slot, !slot.enabled);
        _mergeStrategyRows(setStrategies, [updated]);
      } catch (err) {
        setError(_errorMessage(err, "更新策略配置失败"));
      } finally {
        setUpdatingKey("");
      }
    },
    [amount, liveTradingEnabled, symbol],
  );

  if (loading) {
    return <div className="strategy-empty">正在读取策略配置...</div>;
  }

  return (
    <div className="strategy-control-list">
      {groups.map((group) => (
        <div key={group.strategyKey} className="strategy-control-row">
          <div className="strategy-control-main">
            <strong>{group.name || group.strategyKey}</strong>
            <span>{group.description}</span>
            <StrategyBacktestSummary summary={group.backtestSummary} />
            {group.tradable === false && (
              <span className="strategy-disabled">{group.disabledReason}</span>
            )}
            <div className="strategy-duration-row">
              <span className="strategy-duration-label">预测与下单周期（可多选）</span>
              <div className="strategy-duration-chips">
                {group.slots.map((slot) => (
                  <button
                    key={slot.duration}
                    type="button"
                    className={`chip ${slot.enabled ? "active" : ""}`}
                    disabled={
                      updatingKey === `${slot.strategyKey}:${slot.duration}` || group.tradable === false
                    }
                    onClick={() => void toggleSlot(slot)}
                    title={slot.enabled ? "点击关闭该周期" : "点击开启该周期"}
                  >
                    {_durationChipLabel(slot.durationMinutes)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      ))}
      {!groups.length && <div className="strategy-empty">暂无可用策略</div>}
      {!!error && <div className="predict-error">{error}</div>}
    </div>
  );
}

function _groupStrategies(flat) {
  const map = new Map();
  for (const row of flat) {
    const k = row.strategyKey;
    if (!map.has(k)) {
      map.set(k, {
        strategyKey: k,
        name: row.name,
        description: row.description,
        tradable: row.tradable,
        disabledReason: row.disabledReason,
        backtestSummary: row.backtestSummary,
        slots: [],
      });
    }
    map.get(k).slots.push(row);
  }
  return [...map.values()].map((g) => ({
    ...g,
    slots: g.slots.sort((a, b) => Number(a.durationMinutes) - Number(b.durationMinutes)),
  }));
}

function _mergeStrategyRows(setStrategies, rows) {
  setStrategies((prev) =>
    prev.map((item) => {
      const updated = rows.find(
        (row) => row.strategyKey === item.strategyKey && row.duration === item.duration,
      );
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

function _errorMessage(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}

function _durationChipLabel(minutes) {
  const n = Number(minutes);
  if (n === 1440) return "1天";
  return `${n}分钟`;
}
