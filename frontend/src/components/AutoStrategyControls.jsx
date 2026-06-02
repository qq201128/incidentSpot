import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchAutoTradeStrategies, fetchSimulationSlots, updateAutoTradeStrategy } from "../api/client";
import SimulationSlotDetails, { SimulationSlotReports } from "./SimulationSlotDetails";

export default function AutoStrategyControls({ symbol, amount, reloadKey = 0 }) {
  const [strategies, setStrategies] = useState([]);
  const strategiesRef = useRef([]);
  const [loading, setLoading] = useState(true);
  const [updatingKey, setUpdatingKey] = useState("");
  const [error, setError] = useState("");
  const [slotReports, setSlotReports] = useState([]);
  const activeSymbol = symbol.trim().toUpperCase();
  const visibleStrategies = useMemo(
    () => strategies.filter((item) => String(item.symbol || "").toUpperCase() === activeSymbol),
    [activeSymbol, strategies],
  );

  const groups = useMemo(() => _groupStrategies(visibleStrategies), [visibleStrategies]);

  const enabledKeys = useMemo(
    () =>
      visibleStrategies
        .filter((item) => item.enabled)
        .map((item) => `${item.strategyKey}\t${item.symbol}\t${item.duration}`)
        .sort()
        .join("|"),
    [visibleStrategies],
  );

  function buildSlotUpdate(slot, enabled) {
    return updateAutoTradeStrategy(slot.strategyKey, {
      strategyKey: slot.strategyKey,
      enabled,
      liveTradingEnabled: Boolean(slot.liveTradingEnabled),
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
      .then(async (strategiesData) => {
        if (stopped) return;
        const rows = Array.isArray(strategiesData?.strategies) ? strategiesData.strategies : [];
        setStrategies(rows);
        const reports = await _fetchSimulationReports(activeSymbol, rows);
        if (!stopped) setSlotReports(reports);
      })
      .catch((err) => {
        if (!stopped) setError(_errorMessage(err, "读取执行配置失败"));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [activeSymbol, reloadKey]);

  /** 交易对或数量变更时同步到已开启的周期槽位，保留实盘开关状态。 */
  useEffect(() => {
    if (!enabledKeys) return;
    const enabled = strategiesRef.current.filter(
      (item) => item.enabled && String(item.symbol || "").toUpperCase() === activeSymbol,
    );
    if (!enabled.length) return;
    let stopped = false;
    Promise.all(enabled.map((item) => buildSlotUpdate(item, true)))
      .then((rows) => {
        if (!stopped) _mergeStrategyRows(setStrategies, rows);
      })
      .catch((err) => {
        if (!stopped) setError(_errorMessage(err, "同步执行配置失败"));
      });
    return () => {
      stopped = true;
    };
  }, [activeSymbol, amount, enabledKeys, symbol]);

  const toggleSlot = useCallback(
    async (slot) => {
      if (slot.tradable === false) return;
      const key = `${slot.strategyKey}:${slot.symbol}:${slot.duration}`;
      setUpdatingKey(key);
      setError("");
      try {
        const updated = await buildSlotUpdate(slot, !slot.enabled);
        _mergeStrategyRows(setStrategies, [updated]);
      } catch (err) {
        setError(_errorMessage(err, "更新执行配置失败"));
      } finally {
        setUpdatingKey("");
      }
    },
    [amount, symbol],
  );

  if (loading) {
    return <div className="strategy-empty">正在读取执行配置...</div>;
  }

  return (
    <div className="strategy-control-list">
      {groups.map((group) => (
        <div key={group.strategyKey} className="strategy-control-row">
          <div className="strategy-control-main">
            <div className="strategy-control-head">
              <div className="strategy-control-titles">
                <strong>{group.name || group.strategyKey}</strong>
              </div>
              <button
                type="button"
                className="strategy-live-mode-btn sim"
                aria-pressed={false}
                disabled={
                  updatingKey === `${group.strategyKey}:__live__` ||
                  group.tradable === false
                }
                title={_liveButtonTitle(group)}
              >
                <span className="mode-dot" />
                <span className="strategy-live-mode-label">
                  {_liveButtonLabel(group)}
                </span>
              </button>
            </div>
            <span>{group.description}</span>
            <StrategyBacktestSummary summary={group.backtestSummary} />
            {group.tradable === false && (
              <span className="strategy-disabled">{group.disabledReason}</span>
            )}
            <div className="strategy-duration-row">
              <span className="strategy-duration-label">预测与下单周期（可多选）</span>
              <div className="strategy-duration-chips">
                {group.slots.map((slot) => (
                  <SlotChip
                    key={slot.duration}
                    slot={slot}
                    group={group}
                    updatingKey={updatingKey}
                    onToggle={toggleSlot}
                  />
                ))}
              </div>
            </div>
            <SimulationSlotDetails slots={group.slots} />
          </div>
        </div>
      ))}
      {!groups.length && <div className="strategy-empty">暂无可用执行项</div>}
      <SimulationSlotReports reports={slotReports} />
      {!!error && <div className="predict-error">{error}</div>}
    </div>
  );
}

async function _fetchSimulationReports(symbol, strategies) {
  const durations = [...new Set(strategies.map((row) => row.duration).filter(Boolean))];
  const wanted = durations.length ? durations : ["10m"];
  return Promise.all(wanted.map((duration) => fetchSimulationSlots(symbol, duration)));
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
        (row) =>
          row.strategyKey === item.strategyKey &&
          row.symbol === item.symbol &&
          row.duration === item.duration,
      );
      return updated ? { ...item, ...updated } : item;
    }),
  );
}

function _liveButtonLabel(group) {
  return group.slots.some((slot) => slot.liveTradingEnabled) ? "含实盘" : "仅模拟";
}

function _liveButtonTitle(group) {
  if (group.slots.some((slot) => slot.liveTradingEnabled)) {
    return "该执行项存在已开启实盘的周期槽位；请在实盘配置页管理。";
  }
  return "实盘开关请在研究驾驶舱进入的实盘配置页管理。";
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

function SlotChip({ slot, group, updatingKey, onToggle }) {
  return (
    <button
      type="button"
      className={`chip ${slot.enabled ? "active" : ""}`}
      disabled={
        updatingKey === `${slot.strategyKey}:${slot.symbol}:${slot.duration}` ||
        group.tradable === false ||
        updatingKey === `${group.strategyKey}:__live__`
      }
      onClick={() => void onToggle(slot)}
      title={slot.enabled ? "点击关闭该周期" : "点击开启该周期"}
    >
      {_durationChipLabel(slot.durationMinutes)}
    </button>
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
