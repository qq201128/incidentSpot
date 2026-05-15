import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchAutoTradeStrategies, updateAutoTradeStrategy } from "../api/client";

export default function AutoStrategyControls({ symbol, amount }) {
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
      liveTradingEnabled: !!slot.liveTradingEnabled,
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

  /** 交易对或数量变更时同步到已开启的周期槽位（各槽位保留自身实盘开关） */
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
  }, [amount, enabledKeys, symbol]);

  const toggleStrategyLive = useCallback(
    async (group) => {
      if (group.tradable === false) return;
      const nextLive = !group.slots.some((s) => s.liveTradingEnabled);
      const busyKey = `${group.strategyKey}:__live__`;
      setUpdatingKey(busyKey);
      setError("");
      try {
        const rows = await Promise.all(
          group.slots.map((slot) =>
            updateAutoTradeStrategy(slot.strategyKey, {
              strategyKey: slot.strategyKey,
              enabled: slot.enabled,
              liveTradingEnabled: nextLive,
              symbol,
              duration: slot.duration,
              durationMinutes: slot.durationMinutes,
              qty: Number(amount),
            }),
          ),
        );
        _mergeStrategyRows(setStrategies, rows);
      } catch (err) {
        setError(_errorMessage(err, "更新策略实盘开关失败"));
      } finally {
        setUpdatingKey("");
      }
    },
    [amount, symbol],
  );

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
    [amount, symbol],
  );

  if (loading) {
    return <div className="strategy-empty">正在读取策略配置...</div>;
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
                className={`strategy-live-mode-btn ${group.slots.some((s) => s.liveTradingEnabled) ? "live" : "sim"}`}
                aria-pressed={group.slots.some((s) => s.liveTradingEnabled)}
                disabled={updatingKey === `${group.strategyKey}:__live__` || group.tradable === false}
                onClick={() => void toggleStrategyLive(group)}
                title="该策略下所有结算周期共用此开关；仅对已点亮的周期自动下单。"
              >
                <span className="mode-dot" />
                <span className="strategy-live-mode-label">
                  {group.slots.some((s) => s.liveTradingEnabled) ? "实盘" : "模拟"}
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
                  <button
                    key={slot.duration}
                    type="button"
                    className={`chip ${slot.enabled ? "active" : ""}`}
                    disabled={
                      updatingKey === `${slot.strategyKey}:${slot.duration}` ||
                      group.tradable === false ||
                      updatingKey === `${group.strategyKey}:__live__`
                    }
                    onClick={() => void toggleSlot(slot)}
                    title={slot.enabled ? "点击关闭该周期" : "点击开启该周期"}
                  >
                    {_durationChipLabel(slot.durationMinutes)}
                  </button>
                ))}
              </div>
              <StrategyDemotionSummary slots={group.slots} />
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

function StrategyDemotionSummary({ slots }) {
  const rows = slots.filter((slot) => slot.demotion?.status && slot.demotion.status !== "unknown");
  if (!rows.length) return null;
  return (
    <div className="strategy-demotion-list">
      {rows.map((slot) => (
        <span key={slot.duration} className={`strategy-demotion ${slot.demotion.status}`}>
          {_durationChipLabel(slot.durationMinutes)} {demotionStatusText(slot.demotion)}：
          {demotionReason(slot.demotion)}
        </span>
      ))}
    </div>
  );
}

function demotionStatusText(demotion) {
  if (demotion?.status === "active") return "高胜率可用";
  if (demotion?.status === "collecting") return "持续观察";
  if (demotion?.status === "demoted") return "已降级";
  if (demotion?.status === "paused") return "已暂停";
  return demotion?.status || "未评估";
}

function demotionReason(demotion) {
  const metrics = demotion?.metrics || {};
  const winRate = metrics.winRate == null ? "--" : `${(Number(metrics.winRate) * 100).toFixed(1)}%`;
  const losses = metrics.consecutiveLosses ?? 0;
  if (demotion?.reason === "consecutive_losses") return `连续亏损 ${losses} 次`;
  if (demotion?.reason === "insufficient_settled_samples") {
    return `样本 ${metrics.sampleCount ?? 0}，满 ${demotion.thresholds?.activeSampleCount ?? 20} 单再判定`;
  }
  if (demotion?.reason === "stable_live_target_met") return `最近胜率 ${winRate}`;
  if (demotion?.reason === "profit_factor_below_one") return `盈亏比 ${metrics.profitFactor}`;
  if (demotion?.reason?.includes("win_rate")) return `最近胜率 ${winRate}`;
  if (demotion?.reason === "offline_promotion") return "离线达标，等待实盘模拟样本";
  return demotion?.reason || "等待评估";
}

function _errorMessage(err, fallback) {
  return err?.response?.data?.detail || err?.message || fallback;
}

function _durationChipLabel(minutes) {
  const n = Number(minutes);
  if (n === 1440) return "1天";
  return `${n}分钟`;
}
