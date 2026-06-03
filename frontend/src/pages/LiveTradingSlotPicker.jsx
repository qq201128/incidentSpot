import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchAutoTradeStrategies } from "../api/client";
import { strategyDurationLabel, strategyLabel } from "../utils/strategyLabels";
import { filterSlotRows, liveTradingPath } from "./liveTradingRoutes";
import "./LiveTradingSlotPicker.css";

export default function LiveTradingSlotPicker({ request }) {
  const { error, loading, slots } = useLiveTradingSlots(request);
  if (loading) {
    return <p className="live-trading-alert" role="status">正在读取执行槽位</p>;
  }
  if (error) {
    return <p className="live-trading-alert is-error" role="alert">{error}</p>;
  }
  return (
    <div className="live-trading-picker">
      <PickerHeader request={request} slotCount={slots.length} />
      {slots.length ? <SlotGrid slots={slots} /> : <EmptySlots request={request} />}
    </div>
  );
}

function useLiveTradingSlots(request) {
  const [slots, setSlots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let stopped = false;
    setLoading(true);
    setError("");
    fetchAutoTradeStrategies()
      .then((payload) => {
        if (stopped) return;
        setSlots(filterSlotRows(strategyRows(payload), request));
      })
      .catch((err) => {
        if (!stopped) setError(errorMessage(err, "读取执行槽位失败"));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [request]);

  return { error, loading, slots };
}

function PickerHeader({ request, slotCount }) {
  return (
    <header className="live-trading-picker-head">
      <div>
        <span className="section-kicker">Execution slots</span>
        <h2>选择执行槽位</h2>
      </div>
      <strong>{slotCount} 个槽位</strong>
      {request.hasSymbolParam || request.hasDurationParam ? <FilterText request={request} /> : null}
    </header>
  );
}

function FilterText({ request }) {
  return (
    <small>
      {request.hasSymbolParam ? request.symbol : "全部交易对"} ·{" "}
      {request.hasDurationParam ? strategyDurationLabel(request.duration) : "全部周期"}
    </small>
  );
}

function SlotGrid({ slots }) {
  return (
    <div className="live-trading-slot-grid">
      {slots.map((slot) => (
        <SlotCard key={`${slot.strategyKey}:${slot.symbol}:${slot.duration}`} slot={slot} />
      ))}
    </div>
  );
}

function SlotCard({ slot }) {
  return (
    <Link className="live-trading-slot-card" to={liveTradingPath(slot)}>
      <span>{slot.symbol} · {strategyDurationLabel(slot.duration)}</span>
      <strong>{strategyLabel(slot.strategyKey)}</strong>
      <code title={slot.strategyKey}>{slot.strategyKey}</code>
      <SlotBadges slot={slot} />
      {slot.tradable === false ? <em>{slot.disabledReason}</em> : null}
    </Link>
  );
}

function SlotBadges({ slot }) {
  return (
    <span className="live-trading-slot-badges">
      <b className={slot.enabled ? "is-on" : ""}>{slot.enabled ? "已启用" : "已关闭"}</b>
      <b className={slot.liveTradingEnabled ? "is-live" : ""}>
        {slot.liveTradingEnabled ? "LIVE" : "SIM"}
      </b>
    </span>
  );
}

function EmptySlots({ request }) {
  return <p className="live-trading-alert is-error" role="alert">{emptySlotMessage(request)}</p>;
}

function strategyRows(payload) {
  if (!Array.isArray(payload?.strategies)) {
    throw new Error("auto-trade strategies payload missing strategies array");
  }
  return payload.strategies;
}

function emptySlotMessage(request) {
  if (request.hasSymbolParam || request.hasDurationParam) {
    return `未找到 ${request.symbol} ${request.duration} 的执行槽位`;
  }
  return "未读取到可配置执行槽位";
}

function errorMessage(error, fallback) {
  return error?.response?.data?.detail || error?.message || fallback;
}
