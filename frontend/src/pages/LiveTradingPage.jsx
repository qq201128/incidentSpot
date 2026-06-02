import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { fetchAutoTradeStrategies, updateAutoTradeStrategy } from "../api/client";
import { strategyDurationLabel, strategyLabel } from "../utils/strategyLabels";
import "./LiveTradingPage.css";

const DEFAULT_QTY = 5;

export default function LiveTradingPage() {
  const [searchParams] = useSearchParams();
  const request = useMemo(() => requestFromParams(searchParams), [searchParams]);
  const { error, loading, save, saving, status, slot, updateDraft } = useLiveTradingSlot(request);

  return (
    <main className="live-trading-page layout">
      <header className="live-trading-topbar">
        <div>
          <span className="section-kicker">Live trading configuration</span>
          <h1>实盘配置</h1>
        </div>
        <Link className="live-trading-back" to={researchPath(request)}>
          返回研究驾驶舱
        </Link>
      </header>

      <section className="live-trading-card">
        <SlotHeader request={request} slot={slot} loading={loading} />
        {error ? <p className="live-trading-alert is-error" role="alert">{error}</p> : null}
        {status ? <p className="live-trading-alert is-ok" role="status">{status}</p> : null}
        {slot ? <SlotForm slot={slot} saving={saving} onChange={updateDraft} onSave={save} /> : null}
      </section>
    </main>
  );
}

function useLiveTradingSlot(request) {
  const [slot, setSlot] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  useEffect(() => {
    let stopped = false;
    setLoading(true);
    setError("");
    setStatus("");
    fetchAutoTradeStrategies()
      .then((payload) => {
        if (stopped) return;
        const found = findRequestedSlot(payload?.strategies, request);
        setSlot(found);
        if (!found) setError(missingSlotMessage(request));
      })
      .catch((err) => {
        if (!stopped) setError(errorMessage(err, "读取实盘配置失败"));
      })
      .finally(() => {
        if (!stopped) setLoading(false);
      });
    return () => {
      stopped = true;
    };
  }, [request]);

  const updateDraft = useCallback((patch) => {
    setSlot((current) => (current ? { ...current, ...patch } : current));
    setStatus("");
  }, []);

  const save = useCallback(async () => {
    if (!slot) return;
    setSaving(true);
    setError("");
    setStatus("");
    try {
      const updated = await updateAutoTradeStrategy(slot.strategyKey, payloadFromSlot(slot));
      setSlot((current) => ({ ...(current || slot), ...updated }));
      setStatus("实盘配置已保存");
    } catch (err) {
      setError(errorMessage(err, "保存实盘配置失败"));
    } finally {
      setSaving(false);
    }
  }, [slot]);

  return { error, loading, save, saving, status, slot, updateDraft };
}

function SlotHeader({ request, slot, loading }) {
  const key = slot?.strategyKey || request.strategyKey;
  return (
    <div className="live-trading-slot-head">
      <div>
        <span>{request.symbol} · {strategyDurationLabel(request.duration)}</span>
        <h2>{strategyLabel(key)}</h2>
        <code title={key}>{key || "strategyKey_missing"}</code>
      </div>
      <strong className={slot?.liveTradingEnabled ? "is-live" : "is-sim"}>
        {loading ? "读取中" : slot?.liveTradingEnabled ? "LIVE" : "SIM"}
      </strong>
    </div>
  );
}

function SlotForm({ slot, saving, onChange, onSave }) {
  return (
    <form className="live-trading-form" onSubmit={(event) => handleSubmit(event, onSave)}>
      <div className="live-trading-switches">
        <ToggleRow
          checked={Boolean(slot.enabled)}
          label="启用自动执行"
          onChange={(enabled) => onChange({ enabled })}
        />
        <ToggleRow
          checked={Boolean(slot.liveTradingEnabled)}
          label="启用真实下单"
          onChange={(liveTradingEnabled) => onChange({ liveTradingEnabled })}
        />
      </div>
      <label className="live-trading-field">
        <span>单笔数量</span>
        <input
          min="0.0001"
          step="0.0001"
          type="number"
          value={slot.qty ?? DEFAULT_QTY}
          onChange={(event) => onChange({ qty: event.target.value })}
        />
      </label>
      <dl className="live-trading-details">
        <div><dt>交易对</dt><dd>{slot.symbol}</dd></div>
        <div><dt>观察周期</dt><dd>{strategyDurationLabel(slot.duration)}</dd></div>
        <div><dt>执行项</dt><dd>{slot.strategyKey}</dd></div>
        <div><dt>状态</dt><dd>{slot.enabled ? "已启用" : "已关闭"} / {slot.liveTradingEnabled ? "实盘" : "模拟"}</dd></div>
      </dl>
      <button type="submit" className="live-trading-save" disabled={saving}>
        {saving ? "保存中" : "保存配置"}
      </button>
    </form>
  );
}

function ToggleRow({ checked, label, onChange }) {
  return (
    <label className="live-trading-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function handleSubmit(event, onSave) {
  event.preventDefault();
  void onSave();
}

function requestFromParams(searchParams) {
  return {
    duration: String(searchParams.get("duration") || "10m"),
    strategyKey: String(searchParams.get("strategyKey") || ""),
    symbol: String(searchParams.get("symbol") || "BTCUSDT").trim().toUpperCase(),
  };
}

function findRequestedSlot(strategies, request) {
  const rows = Array.isArray(strategies) ? strategies : [];
  return rows.find((row) => (
    row.strategyKey === request.strategyKey &&
    String(row.symbol || "").toUpperCase() === request.symbol &&
    row.duration === request.duration
  )) || null;
}

function payloadFromSlot(slot) {
  return {
    strategyKey: slot.strategyKey,
    enabled: Boolean(slot.enabled),
    liveTradingEnabled: Boolean(slot.liveTradingEnabled),
    symbol: slot.symbol,
    duration: slot.duration,
    durationMinutes: Number(slot.durationMinutes),
    qty: Number(slot.qty),
  };
}

function missingSlotMessage(request) {
  if (!request.strategyKey) return "URL 缺少 strategyKey，无法定位执行槽位";
  return `未找到 ${request.symbol} ${request.duration} 的执行槽位：${request.strategyKey}`;
}

function researchPath(request) {
  return `/research-dashboard?symbol=${encodeURIComponent(request.symbol)}&duration=${encodeURIComponent(request.duration)}`;
}

function errorMessage(error, fallback) {
  return error?.response?.data?.detail || error?.message || fallback;
}
