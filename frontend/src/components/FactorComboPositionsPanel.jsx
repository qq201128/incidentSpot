import { useEffect, useMemo, useState } from "react";
import { fetchFactorCombinationPositions } from "../api/factorCombinations";
import { settledExpectedProfitUsdt } from "../utils/eventSettlement";
import "./FactorComboPositionsPanel.css";

export default function FactorComboPositionsPanel({ signal, symbol }) {
  const [state, setState] = useState({ loading: false, error: "", data: null });
  const factorName = signal?.factorName || "";
  const duration = signal?.duration || "";

  useEffect(() => {
    if (!factorName || !duration || !symbol) {
      setState({ loading: false, error: "", data: null });
      return undefined;
    }
    const ac = new AbortController();
    setState((prev) => ({ ...prev, loading: true, error: "" }));
    fetchFactorCombinationPositions(symbol, duration, factorName, { signal: ac.signal })
      .then((data) => {
        if (!ac.signal.aborted) setState({ loading: false, error: "", data });
      })
      .catch((error) => {
        if (!isCanceled(error, ac.signal)) {
          setState({ loading: false, error: error.message || "持仓读取失败", data: null });
        }
      });
    return () => ac.abort();
  }, [duration, factorName, symbol]);

  const events = useMemo(() => state.data?.events || [], [state.data]);
  if (!signal) return null;
  return (
    <section className="factor-combo-positions">
      <PanelTitle signal={signal} state={state} />
      {state.error ? <p className="factor-combo-position-error">{state.error}</p> : null}
      <PositionSummary data={state.data} />
      <PositionTable events={events} factorName={factorName} />
    </section>
  );
}

function PanelTitle({ signal, state }) {
  return (
    <div className="factor-combo-position-head">
      <div>
        <span className="section-kicker">组合持仓</span>
        <h3>{signal.duration} · {signal.factorDisplayName || signal.factorName}</h3>
      </div>
      <span>{state.loading ? "读取中…" : "已同步"}</span>
    </div>
  );
}

function PositionSummary({ data }) {
  return (
    <div className="factor-combo-position-summary">
      <SummaryCell label="进行中" value={data?.openCount ?? 0} />
      <SummaryCell label="历史" value={data?.settledCount ?? 0} />
      <SummaryCell label="当前组合" value={data?.currentFactorCount ?? 0} />
      <SummaryCell label="合计PNL" value={signedMoney(Number(data?.totalPnl || 0))} />
    </div>
  );
}

function SummaryCell({ label, value }) {
  return (
    <span>
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function PositionTable({ events, factorName }) {
  if (!events.length) {
    return <p className="factor-combo-position-empty">暂无该周期组合策略持仓记录</p>;
  }
  return (
    <div className="factor-combo-position-table-wrap">
      <table className="factors-table factor-combo-position-table">
        <thead>
          <tr>
            <th>状态</th>
            <th>方向</th>
            <th>金额</th>
            <th>入场</th>
            <th>结算/PNL</th>
            <th>组合</th>
            <th>时间</th>
          </tr>
        </thead>
        <tbody>{events.map((event) => <PositionRow key={event.id} event={event} factorName={factorName} />)}</tbody>
      </table>
    </div>
  );
}

function PositionRow({ event, factorName }) {
  const pnl = settledExpectedProfitUsdt(event);
  const current = event.aiHighWinrateRule && event.aiHighWinrateRule === factorName;
  return (
    <tr>
      <td><StatusPill status={event.status} externalStatus={event.externalStatus} /></td>
      <td className={event.orderSide === "BUY" ? "value-up" : "value-down"}>{sideText(event.orderSide)}</td>
      <td>{money(event.orderQty)}</td>
      <td>{price(event.strikeValue)}</td>
      <td>{pnl == null ? settlementPrice(event) : `${signedMoney(pnl)} U`}</td>
      <td>{current ? "当前" : event.aiHighWinrateRule ? "历史" : "未标记"}</td>
      <td>{formatTime(event.startTime)}</td>
    </tr>
  );
}

function StatusPill({ status, externalStatus }) {
  const settled = status === "SETTLED";
  return (
    <span className={`factor-combo-position-status ${settled ? "settled" : "open"}`}>
      {settled ? "已结算" : externalStatus === "SIMULATED" ? "模拟中" : "进行中"}
    </span>
  );
}

function sideText(side) {
  if (side === "BUY") return "看涨";
  if (side === "SELL") return "看跌";
  return "未下单";
}

function money(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(2)} U` : "--";
}

function price(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "--";
}

function settlementPrice(event) {
  return event.settlementPrice == null ? "--" : price(event.settlementPrice);
}

function signedMoney(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function formatTime(value) {
  if (!value) return "--";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return "--";
  return dt.toLocaleString("zh-CN", { hour12: false });
}

function isCanceled(error, signal) {
  return signal.aborted || error?.code === "ERR_CANCELED" || error?.name === "CanceledError";
}
