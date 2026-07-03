import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { fetchEventsPage } from "../api/eventsClient";
import { eventDurationMinutesFromWindow } from "../utils/eventDuration";
import { settledExpectedProfitUsdt } from "../utils/eventSettlement";
import { factorLabel } from "../utils/factorLearningLabels";
import { eventBacktestWinRatePercent } from "../utils/eventAiMetrics";
import { simulationTypeLabel, strategyLabel } from "../utils/strategyLabels";

const PAGE_SIZE = 8;
const EVENT_TAB = "events";
const SEARCH_DEBOUNCE_MS = 280;

function EventRecordsTable({
  symbol = "",
  compact = false,
  page,
  onPageChange,
  reloadKey = 0,
}) {
  const [scope, setScope] = useState("ALL");
  const [items, setItems] = useState([]);
  const [query, setQuery] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [total, setTotal] = useState(0);
  const [pageCount, setPageCount] = useState(1);
  const debouncedQuery = useDebouncedValue(query, SEARCH_DEBOUNCE_MS);
  const effectiveSymbol = scope === "ALL" ? "" : scope;

  const loadRecords = useCallback(async () => {
    try {
      const data = await fetchEventsPage({
        symbol: effectiveSymbol,
        page,
        pageSize: PAGE_SIZE,
        q: compact ? undefined : debouncedQuery.trim(),
      });
      const serverPage = Math.max(1, Number(data.page) || 1);
      setErrorMessage("");
      setItems(Array.isArray(data.items) ? data.items : []);
      setTotal(Number(data.total) || 0);
      setPageCount(Math.max(1, Number(data.pageCount) || 1));
      if (serverPage !== page) onPageChange?.(serverPage);
    } catch (err) {
      console.error("事件记录加载失败", err);
      setErrorMessage(`事件记录加载失败：${errorMessageFrom(err)}`);
      setItems([]);
      setTotal(0);
      setPageCount(1);
    }
  }, [compact, debouncedQuery, effectiveSymbol, onPageChange, page]);

  useEffect(() => {
    if (compact) return;
    onPageChange?.(1);
  }, [compact, debouncedQuery, effectiveSymbol, onPageChange]);

  useEffect(() => {
    void loadRecords();
  }, [loadRecords, reloadKey]);

  useEffect(() => {
    if (page > pageCount) onPageChange?.(pageCount);
  }, [onPageChange, page, pageCount]);

  const view = useMemo(
    () => buildView(items, compact),
    [compact, items],
  );

  const sectionClass = compact
    ? "event-records event-records--compact"
    : "event-records";

  return (
    <section className={sectionClass}>
      <RecordTabs
        compact={compact}
        scopeLabel={scopeText(scope)}
        scope={scope}
        onScopeChange={setScope}
        page={page}
        pageCount={pageCount}
        query={query}
        errorMessage={errorMessage}
        onQueryChange={setQuery}
        total={total}
      />
      <div className="event-records-table" role="table" aria-label={view.title}>
        <div className="event-records-rows">
          <RecordHeader labels={view.labels} viewKey={view.key} />
          {view.rows.map((row) => (
            <RecordRow key={row.key} cells={row.cells} viewKey={view.key} />
          ))}
        </div>
        {errorMessage ? <p className="event-records-error">{errorMessage}</p> : null}
        {!total && !errorMessage ? <p className="event-records-empty">{view.emptyText}</p> : null}
      </div>
      {total ? (
        <RecordsPagination page={page} pageCount={pageCount} total={total} onPageChange={onPageChange} />
      ) : null}
    </section>
  );
}

function RecordTabs({
  compact,
  errorMessage,
  onScopeChange,
  onQueryChange,
  page,
  pageCount,
  query,
  scope,
  scopeLabel,
  total,
}) {
  const meta = errorMessage
    ? "加载失败"
    : total
      ? `${scopeLabel} · 共 ${total} 条 · ${page}/${pageCount} 页`
      : `${scopeLabel} · 暂无记录`;

  if (compact) {
    return (
      <div className="event-records-head">
        <h2>事件列表</h2>
        <span>{meta}</span>
      </div>
    );
  }
  return (
    <div className="event-records-head">
      <h2>事件合约记录</h2>
      <label className="event-records-scope">
        <span className="sr-only">交易对筛选</span>
        <select value={scope} onChange={(event) => onScopeChange?.(event.target.value)}>
          <option value="ALL">全部交易对</option>
          <option value="BTCUSDT">BTCUSDT</option>
          <option value="ETHUSDT">ETHUSDT</option>
        </select>
      </label>
      <label className="event-records-search">
        <span className="sr-only">搜索事件记录</span>
        <input
          value={query}
          onChange={(event) => onQueryChange?.(event.target.value)}
          placeholder="搜索事件ID/策略/状态"
        />
      </label>
      <span className="event-records-meta">{meta}</span>
    </div>
  );
}

function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [delayMs, value]);
  return debounced;
}

function errorMessageFrom(error) {
  return error?.response?.data?.detail || error?.message || "unknown_error";
}

function scopeText(scope) {
  if (scope === "BTCUSDT") return "BTCUSDT";
  if (scope === "ETHUSDT") return "ETHUSDT";
  return "全部交易对";
}

function RecordsPagination({ page, pageCount, total, onPageChange }) {
  return (
    <nav className="event-records-pagination" aria-label="事件记录分页">
      <span className="event-records-page-total">共 {total} 条</span>
      <div className="event-records-page-actions">
        <button type="button" disabled={page <= 1} onClick={() => onPageChange?.(1)} aria-label="首页">
          «
        </button>
        <button type="button" disabled={page <= 1} onClick={() => onPageChange?.(page - 1)} aria-label="上一页">
          ‹
        </button>
        <strong>
          {page} / {pageCount}
        </strong>
        <button
          type="button"
          disabled={page >= pageCount}
          onClick={() => onPageChange?.(page + 1)}
          aria-label="下一页"
        >
          ›
        </button>
        <button
          type="button"
          disabled={page >= pageCount}
          onClick={() => onPageChange?.(pageCount)}
          aria-label="末页"
        >
          »
        </button>
      </div>
    </nav>
  );
}

function RecordHeader({ labels, viewKey }) {
  return (
    <div className={`event-records-row event-records-row--header event-records-row--${viewKey}`} role="row">
      {labels.map((label) => (
        <span key={label} role="columnheader" title={label}>
          {label}
        </span>
      ))}
    </div>
  );
}

function RecordRow({ cells, viewKey }) {
  return (
    <div className={`event-records-row event-records-row--${viewKey}`} role="row">
      {cells.map((cell) => (
        <span key={cell.key} className={cell.className || ""} role="cell" title={cell.title || String(cell.value)}>
          {cell.value}
        </span>
      ))}
    </div>
  );
}

function buildView(events, compact) {
  const source = Array.isArray(events) ? events : [];
  if (compact) {
    return view("compact", compactLabels(), source.map(compactRow), "暂无事件", "事件列表");
  }
  return view("events", eventLabels(), source.map(eventRow), "暂无事件", "事件合约记录");
}

function view(key, labels, rows, emptyText, title) {
  return { key, labels, rows, emptyText, title };
}

function eventRow(event) {
  return row(event.id, [
    text("id", eventCode(event.id)),
    text("symbol", event.symbol),
    sideCell(event.orderSide),
    text("start", timeHm(event.startTime)),
    text("strike", price(event.strikeValue)),
    text("duration", duration(event)),
    text("end", timeHm(event.endTime)),
    text("settle", nullablePrice(event.settlementPrice)),
    statusCell(event.status),
    modeCell(event.externalStatus),
    resultCell(event),
    pnlCell(event),
    backtestWinRateCell(event),
    text("rule", ruleName(event)),
  ]);
}

function compactRow(event) {
  return row(event.id, [
    text("id", eventCode(event.id)),
    sideCell(event.orderSide),
    text("strike", price(event.strikeValue)),
    text("duration", duration(event)),
    statusCell(event.status),
    modeCell(event.externalStatus),
    resultCell(event),
  ]);
}

function row(key, cells) {
  return { key, cells };
}

function text(key, value, className = "", title = "") {
  return { key, value, className, title };
}

function sideCell(side) {
  if (side === "BUY") return text("side", "BUY", "value-up");
  if (side === "SELL") return text("side", "SELL", "value-down");
  return text("side", "—");
}

function statusCell(status) {
  const ok = status === "OPEN" || status === "SETTLED";
  return text("status", status || "—", ok ? "tag tag--green" : "tag tag--red");
}

function modeCell(externalStatus) {
  const live = externalStatus && externalStatus !== "SIMULATED";
  return text("mode", live ? "LIVE" : "SIM", live ? "tag tag--orange" : "tag tag--blue");
}

function resultCell(event) {
  if (event.status !== "SETTLED") return text("result", "—");
  const win = Number(event.aiPredictionCorrect) === 1;
  return text("result", win ? "WIN" : "LOSE", win ? "tag tag--green" : "tag tag--red");
}

function pnlCell(event) {
  const pnl = settledExpectedProfitUsdt(event);
  if (pnl == null) return text("pnl", "—");
  return text("pnl", signed(pnl), pnl >= 0 ? "value-up" : "value-down");
}

function backtestWinRateCell(event) {
  const pct = eventBacktestWinRatePercent(event);
  if (pct == null) return text("backtestWinRate", "—");
  return text("backtestWinRate", `${pct}%`);
}

function eventCode(id) {
  return `EVT-${String(id).padStart(6, "0")}`;
}

function duration(event) {
  const minutes = eventDurationMinutesFromWindow(event);
  return minutes == null ? "—" : minutes === 1440 ? "1d" : `${minutes}m`;
}

function price(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—";
}

function nullablePrice(value) {
  return value == null ? "—" : price(value);
}

function timeHm(value) {
  if (!value) return "—";
  const dt = new Date(value);
  if (!Number.isFinite(dt.getTime())) return "—";
  return dt.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function signed(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function ruleName(event) {
  const sim = simulationTypeLabel(event.strategyKey, event.aiHighWinrateRule);
  if (sim) return sim;
  if (event.strategyKey) return strategyLabel(event.strategyKey);
  if (event.aiHighWinrateRule) return factorLabel(event.aiHighWinrateRule);
  return event.title || "手动";
}

const eventLabels = () => [
  "事件ID",
  "交易对",
  "方向",
  "入场",
  "入场价",
  "周期",
  "结算",
  "结算价",
  "状态",
  "模式",
  "结果",
  "PnL",
  "回测胜率",
  "类型",
];
const compactLabels = () => ["事件ID", "方向", "入场价", "周期", "状态", "模式", "结果"];

export default memo(EventRecordsTable, (prev, next) =>
  prev.symbol === next.symbol &&
  prev.compact === next.compact &&
  prev.page === next.page &&
  prev.reloadKey === next.reloadKey &&
  prev.ensembleDuration === next.ensembleDuration &&
  prev.ensembleReloadKey === next.ensembleReloadKey
);
