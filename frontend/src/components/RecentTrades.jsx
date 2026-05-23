/**
 * 最新成交列表（数据由父组件 WebSocket /ws/agg-trades 推送）。
 * @param {{ symbol: string, trades: null | Array<{ price: number, qty: number, quoteQty: number, time: number, side: string }> }} props
 */
export default function RecentTrades({ symbol, trades }) {
  const rows = Array.isArray(trades) ? trades : [];
  const loading = trades === null;

  return (
    <section className="recent-trades" aria-label={`${symbol} 最新成交`}>
      <div className="recent-trades-tabs" role="tablist">
        <button type="button" className="recent-trades-tab recent-trades-tab--active" role="tab" aria-selected="true">
          最新成交
        </button>
        <button type="button" className="recent-trades-tab" role="tab" aria-disabled="true" title="敬请期待">
          市场异动
        </button>
      </div>
      <div className="recent-trades-table" role="grid">
        <div className="recent-trades-row recent-trades-row--header">
          <span className="recent-trades-col-price">价格 (USDT)</span>
          <span className="recent-trades-col-qty">数量 (USDT)</span>
          <span className="recent-trades-col-time">时间</span>
        </div>
        <div className="recent-trades-body">
          {loading ? (
            <p className="recent-trades-empty">加载中…</p>
          ) : rows.length === 0 ? (
            <p className="recent-trades-empty">暂无成交</p>
          ) : (
            rows.map((t, i) => (
              <div key={`${t.time}-${i}`} className="recent-trades-row" role="row">
                <span
                  className={`recent-trades-price recent-trades-price--${t.side === "sell" ? "sell" : "buy"}`}
                >
                  {fmtTradePrice(t.price, t.price)}
                </span>
                <span className="recent-trades-qty">{fmtQuoteUsdt(t.quoteQty)}</span>
                <span className="recent-trades-time">{fmtTimeHms(t.time)}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}

function fmtTradePrice(value, ref) {
  const s = String(fixedPrice(value, ref));
  if (s === "—") return s;
  const parts = s.split(".");
  const intPart = parts[0] ?? "";
  const dec = parts[1];
  const withSep = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return dec !== undefined ? `${withSep}.${dec}` : withSep;
}

function fixedPrice(value, ref) {
  const v = Number(value);
  const r = Number(ref);
  const decimals =
    !Number.isFinite(v) || v <= 0
      ? 4
      : v >= 1000 || (Number.isFinite(r) && r >= 1000)
        ? 2
        : v >= 1
          ? 4
          : 6;
  return Number.isFinite(v) ? v.toFixed(decimals) : "—";
}

function fmtQuoteUsdt(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(2)}K`;
  return v.toFixed(2);
}

function fmtTimeHms(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return "—";
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
