export default function WorkbenchStatusBar({ latencyMs, summary }) {
  const latency = Number.isFinite(latencyMs) ? `${Math.round(latencyMs)}ms` : "—";
  return (
    <footer className="workbench-statusbar">
      <div>
        <span className="status-dot status-dot--muted" />
        数据来源: {summary?.dataSource || "Binance Index"}
      </div>
      <div className="statusbar-latency">延迟: {latency}</div>
      <div className="statusbar-risk">风险提示: 事件交易存在亏损风险，请确保充分理解后使用</div>
      <button type="button" className="statusbar-btn">⇩ 导出数据</button>
      <button type="button" className="statusbar-btn">⟳ 刷新数据</button>
    </footer>
  );
}
