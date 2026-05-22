import { formatTime } from "./miningFormatters";

export default function MiningPageHeader({
  symbol,
  duration,
  durationOptions,
  onSymbolChange,
  onDurationChange,
  header,
  updatedAt,
  onReload,
  reloading,
}) {
  return (
    <header className="mining-topbar">
      <div className="mining-topbar-main">
        <h1>因子学习与候选挖掘</h1>
        <p>从亏损样本、Agent 筛选、算子库和 LSTM 影子信号中提炼入库因子</p>
      </div>
      <div className="mining-topbar-right">
        <div className="mining-topbar-controls">
          <label>
            <span>交易对</span>
            <select value={symbol} onChange={(e) => onSymbolChange(e.target.value)}>
              <option value="BTCUSDT">BTCUSDT</option>
              <option value="ETHUSDT">ETHUSDT</option>
            </select>
          </label>
          <label>
            <span>观测周期</span>
            <select value={duration} onChange={(e) => onDurationChange(e.target.value)}>
              {durationOptions.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="mining-topbar-status">
          <StatusChip tone={header?.localReplayStatus} dot>
            {header?.localReplayLabel || "—"}
          </StatusChip>
          <StatusChip tone="info" title="最近一次联网 Agent 挖掘提出的单因子研究想法">
            本轮想法 {header?.agentIdeaCount ?? header?.agentCandidateCount ?? 0} 项
          </StatusChip>
          <StatusChip tone="info" title="当前交易对与周期在 Agent 入库库中的记录数">
            库内 {header?.agentLibraryPairCount ?? 0} 项
          </StatusChip>
          <StatusChip tone={header?.pendingVerificationCount ? "warn" : "info"}>
            待验证 {header?.pendingVerificationCount ?? 0} 项
          </StatusChip>
          {header?.agentModel ? (
            <StatusChip tone="info" title="最近一次或当前配置的联网 LLM 模型（见 SILICONFLOW_MODEL）">
              模型 {header.agentModel}
            </StatusChip>
          ) : null}
          {header?.agentReviewedAt ? (
            <StatusChip tone="info" title="最近一次联网 Agent 完成并写回候选想法的时间">
              Agent {formatTime(header.agentReviewedAt)}
            </StatusChip>
          ) : null}
          <button type="button" className="mining-refresh-time" onClick={onReload} disabled={reloading}>
            更新时间：{formatTime(updatedAt)}
            <span aria-hidden>↻</span>
          </button>
        </div>
      </div>
    </header>
  );
}

function StatusChip({ tone = "info", dot = false, children }) {
  return (
    <span className={`mining-status-chip is-${tone}`}>
      {dot ? <i className="mining-status-dot" aria-hidden /> : null}
      {children}
    </span>
  );
}
