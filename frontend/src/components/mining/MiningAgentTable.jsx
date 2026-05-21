import { operatorTraceLabel } from "../../utils/factorLearningLabels";
import { formatTime } from "./miningFormatters";

const STATUS_CLASS = {
  pending_backtest: "is-pending",
  materialized: "is-blue",
  promoted: "is-green",
  rejected_metrics: "is-red",
  failed: "is-red",
  duplicate: "is-muted",
};

export default function MiningAgentTable({ rows }) {
  const total = rows.length;
  return (
    <section className="mining-agent-section">
      <header className="mining-section-head">
        <h2>Agent 单因子候选想法 {total} 项</h2>
      </header>
      <div className="mining-agent-table-wrap">
        <table className="mining-agent-table">
          <thead>
            <tr>
              <th>因子名称</th>
              <th>算子轨迹</th>
              <th>公式提示</th>
              <th>动机 / 直觉</th>
              <th>验证状态</th>
              <th>来源</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>
                  <strong>{row.factorName}</strong>
                </td>
                <td>{operatorTraceLabel(row.operatorTrace).join(" · ") || "—"}</td>
                <td>
                  <code>{row.formulaHint || "—"}</code>
                </td>
                <td className="mining-agent-rationale">{row.rationale || "—"}</td>
                <td>
                  <span className={`mining-agent-status ${STATUS_CLASS[row.validationStatusKey] || ""}`}>
                    {row.validationStatus}
                  </span>
                </td>
                <td>{row.source}</td>
                <td>{formatTime(row.createdAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!total ? <p className="mining-empty">暂无 Agent 候选，请先运行 Agent 复盘</p> : null}
      </div>
      {total ? (
        <footer className="mining-agent-footer">查看全部 {total} 项候选 →</footer>
      ) : null}
    </section>
  );
}
