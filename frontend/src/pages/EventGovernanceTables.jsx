import { factorLabel } from "../utils/factorLearningLabels";
import { strategyLabel } from "../utils/strategyLabels";

const REASON_LABELS = {
  consecutive_losses: "连续亏损",
  insufficient_event_samples: "Event 样本不足",
  insufficient_settled_samples: "已结算样本不足",
  live_win_rate_below_target: "Event 胜率低于 62%",
  profit_factor_below_one: "Event 盈亏比低于 1.05",
  stable_live_target_met: "Event 指标达标",
  unsupported_strategy_key: "非模拟策略",
};

const STATUS_LABELS = {
  active: "正常",
  demoted: "需关注",
  collecting: "收集中",
  insufficient_samples: "样本不足",
};

export function SimulationObservationSection({ title, emptyText, demotion, watchlist, kind }) {
  return (
    <section className="event-gov-panel">
      <div className="event-gov-panel-head">
        <h2>{title}</h2>
        <span>观察模式 · 已评估 {demotion?.evaluatedCount ?? 0} 个 · 不自动 disable</span>
      </div>
      {watchlist.length === 0 ? (
        <p className="event-gov-empty">{emptyText}</p>
      ) : (
        <div className="event-gov-table-wrap">
          <table className="event-gov-table">
            <thead>
              <ObservationHeader kind={kind} />
            </thead>
            <tbody>
              {watchlist.map((row) => (
                <ObservationRow key={row.strategyKey} row={row} kind={kind} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function AllEvaluationsSection({ title, demotion, watchlistCount }) {
  const evaluations = demotion?.evaluations ?? [];
  if (evaluations.length <= watchlistCount) {
    return null;
  }
  return (
    <section className="event-gov-panel event-gov-panel-muted">
      <div className="event-gov-panel-head">
        <h2>{title}</h2>
        <span>{evaluations.length} 个</span>
      </div>
      <div className="event-gov-table-wrap">
        <table className="event-gov-table">
          <thead>
            <EvaluationHeader />
          </thead>
          <tbody>
            {evaluations.map((row) => (
              <EvaluationRow key={row.strategyKey} row={row} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ObservationHeader({ kind }) {
  return (
    <tr>
      <th>{kind === "single" ? "因子" : "策略"}</th>
      <th>原因</th>
      <th>样本</th>
      <th>胜率</th>
      <th>盈亏比</th>
      <StreakHeaders />
      <th>Event PnL (U)</th>
    </tr>
  );
}

function EvaluationHeader() {
  return (
    <tr>
      <th>名称</th>
      <th>状态</th>
      <th>样本</th>
      <th>胜率</th>
      <StreakHeaders />
      <th>Event PnL (U)</th>
    </tr>
  );
}

function StreakHeaders() {
  return (
    <>
      <th>最高连胜</th>
      <th>最高连亏</th>
      <th>目前连胜</th>
      <th>目前连亏</th>
    </>
  );
}

function ObservationRow({ row, kind }) {
  return (
    <tr>
      <td title={row.strategyKey}>{rowDisplayName(row, kind)}</td>
      <td>{REASON_LABELS[row.reason] ?? row.reason}</td>
      <td>{row.sampleCount ?? "—"}</td>
      <td>{formatRate(row.winRate)}</td>
      <td>{formatNumber(row.profitFactor)}</td>
      <StreakCells row={row} />
      <PnlCell value={row.totalPnlU} />
    </tr>
  );
}

function EvaluationRow({ row }) {
  return (
    <tr>
      <td title={row.strategyKey}>{rowDisplayName(row)}</td>
      <td>
        <StatusBadge status={row.status} />
      </td>
      <td>{row.sampleCount ?? "—"}</td>
      <td>{formatRate(row.winRate)}</td>
      <StreakCells row={row} />
      <PnlCell value={row.totalPnlU} />
    </tr>
  );
}

function StreakCells({ row }) {
  return (
    <>
      <td>{formatInteger(row.maxConsecutiveWins)}</td>
      <td>{formatInteger(row.maxConsecutiveLosses)}</td>
      <td>{formatInteger(row.currentConsecutiveWins)}</td>
      <td>{formatInteger(row.currentConsecutiveLosses ?? row.consecutiveLosses)}</td>
    </>
  );
}

function PnlCell({ value }) {
  return <td className={Number(value) < 0 ? "neg" : "pos"}>{formatNumber(value)}</td>;
}

function StatusBadge({ status }) {
  return <span className={`event-gov-badge status-${status}`}>{STATUS_LABELS[status] ?? status}</span>;
}

function rowDisplayName(row, kind = "auto") {
  if (row.displayRule) {
    return factorLabel(row.displayRule);
  }
  if (kind === "single" || String(row.strategyKey || "").startsWith("factor_candidate_signal_")) {
    return strategyLabel(row.strategyKey);
  }
  return strategyLabel(row.strategyKey);
}

function formatRate(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(2);
}

function formatInteger(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(0);
}
