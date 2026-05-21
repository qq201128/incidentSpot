import { factorLabel } from "../utils/factorLearningLabels";
import "./FactorLearningStatusBoxes.css";

export default function FactorLearningStatusBoxes({ memory, refreshing = false, onRefreshLocal }) {
  return (
    <div className="factor-learning-grid factor-learning-status-grid">
      <AgentLibraryBox library={memory?.agentMinedFactorLibrary || {}} />
      <LibraryBox library={memory?.minedFactorLibrary || {}} />
      <PromotionBox promotion={memory?.agentCandidatePromotion || {}} />
      <MonitorBox monitoring={memory?.monitoring || {}} />
      <IssueBox monitoring={memory?.monitoring || {}} />
      <SolutionBox
        monitoring={memory?.monitoring || {}}
        refreshing={refreshing}
        onRefreshLocal={onRefreshLocal}
      />
    </div>
  );
}

function AgentLibraryBox({ library }) {
  const rows = Array.isArray(library.factors) ? library.factors : [];
  const candidateTotal = library.candidateTotal ?? library.total ?? rows.length;
  return (
    <section
      className="factor-learning-box"
      title="联网Agent提出的单因子候选，经过公式物化和单因子回测达标后写入。"
    >
      <BoxTitle title="Agent单因子库" count={library.total ?? 0} />
      <ul>
        <MetricRow label="总候选" value={candidateTotal} />
        <MetricRow label="达标" value={library.total ?? 0} />
        <MetricRow label="未入库" value={library.rejectedTotal ?? "—"} />
        {rows.map((factor) => (
          <li key={factor.factorName}>
            <strong>{factor.factorDisplayName || factorLabel(factor.factorName)}</strong>
            <span>{formatPct(factor.metrics?.winRate, 1)}</span>
          </li>
        ))}
      </ul>
      {!rows.length ? <p className="factor-learning-empty small">暂无达标入库单因子</p> : null}
    </section>
  );
}

function LibraryBox({ library }) {
  const rows = Array.isArray(library.factors) ? library.factors : [];
  return (
    <section
      className="factor-learning-box"
      title="已通过组合排名筛选并回灌的组合。列表右侧为该组合的历史胜率。"
    >
      <BoxTitle title="组合回灌库" count={library.total ?? 0} />
      <ul>
        {rows.map((factor) => (
          <li key={factor.factorName}>
            <strong>{factor.factorDisplayName || factorLabel(factor.factorName)}</strong>
            <span>{formatPct(factor.metrics?.winRate, 1)}</span>
          </li>
        ))}
      </ul>
      {!rows.length ? <p className="factor-learning-empty small">暂无入库组合</p> : null}
    </section>
  );
}

function PromotionBox({ promotion }) {
  const records = Array.isArray(promotion.records) ? promotion.records : [];
  const failed = records.filter((record) => record.status !== "promoted").length;
  return (
    <section className="factor-learning-box" title="Agent候选的公式物化、回测和入库结果。">
      <BoxTitle title="Agent候选结果" count={promotion.promoted ?? 0} />
      <ul>
        <MetricRow label="候选" value={promotion.candidateCount ?? "—"} />
        <MetricRow label="本轮入库" value={promotion.promoted ?? "—"} />
        <MetricRow label="未入库" value={records.length ? failed : "—"} />
        {records.slice(0, 3).map((record) => (
          <li key={record.factorName || record.formula}>
            <strong>{record.displayName || record.factorName}</strong>
            <span>{agentRecordStatus(record)}</span>
          </li>
        ))}
      </ul>
      {!records.length ? <p className="factor-learning-empty small">暂无候选回灌结果</p> : null}
    </section>
  );
}

function agentRecordStatus(record) {
  if (record.status === "promoted") return "本轮入库";
  if (record.status === "duplicate_existing") return "重复";
  if (record.status === "failed") return "失败";
  if (record.status === "rejected_metrics") {
    const winRate = formatPct(record.metrics?.winRate, 1);
    const profitFactor = record.metrics?.profitFactor ?? "—";
    return `未达标 ${winRate} / PF ${profitFactor}`;
  }
  return record.status || "—";
}

function MonitorBox({ monitoring }) {
  const metrics = monitoring.metrics || {};
  return (
    <section
      className="factor-learning-box"
      title="基于已结算多因子模拟预测的健康度。样本不足 10 为「样本少」；否则有告警为「预警」。"
    >
      <BoxTitle title="实盘模拟监控" count={statusText(monitoring.status)} />
      <ul>
        <MetricRow
          label="样本"
          value={metrics.sampleCount ?? "—"}
          hint="用于监控的最近已结算多因子预测条数（最多 200）。"
        />
        <MetricRow
          label="成功率"
          value={formatPct(metrics.predictionSuccessRate, 1)}
          hint="全部已结算样本中，预测是否判对的比例（与系统内对错标记一致）。"
        />
        <MetricRow
          label="候选成功率"
          value={formatPct(metrics.qualityPassedSuccessRate, 1)}
          hint="仅统计已通过质量过滤的模拟单中判对的比例。"
        />
        <MetricRow
          label="连续亏损"
          value={metrics.latestConsecutiveLosses ?? "—"}
          hint="从最新一条往历史数，连续预测错误的条数。"
        />
      </ul>
    </section>
  );
}

function IssueBox({ monitoring }) {
  const issues = Array.isArray(monitoring.issues) ? monitoring.issues : [];
  return (
    <section className="factor-learning-box" title="右侧高/中/低为严重级别；与成功率、候选成功率、连续亏损等阈值相关。">
      <BoxTitle title="低胜率告警" count={issues.length} />
      <ul>
        {issues.map((issue) => (
          <li key={issue.code}>
            <strong>{issue.message || issue.code}</strong>
            <span>{severityText(issue.severity)}</span>
          </li>
        ))}
      </ul>
      {!issues.length ? <p className="factor-learning-empty small">暂无告警</p> : null}
    </section>
  );
}

function SolutionBox({ monitoring, refreshing, onRefreshLocal }) {
  const rows = Array.isArray(monitoring.solutions)
    ? monitoring.solutions.map(normalizeSolution)
    : [];
  return (
    <section className="factor-learning-box" title="与告警关联的处理建议；会改变候选逻辑的项目需人工确认。">
      <BoxTitle title="解决方案" count={rows.length} />
      <ul>
        {rows.map((solution) => (
          <li className="factor-learning-solution-row" key={solution.key}>
            <strong>{solution.text}</strong>
            <SolutionControl
              solution={solution}
              refreshing={refreshing}
              onRefreshLocal={onRefreshLocal}
            />
          </li>
        ))}
      </ul>
      {!rows.length ? <p className="factor-learning-empty small">暂无处理项</p> : null}
    </section>
  );
}

function SolutionControl({ solution, refreshing, onRefreshLocal }) {
  if (solution.action === "refresh_learning") {
    return (
      <button
        className="factor-learning-solution-button"
        type="button"
        disabled={refreshing}
        onClick={() => runSolutionAction(solution, onRefreshLocal)}
      >
        {refreshing ? "执行中" : solution.actionLabel}
      </button>
    );
  }
  return <span className="factor-learning-solution-badge">{solution.actionLabel}</span>;
}

function runSolutionAction(solution, onRefreshLocal) {
  if (solution.requiresConfirmation && !window.confirm(solution.text)) return;
  if (solution.action === "refresh_learning") onRefreshLocal?.();
}

function normalizeSolution(solution) {
  if (typeof solution === "string") {
    const inferred = legacySolutionAction(solution);
    return {
      key: solution,
      text: solution,
      action: inferred.action,
      actionLabel: inferred.actionLabel,
      requiresConfirmation: inferred.requiresConfirmation,
    };
  }
  return {
    key: solution.key || solution.text,
    text: solution.text || "",
    action: solution.action || null,
    actionLabel: solution.actionLabel || (solution.action ? "执行" : "需人工确认"),
    requiresConfirmation: Boolean(solution.requiresConfirmation),
  };
}

function legacySolutionAction(text) {
  if (text.includes("本地复盘")) {
    return {
      action: "refresh_learning",
      actionLabel: "本地复盘",
      requiresConfirmation: true,
    };
  }
  return { action: null, actionLabel: "需人工确认", requiresConfirmation: false };
}

function MetricRow({ label, value, hint }) {
  return (
    <li title={hint}>
      <strong>{label}</strong>
      <span>{value}</span>
    </li>
  );
}

function BoxTitle({ title, count }) {
  return (
    <div className="factor-learning-title compact">
      <h3>{title}</h3>
      <span>{count}</span>
    </div>
  );
}

function statusText(status) {
  if (status === "healthy") return "正常";
  if (status === "warning") return "预警";
  if (status === "insufficient_data") return "样本少";
  return "—";
}

function severityText(severity) {
  if (severity === "high") return "高";
  if (severity === "medium") return "中";
  if (severity === "low") return "低";
  return "—";
}

function formatPct(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
