import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchPaperLiveCandidates,
  runPaperLiveDailyLoop,
} from "../api/factorCombinations";
import "./FactorPaperLiveCandidatePool.css";

const CANDIDATE_LIMIT = 6;
const FAILURE_LIMIT = 5;
const LOG_LIMIT = 6;
const EMPTY_LABEL = "—";

export default function FactorPaperLiveCandidatePool({ symbol, duration }) {
  const { loading, report, status, runDailyLoop } = usePaperLiveCandidatePool(symbol, duration);
  const counts = candidateCounts(report);

  return (
    <section className="paper-live-panel">
      <header className="paper-live-head">
        <div>
          <span className="section-kicker">Paper-live 候选池</span>
          <h3>{symbol} / {duration}</h3>
        </div>
        <div className="paper-live-actions">
          <span>{status}</span>
          <button type="button" className="factors-btn-outline" disabled={loading} onClick={runDailyLoop}>
            {loading ? "执行中…" : "运行日闭环"}
          </button>
        </div>
      </header>
      <div className="paper-live-summary" aria-label="paper-live 候选池概览">
        <SummaryMetric label="稳定" value={counts.stable} tone="stable" />
        <SummaryMetric label="观察中" value={counts.collecting} tone="collecting" />
        <SummaryMetric label="失败" value={counts.failed} tone="failed" />
        <SummaryMetric label="预测失败" value={counts.predictionFailures} tone="failed" />
      </div>
      <CandidateColumns report={report} />
      <EvidencePanel report={report} />
    </section>
  );
}

function usePaperLiveCandidatePool(symbol, duration) {
  const [state, setState] = useState(() => ({
    loading: false,
    report: null,
    status: "等待加载候选池",
  }));
  const normalizedSymbol = useMemo(() => String(symbol || "").trim().toUpperCase(), [symbol]);

  const load = useCallback(async (signal) => {
    if (!normalizedSymbol || !duration) {
      setState({ loading: false, report: null, status: "交易对或周期无效" });
      return;
    }
    setState((current) => ({ ...current, loading: true, status: "读取 paper-live 候选池…" }));
    try {
      const report = await fetchPaperLiveCandidates(normalizedSymbol, duration, { signal });
      if (signal?.aborted) return;
      setState({ loading: false, report, status: reportStatus(report) });
    } catch (error) {
      if (signal?.aborted) return;
      setState((current) => ({ ...current, loading: false, status: `候选池失败：${errorMessage(error)}` }));
    }
  }, [duration, normalizedSymbol]);

  useEffect(() => {
    const ac = new AbortController();
    void load(ac.signal);
    return () => ac.abort();
  }, [load]);

  const runDaily = useCallback(async () => {
    if (!normalizedSymbol || !duration) return;
    setState((current) => ({ ...current, loading: true, status: "执行 paper-live 日闭环…" }));
    try {
      const result = await runPaperLiveDailyLoop(normalizedSymbol, duration);
      const first = Array.isArray(result.results) ? result.results[0] : null;
      setState({
        loading: false,
        report: first?.candidatePool || null,
        status: dailyLoopStatus(result, first),
      });
    } catch (error) {
      setState((current) => ({ ...current, loading: false, status: `日闭环失败：${errorMessage(error)}` }));
    }
  }, [duration, normalizedSymbol]);

  return {
    loading: state.loading,
    report: state.report,
    status: state.status,
    runDailyLoop: runDaily,
  };
}

function CandidateColumns({ report }) {
  const groups = [
    { key: "stable", title: "稳定", rows: report?.stable || [], empty: "暂无稳定候选" },
    { key: "collecting", title: "观察中", rows: report?.collecting || [], empty: "暂无观察候选" },
    { key: "failed", title: "失败", rows: report?.failed || [], empty: "暂无失败候选" },
  ];
  return (
    <div className="paper-live-columns">
      {groups.map((group) => (
        <section key={group.key} className={`paper-live-column is-${group.key}`}>
          <header>
            <h4>{group.title}</h4>
            <span>{group.rows.length}</span>
          </header>
          <div className="paper-live-cards">
            {group.rows.slice(0, CANDIDATE_LIMIT).map((row) => (
              <CandidateCard key={`${group.key}-${row.candidateKey || row.modelVersion || row.factorName}`} row={row} />
            ))}
            {!group.rows.length ? <p className="paper-live-empty">{group.empty}</p> : null}
            {group.rows.length > CANDIDATE_LIMIT ? (
              <p className="paper-live-more">另有 {group.rows.length - CANDIDATE_LIMIT} 个未展开</p>
            ) : null}
          </div>
        </section>
      ))}
    </div>
  );
}

function CandidateCard({ row }) {
  return (
    <article className={`paper-live-card ${statusClass(row.paperLiveStatus || row.status)}`}>
      <div className="paper-live-card-top">
        <strong title={candidateName(row)}>{candidateName(row)}</strong>
        <span>{candidateTypeLabel(row)}</span>
      </div>
      <p>{reasonLabel(row.reason)}</p>
      <div className="paper-live-card-metrics">
        <Metric label="纸盘胜率" value={formatPct(row.paperLiveWinRate)} strong />
        <Metric label="样本" value={row.paperLiveSampleCount ?? 0} />
        <Metric label="回测" value={formatPct(row.backtestWinRate)} />
        <Metric label="OOS" value={formatPct(row.oosWinRate)} />
        <Metric label="PF" value={formatNum(row.metrics?.profitFactor, 2)} />
        <Metric label="均收益" value={formatPct(row.metrics?.avgReturn)} />
      </div>
      <div className="paper-live-card-tags">
        <small>数据: {row.dataFreshnessStatus || EMPTY_LABEL}</small>
        <small>特征: {row.missingFeatureStatus || EMPTY_LABEL}</small>
      </div>
    </article>
  );
}

function EvidencePanel({ report }) {
  const failures = Array.isArray(report?.predictionFailures) ? report.predictionFailures.slice(0, FAILURE_LIMIT) : [];
  const logs = Array.isArray(report?.stageLogs) ? report.stageLogs.slice(0, LOG_LIMIT) : [];
  const avoid = Array.isArray(report?.avoidNextSearch) ? report.avoidNextSearch.slice(0, FAILURE_LIMIT) : [];
  return (
    <div className="paper-live-evidence">
      <EvidenceList title="预测失败" rows={failures} empty="暂无预测失败" render={failureRow} />
      <EvidenceList title="阶段日志" rows={logs} empty="暂无阶段日志" render={stageLogRow} />
      <EvidenceList title="下轮避开" rows={avoid} empty="暂无避开项" render={avoidRow} />
    </div>
  );
}

function EvidenceList({ title, rows, empty, render }) {
  return (
    <section className="paper-live-evidence-card">
      <header>
        <h4>{title}</h4>
        <span>{rows.length}</span>
      </header>
      {rows.length ? (
        <ul>
          {rows.map((row, index) => (
            <li key={`${title}-${index}`}>{render(row)}</li>
          ))}
        </ul>
      ) : (
        <p>{empty}</p>
      )}
    </section>
  );
}

function SummaryMetric({ label, value, tone }) {
  return (
    <span className={`paper-live-summary-metric is-${tone}`}>
      <b>{value ?? 0}</b>
      <small>{label}</small>
    </span>
  );
}

function Metric({ label, value, strong = false }) {
  return (
    <span className={strong ? "is-strong" : ""}>
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function failureRow(row) {
  return (
    <>
      <strong>{row.candidateKey || row.strategyKey || EMPTY_LABEL}</strong>
      <span>{row.stage || EMPTY_LABEL} · {row.reason || EMPTY_LABEL}</span>
    </>
  );
}

function stageLogRow(row) {
  return (
    <>
      <strong>{row.stage || EMPTY_LABEL}</strong>
      <span>{row.status || EMPTY_LABEL} · {row.reason || row.createdAt || row.loggedAt || EMPTY_LABEL}</span>
    </>
  );
}

function avoidRow(row) {
  return (
    <>
      <strong>{row.candidateKey || EMPTY_LABEL}</strong>
      <span>{row.reason || EMPTY_LABEL}</span>
    </>
  );
}

function candidateCounts(report) {
  return {
    stable: report?.stable?.length || 0,
    collecting: report?.collecting?.length || 0,
    failed: report?.failed?.length || 0,
    predictionFailures: report?.predictionFailures?.length || 0,
  };
}

function reportStatus(report) {
  if (!report) return "候选池未返回数据";
  const counts = candidateCounts(report);
  const updated = report.updatedAt ? ` · ${formatDate(report.updatedAt)}` : "";
  return `稳定 ${counts.stable} · 观察 ${counts.collecting} · 失败 ${counts.failed}${updated}`;
}

function dailyLoopStatus(result, first) {
  const status = result?.status || first?.status || "unknown";
  const checklist = Array.isArray(first?.dailyChecklist) ? first.dailyChecklist.length : 0;
  return `日闭环 ${status} · 检查项 ${checklist}`;
}

function candidateName(row) {
  return row.modelFamily || row.factorName || row.candidateKey || row.strategyKey || EMPTY_LABEL;
}

function candidateTypeLabel(row) {
  if (row.candidateType === "model") return row.modelVersion || "模型";
  if (row.candidateType === "factor_combo") return "组合";
  return "因子";
}

function reasonLabel(reason) {
  const labels = {
    insufficient_settled_samples: "已结算样本不足",
    paper_live_win_rate_below_target: "纸盘胜率低于目标",
    paper_live_profit_factor_below_target: "纸盘盈亏比低于目标",
    paper_live_avg_return_below_target: "纸盘均收益低于目标",
    stable_paper_live_target_met: "纸盘稳定达标",
    consecutive_losses: "连续亏损触发失败",
    invalid_data_leakage: "数据泄漏",
  };
  return labels[reason] || reason || EMPTY_LABEL;
}

function statusClass(status) {
  if (status === "paper_stable") return "is-stable";
  if (status === "paper_failed" || status === "invalid_data_leakage") return "is-failed";
  return "is-collecting";
}

function formatPct(value) {
  if (value == null || Number.isNaN(Number(value))) return EMPTY_LABEL;
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNum(value, digits) {
  if (value == null || Number.isNaN(Number(value))) return EMPTY_LABEL;
  return Number(value).toFixed(digits);
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || EMPTY_LABEL);
  return date.toLocaleString();
}

function errorMessage(error) {
  return error?.response?.data?.detail || error?.message || "unknown_error";
}
