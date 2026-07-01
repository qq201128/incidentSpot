import {
  EMPTY,
  EVIDENCE_TARGETS,
  TOP_ROW_LIMIT,
  candidateTypeLabel,
  formatGap,
  formatNum,
  formatPct,
  formatWindow,
  rollingWindowItems,
  visibleSettledRows,
} from "./researchDashboardData";
import {
  dataFreshnessLabel,
  missingFeatureLabel,
  reasonLabel,
  statusClass,
  statusLabel,
} from "./researchDashboardLabels";

const STABLE_STATUS = "paper_stable";

export function SettledSampleMatrix({
  liveToggleKey,
  loadError,
  loading,
  onLiveToggle,
  onPageChange,
  onPageSizeChange,
  pagination,
  reportLoaded,
  rows,
}) {
  const pageSize = pagination?.pageSize || TOP_ROW_LIMIT;
  const visibleRows = visibleSettledRows(rows, pageSize);
  const candidateCountText = matrixStatusText({
    loadError,
    loading,
    pagination,
    reportLoaded,
    visibleRowCount: visibleRows.length,
  });
  return (
    <section className="research-matrix">
      <header className="research-section-head">
        <div>
          <span className="section-kicker">Settled sample matrix</span>
          <h2>结算样本矩阵</h2>
        </div>
        <div className="research-matrix-head-actions">
          <small>{candidateCountText}</small>
          <PaginationControls
            loading={loading}
            onPageChange={onPageChange}
            onPageSizeChange={onPageSizeChange}
            pagination={pagination}
          />
        </div>
      </header>
      {rows.length ? (
        <MatrixTable
          liveToggleKey={liveToggleKey}
          onLiveToggle={onLiveToggle}
          rows={visibleRows}
        />
      ) : (
        <p className="research-empty" role={loadError ? "alert" : undefined}>
          {matrixEmptyText(loadError, loading, reportLoaded)}
        </p>
      )}
    </section>
  );
}

function MatrixTable({ liveToggleKey, onLiveToggle, rows }) {
  return (
    <div className="research-table-wrap">
      <table className="research-table">
        <thead>
          <tr>
            <th>候选</th>
            <th>状态</th>
            <th>实盘</th>
            <th>样本</th>
            <th>胜率</th>
            <th>近30</th>
            <th>近60</th>
            <th>近100</th>
            <th>滚动10</th>
            <th>PF</th>
            <th>均收益</th>
            <th>回测差</th>
            <th>连续亏损</th>
            <th>数据/特征</th>
            <th>原因</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <SettledRow
              key={row.rowKey}
              liveToggleKey={liveToggleKey}
              onLiveToggle={onLiveToggle}
              row={row}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PaginationControls({ loading, onPageChange, onPageSizeChange, pagination }) {
  if (!pagination) return null;
  const page = Number(pagination.page || 1);
  const pageSize = Number(pagination.pageSize || TOP_ROW_LIMIT);
  const totalPages = Math.max(Number(pagination.totalPages || 1), 1);
  return (
    <div className="research-pagination" aria-label="候选分页">
      <button
        disabled={loading || page <= 1}
        onClick={() => onPageChange?.(page - 1)}
        type="button"
      >
        上一页
      </button>
      <span>{page} / {totalPages}</span>
      <button
        disabled={loading || page >= totalPages}
        onClick={() => onPageChange?.(page + 1)}
        type="button"
      >
        下一页
      </button>
      <select
        aria-label="每页候选数"
        disabled={loading}
        onChange={(event) => onPageSizeChange?.(Number(event.target.value))}
        value={pageSize}
      >
        {[18, 30, 50, 100].map((value) => (
          <option key={value} value={value}>{value}/页</option>
        ))}
      </select>
    </div>
  );
}

function SettledRow({ liveToggleKey, onLiveToggle, row }) {
  const gapRisk = row.backtestGap != null && row.backtestGap >= EVIDENCE_TARGETS.backtestGapWarn;
  return (
    <tr>
      <td>
        <div className="research-candidate-cell">
          <strong title={row.name}>{row.name}</strong>
          <span>{candidateTypeLabel(row)}</span>
        </div>
      </td>
      <td><StatusPill status={row.status} /></td>
      <td>
        <LiveToggleButton liveToggleKey={liveToggleKey} onLiveToggle={onLiveToggle} row={row} />
      </td>
      <td>{sampleCountText(row)}</td>
      <td className={metricClass(row.winRate, EVIDENCE_TARGETS.winRateMin)}>{formatPct(row.winRate)}</td>
      <td>{formatWindow(row.windows?.recent30)}</td>
      <td>{formatWindow(row.windows?.recent60)}</td>
      <td>{formatWindow(row.windows?.recent100)}</td>
      <td><RollingWindows row={row} /></td>
      <td className={metricClass(row.profitFactor, EVIDENCE_TARGETS.profitFactorMin)}>{formatNum(row.profitFactor, 2)}</td>
      <td className={metricClass(row.avgReturn, 0)}>{formatPct(row.avgReturn)}</td>
      <td className={gapRisk ? "is-warn" : ""}>{formatGap(row)}</td>
      <td className={row.maxConsecutiveLosses >= EVIDENCE_TARGETS.lossStreakLimit ? "is-bad" : ""}>{row.maxConsecutiveLosses}</td>
      <td>
        <span className="research-state-stack">
          <small>{dataFreshnessLabel(row.dataFreshnessStatus)}</small>
          <small>{missingFeatureLabel(row.missingFeatureStatus)}</small>
        </span>
      </td>
      <td><span className="research-reason">{reasonLabel(row.reason)}</span></td>
    </tr>
  );
}

function RollingWindows({ row }) {
  const items = rollingWindowItems(row);
  if (!items.length) return <span className="research-live-empty">{EMPTY}</span>;
  return (
    <span className="research-rolling-windows">
      {items.map((item) => (
        <small className={item.passed ? "is-good" : "is-bad"} key={item.key}>{item.text}</small>
      ))}
    </span>
  );
}

function LiveToggleButton({ liveToggleKey, onLiveToggle, row }) {
  const liveEnabled = Boolean(row.liveTradingEnabled);
  if (row.status !== STABLE_STATUS && !liveEnabled) return <span className="research-live-empty">{EMPTY}</span>;
  const pending = liveToggleKey === row.candidateKey;
  return (
    <button
      className={`research-live-toggle ${liveEnabled ? "is-on" : "is-off"}`}
      disabled={pending || !onLiveToggle}
      onClick={() => onLiveToggle(row, !liveEnabled)}
      type="button"
    >
      {pending ? "切换中" : liveEnabled ? "实盘开" : "实盘关"}
    </button>
  );
}

function StatusPill({ status }) {
  return <span className={`research-status-pill ${statusClass(status)}`}>{statusLabel(status)}</span>;
}

function sampleCountText(row) {
  if (row.type === "model" && row.sampleCount > 0 && row.validationSampleCount > 0) return `${row.sampleCount} / 验证 ${row.validationSampleCount}`;
  if (row.sampleCount > 0) return row.sampleCount;
  if (row.type === "model" && row.validationSampleCount > 0) return `验证 ${row.validationSampleCount}`;
  return row.sampleCount;
}

function metricClass(value, target) {
  if (value == null) return "";
  return Number(value) >= target ? "is-good" : "is-bad";
}

function matrixStatusText({ loadError, loading, pagination, reportLoaded, visibleRowCount }) {
  if (loadError) return "候选报告读取失败";
  if (loading) return "正在读取候选报告";
  if (reportLoaded && pagination) {
    const returnedRows = Number(pagination.returnedRows || 0);
    const totalCandidates = Number(pagination.allCandidateCount || returnedRows);
    return `本页可展示 ${visibleRowCount} · 本页候选 ${returnedRows} · 全量候选 ${totalCandidates}`;
  }
  if (reportLoaded) return `可展示 ${visibleRowCount} 个已结算候选`;
  return "等待报告返回";
}

function matrixEmptyText(loadError, loading, reportLoaded) {
  if (loadError) return `候选报告读取失败：${loadError}`;
  if (loading) return "正在读取候选报告…";
  if (reportLoaded) return "暂无候选或模型族状态";
  return "尚未返回候选报告";
}
