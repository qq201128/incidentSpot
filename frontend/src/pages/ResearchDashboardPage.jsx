import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { TOP_ROW_LIMIT, researchSummary, settledRows } from "./researchDashboardData";
import { LiveTradingOverview, ResearchHeader, SummaryStrip } from "./ResearchDashboardSummary";
import { ResearchSidePanel } from "./ResearchDashboardEvidence";
import { SettledSampleMatrix } from "./ResearchDashboardMatrix";
import { useResearchDashboard } from "./useResearchDashboard";
import "./ResearchDashboardPage.css";
import "./ResearchDashboardMatrix.css";
import "./ResearchDashboardSidePanel.css";
import "./ResearchDashboardPage.responsive.css";

export default function ResearchDashboardPage() {
  const [searchParams] = useSearchParams();
  const [symbol, setSymbol] = useState(searchParams.get("symbol") || "BTCUSDT");
  const [duration, setDuration] = useState(searchParams.get("duration") || "10m");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(TOP_ROW_LIMIT);
  const {
    liveToggleKey,
    loadError,
    loading,
    mergingModels,
    report,
    liveOverview,
    liveOverviewError,
    runDailyLoop,
    status,
    toggleCandidateLiveTrading,
  } = useResearchDashboard(symbol, duration, { page, pageSize });
  const rows = useMemo(() => settledRows(report), [report]);
  const summary = useMemo(() => researchSummary(report, rows), [report, rows]);
  const pagination = report?.pagination || {
    page,
    pageSize,
    totalRows: rows.length,
    totalPages: 1,
    returnedRows: rows.length,
  };

  return (
    <main className="research-page layout">
      <ResearchHeader
        duration={duration}
        loading={loading || mergingModels}
        onDurationChange={(value) => {
          setDuration(value);
          setPage(1);
        }}
        onRunDailyLoop={runDailyLoop}
        onSymbolChange={(value) => {
          setSymbol(value);
          setPage(1);
        }}
        status={status}
        symbol={symbol}
      />
      <SummaryStrip summary={summary} />
      <LiveTradingOverview error={liveOverviewError} overview={liveOverview} />
      <section className="research-main-grid">
        <SettledSampleMatrix
          liveToggleKey={liveToggleKey}
          loadError={loadError}
          loading={loading && !report}
          onLiveToggle={toggleCandidateLiveTrading}
          onPageChange={setPage}
          onPageSizeChange={(value) => {
            setPageSize(value);
            setPage(1);
          }}
          pagination={pagination}
          reportLoaded={summary.reportLoaded}
          rows={rows}
        />
        <ResearchSidePanel
          report={report}
          rows={rows}
          summary={summary}
          modelStatuses={report?.modelFamilyStatuses}
        />
      </section>
    </main>
  );
}
