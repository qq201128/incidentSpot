import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { researchSummary, settledRows } from "./researchDashboardData";
import { ResearchHeader, SummaryStrip } from "./ResearchDashboardSummary";
import { ResearchSidePanel, SettledSampleMatrix } from "./ResearchDashboardEvidence";
import { useResearchDashboard } from "./useResearchDashboard";
import "./ResearchDashboardPage.css";
import "./ResearchDashboardMatrix.css";
import "./ResearchDashboardSidePanel.css";
import "./ResearchDashboardPage.responsive.css";

export default function ResearchDashboardPage() {
  const [searchParams] = useSearchParams();
  const [symbol, setSymbol] = useState(searchParams.get("symbol") || "BTCUSDT");
  const [duration, setDuration] = useState(searchParams.get("duration") || "10m");
  const { loadError, loading, mergingModels, report, runDailyLoop, status } = useResearchDashboard(
    symbol,
    duration,
  );
  const rows = useMemo(() => settledRows(report), [report]);
  const summary = useMemo(() => researchSummary(report, rows), [report, rows]);

  return (
    <main className="research-page layout">
      <ResearchHeader
        duration={duration}
        loading={loading || mergingModels}
        onDurationChange={setDuration}
        onRunDailyLoop={runDailyLoop}
        onSymbolChange={setSymbol}
        status={status}
        symbol={symbol}
      />
      <SummaryStrip summary={summary} />
      <section className="research-main-grid">
        <SettledSampleMatrix
          duration={duration}
          loadError={loadError}
          loading={loading && !report}
          reportLoaded={summary.reportLoaded}
          rows={rows}
          symbol={symbol}
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
