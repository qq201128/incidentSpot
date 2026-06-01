import { useState } from "react";
import MiningAgentTable from "../components/mining/MiningAgentTable";
import MiningKpiCards from "../components/mining/MiningKpiCards";
import MiningModelGrid from "../components/mining/MiningModelGrid";
import MiningPageHeader from "../components/mining/MiningPageHeader";
import MiningSidebar from "../components/mining/MiningSidebar";
import { useMiningPageData } from "./useMiningPageData";
import "./MiningPage.css";

const DURATIONS = [
  { value: "10m", label: "10分钟" },
  { value: "30m", label: "30分钟" },
  { value: "60m", label: "60分钟" },
  { value: "1d", label: "1天" },
];

export default function FactorLearningPage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [duration, setDuration] = useState("10m");
  const mining = useMiningPageData(symbol, duration);

  const overview = mining.overview;
  const initialLoading = mining.loading && !overview;

  return (
    <main className="mining-page layout">
      <MiningPageHeader
        symbol={symbol}
        duration={duration}
        durationOptions={DURATIONS}
        onSymbolChange={setSymbol}
        onDurationChange={setDuration}
        header={overview?.header}
        updatedAt={overview?.updatedAt}
        onReload={() => void mining.reload()}
        reloading={mining.busy !== "" || mining.loading}
      />

      {initialLoading ? (
        <div className="mining-loading" role="status">
          正在加载自动挖掘数据…（汇总 10 个模型族状态，首次可能需数秒）
        </div>
      ) : null}

      {mining.status ? <div className="mining-banner">{mining.status}</div> : null}

      {overview ? (
        <div className="mining-page-body">
          <MiningKpiCards
            summary={overview.summary}
            trainingRules={overview.trainingRules}
            busy={mining.busy}
            onRefreshLocal={() => void mining.refreshLocal()}
            onRefreshAgent={() => void mining.refreshAgent()}
            onSearchAll={() => void mining.searchAllModels()}
          />
          <section className="mining-workspace">
            <MiningModelGrid
              models={overview.models}
              runStatus={overview.runStatus}
              summary={overview.summary}
              busy={mining.busy}
              onSearchModel={(family) => void mining.searchModel(family)}
            />
            <MiningSidebar
              sidebar={overview.sidebar}
              operators={overview.operators}
              ingestionPath={overview.ingestionPath}
            />
            <MiningAgentTable rows={overview.agentCandidates} />
          </section>
        </div>
      ) : !initialLoading && !mining.status ? (
        <div className="mining-loading">暂无数据</div>
      ) : null}
    </main>
  );
}
