import { useState } from "react";
import MiningAgentTable from "../components/mining/MiningAgentTable";
import MiningKpiCards from "../components/mining/MiningKpiCards";
import MiningModelGrid from "../components/mining/MiningModelGrid";
import MiningPageHeader from "../components/mining/MiningPageHeader";
import MiningSidebar from "../components/mining/MiningSidebar";
import { useMiningPageData } from "./useMiningPageData";
import "./MiningPage.css";
import "./MiningPage.responsive.css";

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

      {initialLoading ? <MiningInitialState /> : null}

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
            onRetrainAll={() => void mining.retrainAllModels()}
          />
          <section className="mining-workspace">
            <MiningModelGrid
              models={overview.models}
              runStatus={overview.runStatus}
              summary={overview.summary}
              busy={mining.busy}
              symbol={overview?.symbol || symbol}
              duration={overview?.duration || duration}
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

function MiningInitialState() {
  return (
    <section className="mining-loading-panel" role="status" aria-live="polite">
      <div>
        <span className="mining-loading-kicker">REAL API REQUEST</span>
        <h2>正在读取自动挖掘数据</h2>
        <p>汇总模型族运行态、候选库、Agent 入库和 Worker 日志，首次请求可能需要数秒。</p>
      </div>
      <div className="mining-loading-steps" aria-label="读取中的数据源">
        <span>模型族状态</span>
        <span>候选记录</span>
        <span>Worker 运行态</span>
      </div>
    </section>
  );
}
