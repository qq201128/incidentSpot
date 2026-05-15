import { useRef } from "react";
import FactorCombinationPanel from "../components/FactorCombinationPanel";
import FactorDetailPanel from "../components/FactorDetailPanel";
import FactorLearningPanel from "../components/FactorLearningPanel";
import FactorListPanel from "../components/FactorListPanel";
import "./FactorsPage.css";
import { useFactorPageAnimations } from "./useFactorPageAnimations";
import { useFactorsPageData } from "./useFactorsPageData";

const DURATIONS = [
  { value: "10m", label: "10 分钟" },
  { value: "30m", label: "30 分钟" },
  { value: "60m", label: "60 分钟" },
  { value: "1d", label: "1 天" },
];

export default function FactorsPage() {
  const pageRef = useRef(null);
  const { actions, animationKeys, state } = useFactorsPageData();

  useFactorPageAnimations({ pageRef, ...animationKeys });

  return (
    <main ref={pageRef} className="factors-page layout">
      <FactorsTopbar listStatus={state.listStatus} />
      <FactorsToolbar actions={actions} state={state} />
      <FactorsPrimaryGrid actions={actions} state={state} />
      <FactorsSecondaryGrid duration={state.duration} symbol={state.symbol} />
    </main>
  );
}

function FactorsTopbar({ listStatus }) {
  return (
    <header className="factors-topbar topbar" data-factor-motion="hero">
      <div>
        <span className="eyebrow">因子库</span>
        <h1>量化因子目录与回测</h1>
      </div>
      <p className="status-pill factors-status-pill">{listStatus}</p>
    </header>
  );
}

function FactorsToolbar({ actions, state }) {
  return (
    <section className="factors-toolbar card-surface" data-factor-motion="toolbar">
      <div className="factors-toolbar-row">
        <label>
          交易对
          <input
            value={state.symbol}
            onChange={(event) => actions.setSymbol(event.target.value.toUpperCase())}
            placeholder="BTCUSDT"
          />
        </label>
        <label>
          规则周期
          <select value={state.duration} onChange={(event) => actions.setDuration(event.target.value)}>
            {DURATIONS.map((duration) => (
              <option key={duration.value} value={duration.value}>
                {duration.label}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="factors-btn-secondary" onClick={actions.requestRankingRefresh}>
          请求后台刷新排名
        </button>
      </div>
      <p className="factors-rank-hint">{state.ranking.status}</p>
    </section>
  );
}

function FactorsPrimaryGrid({ actions, state }) {
  return (
    <div className="factors-grid" data-factor-motion="primary-grid">
      <FactorListPanel
        categories={state.categories}
        category={state.category}
        factors={state.filteredFactors}
        onCategoryChange={actions.setCategory}
        onQueryChange={actions.setQuery}
        onSelectFactor={actions.setSelectedName}
        query={state.query}
        selectedName={state.selectedName}
        total={state.total}
      />
      <FactorDetailPanel
        backtest={state.backtest.data}
        backtestError={state.backtest.error}
        backtestLoading={state.backtest.loading}
        detail={state.detail.data}
        detailError={state.detail.error}
        onRunBacktest={actions.runBacktest}
        onSelectFactor={actions.setSelectedName}
        ranking={state.ranking.items}
        selectedName={state.selectedName}
      />
    </div>
  );
}

function FactorsSecondaryGrid({ duration, symbol }) {
  return (
    <div className="factors-secondary-grid" data-factor-motion="secondary-grid">
      <FactorCombinationPanel symbol={symbol} duration={duration} />
      <FactorLearningPanel symbol={symbol} duration={duration} />
    </div>
  );
}
