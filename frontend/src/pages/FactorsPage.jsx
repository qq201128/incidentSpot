import { useRef, useState } from "react";
import FactorCombinationPanel from "../components/FactorCombinationPanel";
import FactorDetailPanel from "../components/FactorDetailPanel";
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

const WORKSPACE_TABS = [
  { key: "detail", label: "因子详情" },
  { key: "combination", label: "多因子组合" },
];

export default function FactorsPage() {
  const pageRef = useRef(null);
  const [workspaceTab, setWorkspaceTab] = useState(WORKSPACE_TABS[0].key);
  const { actions, animationKeys, state } = useFactorsPageData();

  useFactorPageAnimations({ pageRef, ...animationKeys });

  return (
    <main ref={pageRef} className="factors-page layout">
      <FactorsTopbar listStatus={state.listStatus} />
      <FactorsToolbar actions={actions} state={state} />
      <FactorsWorkbench actions={actions} state={state} tab={workspaceTab} onTabChange={setWorkspaceTab} />
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

function FactorsWorkbench({ actions, state, tab, onTabChange }) {
  const selectFactor = (factorName) => {
    actions.setSelectedName(factorName);
    onTabChange("detail");
  };

  return (
    <div className="factors-workbench" data-factor-motion="primary-grid">
      <FactorListPanel
        categories={state.categories}
        category={state.category}
        comboFactors={state.filteredComboFactors}
        comboTotal={state.comboTotal}
        factors={state.filteredFactors}
        onCategoryChange={actions.setCategory}
        onQueryChange={actions.setQuery}
        onSelectFactor={selectFactor}
        query={state.query}
        selectedName={state.selectedName}
        total={state.total}
      />
      <FactorsWorkspacePanel
        actions={actions}
        duration={state.duration}
        state={state}
        symbol={state.symbol}
        tab={tab}
        onTabChange={onTabChange}
      />
    </div>
  );
}

function FactorsWorkspacePanel({ actions, duration, state, symbol, tab, onTabChange }) {
  return (
    <section className="factors-workspace-panel card-surface" data-factor-motion="secondary-grid">
      <div className="factors-workspace-tabs" role="tablist" aria-label="因子工作区">
        {WORKSPACE_TABS.map((item) => (
          <button
            key={item.key}
            type="button"
            role="tab"
            aria-selected={tab === item.key}
            className={tab === item.key ? "factors-workspace-tab-active" : ""}
            onClick={() => onTabChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="factors-workspace-content">
        {tab === "detail" ? (
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
        ) : null}
        {tab === "combination" ? <FactorCombinationPanel symbol={symbol} duration={duration} /> : null}
      </div>
    </section>
  );
}
