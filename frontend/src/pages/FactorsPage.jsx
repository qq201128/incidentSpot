import { useRef, useState } from "react";
import FactorCombinationPanel from "../components/FactorCombinationPanel";
import FactorDetailPanel from "../components/FactorDetailPanel";
import FactorHighWinrateCard from "../components/FactorHighWinrateCard";
import FactorLibraryAlerts from "../components/FactorLibraryAlerts";
import FactorListPanel from "../components/FactorListPanel";
import FactorRankingTable from "../components/FactorRankingTable";
import { durationLabel, formatUpdatedTime, sourcePillClass } from "../components/factorDisplayUtils";
import { toolbarCategoryOptions } from "../utils/factorCatalogLabels";
import "./FactorsPage.css";
import "./FactorsPage.responsive.css";
import "../components/FactorDetailPanel.css";
import { useFactorPageAnimations } from "./useFactorPageAnimations";
import { useFactorsPageData } from "./useFactorsPageData";

const DURATIONS = [
  { value: "10m", label: "10 分钟" },
  { value: "30m", label: "30 分钟" },
  { value: "60m", label: "60 分钟" },
  { value: "1d", label: "1 天" },
];

const SOURCE_LABELS = [
  { key: "local_definition", label: "本地定义" },
  { key: "agent_candidate", label: "Agent候选" },
  { key: "lstm_shadow", label: "LSTM影子" },
  { key: "composite_cache", label: "组合缓存" },
];

const WORKSPACE_TABS = [
  { key: "detail", label: "因子详情" },
  { key: "combination", label: "多因子组合" },
];

export default function FactorsPage() {
  const pageRef = useRef(null);
  const [workspaceTab, setWorkspaceTab] = useState("detail");
  const { actions, animationKeys, state } = useFactorsPageData();

  useFactorPageAnimations({ pageRef, ...animationKeys });

  const selectFactor = (factorName) => {
    actions.setSelectedName(factorName);
    setWorkspaceTab("detail");
  };

  return (
    <main ref={pageRef} className="factors-page layout">
      <FactorsTopbar state={state} onRefreshRanking={actions.requestRankingRefresh} />
      <FactorsToolbar actions={actions} state={state} />
      <div className="factors-main-grid" data-factor-motion="primary-grid">
        <FactorListPanel
          category={state.category}
          factors={state.filteredFactors}
          listPage={state.listPage}
          listPageCount={state.listPageCount}
          listPageSize={state.listPageSize}
          listTab={state.listTab}
          listTotal={state.listTotal}
          onCategoryChange={actions.setCategory}
          onListPageChange={actions.setListPage}
          onListPageSizeChange={actions.setListPageSize}
          onListQueryChange={actions.setQuery}
          onListTabChange={actions.setListTab}
          onRefreshList={actions.reloadList}
          onSelectFactor={selectFactor}
          query={state.query}
          selectedName={state.selectedName}
          total={state.total}
          comboTotal={state.comboTotal}
        />
        <FactorsWorkspacePanel
          actions={actions}
          state={state}
          tab={workspaceTab}
          onSelectFactor={selectFactor}
          onTabChange={setWorkspaceTab}
        />
      </div>
    </main>
  );
}

function FactorsTopbar({ state, onRefreshRanking }) {
  const summary = state.sourceSummary || {};
  const summaryGlobal = state.sourceSummaryGlobal || {};
  const sym = state.symbol;
  const dur = durationLabel(state.duration);
  const rankingReady = state.ranking.items.length > 0 || state.overview?.rankingSource === "cache";
  const updatedAt = state.overview?.rankingUpdatedAt ?? null;

  return (
    <header className="factors-topbar topbar" data-factor-motion="hero">
      <div className="factors-topbar-title">
        <span className="eyebrow">因子库 /</span>
        <h1>量化因子目录与回测</h1>
        <p>统一管理单因子、组合因子、回测评分与排名缓存</p>
      </div>
      <div className="factors-topbar-meta">
        <div className="factors-topbar-metrics" aria-label="因子库概览">
          <TopbarMetric label="单因子" value={state.total} />
          <TopbarMetric label="组合因子" value={state.comboTotal} />
          <TopbarMetric label="排名" value={state.rankingTotal} />
        </div>
        <div className="factors-topbar-sources-wrap">
          <span className="factors-sources-label">来源状态</span>
          <div className="factors-topbar-sources" aria-label="因子来源">
            {SOURCE_LABELS.map((item) => (
              <SourcePill
                key={item.key}
                kind={item.key}
                label={item.key === "agent_candidate" ? "Agent入库" : item.label}
                value={summary[item.key] ?? 0}
                hint={
                  item.key === "agent_candidate" && summaryGlobal.agent_candidate != null
                    ? `全库 ${summaryGlobal.agent_candidate}`
                    : ""
                }
              />
            ))}
          </div>
        </div>
      </div>
      <div className="factors-topbar-status">
        <span className={`status-dot${rankingReady ? "" : " is-warn"}`} />
        <p>
          {rankingReady
            ? `排名缓存已刷新 · ${sym} · ${dur}`
            : `暂无排名缓存 · ${sym} · ${dur}`}
        </p>
        <div className="factors-topbar-status-row">
          <small>更新时间: {formatUpdatedTime(updatedAt)}</small>
          <button type="button" className="factors-icon-btn" title="刷新排名" onClick={onRefreshRanking}>
            ↻
          </button>
        </div>
      </div>
    </header>
  );
}

function TopbarMetric({ label, value }) {
  return (
    <span className="factors-topbar-metric">
      <small>{label}</small>
      <b>{value ?? 0}</b>
    </span>
  );
}

function SourcePill({ kind, label, value, hint = "" }) {
  return (
    <span className={`factors-source-pill ${sourcePillClass(kind)}`} title={hint || undefined}>
      {label} <b>{value ?? 0}</b>
      {hint ? <small className="factors-source-pill-hint">{hint}</small> : null}
    </span>
  );
}

function FactorsToolbar({ actions, state }) {
  return (
    <section className="factors-toolbar card-surface" data-factor-motion="toolbar">
      <div className="factors-toolbar-filters">
        <label>
          <span>交易对</span>
          <select value={state.symbol} onChange={(event) => actions.setSymbol(event.target.value)}>
            <option value="BTCUSDT">BTCUSDT</option>
            <option value="ETHUSDT">ETHUSDT</option>
          </select>
        </label>
        <label>
          <span>周期</span>
          <select value={state.duration} onChange={(event) => actions.setDuration(event.target.value)}>
            {DURATIONS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>分类</span>
          <select value={state.category} onChange={(event) => actions.setCategory(event.target.value)}>
            {toolbarCategoryOptions(state.categories).map((item) => (
              <option key={item.key || "all"} value={item.key}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="factors-toolbar-search">
          <span className="sr-only">搜索</span>
          <input
            value={state.query}
            onChange={(event) => actions.setQuery(event.target.value)}
            placeholder="搜索因子名/公式/来源"
          />
        </label>
      </div>
      <div className="factors-toolbar-actions">
        <button type="button" className="factors-btn-outline" onClick={actions.requestRankingRefresh}>
          刷新排名
        </button>
        <button
          type="button"
          className="factors-btn-primary"
          disabled={!state.selectedName || state.backtest.loading}
          onClick={actions.runBacktest}
        >
          {state.backtest.loading ? "计算中…" : "运行回测"}
        </button>
      </div>
      <p className="factors-toolbar-hint" role="status">
        数据缺失或回测失败将直接显示错误原因
      </p>
    </section>
  );
}

function FactorsWorkspacePanel({ actions, state, tab, onSelectFactor, onTabChange }) {
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
          <div className="factors-workspace-detail-layout">
            <div className="factors-workspace-detail-top">
              <FactorDetailPanel
                backtestError={state.backtest.error}
                detail={state.detail.data}
                detailError={state.detail.error}
                detailLoading={state.detail.loading}
                displayMetrics={state.displayMetrics}
                duration={state.previewDuration}
                onDurationChange={actions.setPreviewDuration}
                periodScores={state.periodScores}
                periodScoresPending={state.periodScoresPending}
                selectedFactor={state.selectedFactor}
                selectedName={state.selectedName}
              />
            </div>
            <div className="factors-workspace-bottom">
              <FactorRankingTable
                onPageChange={actions.setRankingPage}
                onQueryChange={actions.setRankingQuery}
                page={state.ranking.page}
                pageCount={state.ranking.pageCount}
                query={state.ranking.query}
                ranking={state.ranking.items}
                selectedName={state.selectedName}
                total={state.ranking.total}
                unfilteredTotal={state.ranking.unfilteredTotal}
                onSelectFactor={onSelectFactor}
              />
              <FactorHighWinrateCard combo={state.highWinrateCombo} />
            </div>
            <FactorLibraryAlerts alerts={state.alerts} />
          </div>
        ) : null}
        {tab === "combination" ? (
          <FactorCombinationPanel symbol={state.symbol} duration={state.duration} />
        ) : null}
      </div>
    </section>
  );
}
