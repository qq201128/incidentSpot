import { directionLabel, factorTitle, sourceLabel, sourceTagClass } from "./factorDisplayUtils";
import { factorTableCategoryLabel, SIDEBAR_CATEGORY_CHIPS } from "../utils/factorCatalogLabels";
import "./FactorListPanel.css";
import "./FactorListPanel.observe.css";

const PAGE_SIZE_OPTIONS = [20, 48, 96];
const LIST_TABS = [
  { key: "single", label: "单因子" },
  { key: "combo", label: "组合因子" },
];

export default function FactorListPanel({
  category,
  comboTotal,
  factors,
  listPage,
  listPageCount,
  listPageSize,
  listTab,
  listTotal,
  onCategoryChange,
  onListPageChange,
  onListPageSizeChange,
  onListQueryChange,
  onListTabChange,
  onRefreshList,
  onSelectFactor,
  query,
  selectedName,
  total,
}) {
  const catalogTotal = listTab === "combo" ? comboTotal : total;

  const directoryTitle = listTab === "combo" ? "组合因子目录" : "单因子目录";

  return (
    <section className="factors-list-panel card-surface">
      <header className="factors-directory-head">
        <span className="section-kicker">筛选</span>
        <span className="factors-directory-sep">/</span>
        <h2>{directoryTitle}</h2>
      </header>
      <ListTabs activeTab={listTab} onChange={onListTabChange} />

      {listTab === "single" ? (
        <CategoryChips category={category} onChange={onCategoryChange} />
      ) : null}

      <div className="factors-list-controls">
        <label className="factors-search">
          <span className="sr-only">搜索因子</span>
          <span className="factors-search-icon" aria-hidden>
            ⌕
          </span>
          <input
            value={query}
            onChange={(event) => onListQueryChange(event.target.value)}
            placeholder="搜索因子…"
          />
        </label>
        <label className="factors-page-size">
          <span className="sr-only">每页条数</span>
          <select
            value={listPageSize}
            onChange={(event) => onListPageSizeChange(Number(event.target.value))}
          >
            {PAGE_SIZE_OPTIONS.map((size) => (
              <option key={size} value={size}>
                每页 {size} 条
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="factors-filter-icon" title="筛选" aria-label="筛选">
          ☰
        </button>
      </div>

      <FactorTable
        factors={factors}
        listTab={listTab}
        page={listPage}
        pageSize={listPageSize}
        selectedName={selectedName}
        onSelect={onSelectFactor}
      />

      <Pagination
        catalogTotal={catalogTotal}
        listTotal={listTotal}
        page={listPage}
        pageCount={listPageCount}
        onPageChange={onListPageChange}
        onRefresh={onRefreshList}
      />
    </section>
  );
}

function ListTabs({ activeTab, onChange }) {
  return (
    <div className="factors-list-tabs" role="tablist" aria-label="因子列表类型">
      {LIST_TABS.map((item) => (
        <button
          key={item.key}
          type="button"
          role="tab"
          aria-selected={activeTab === item.key}
          className={`factors-list-tab${activeTab === item.key ? " factors-list-tab-active" : ""}`}
          onClick={() => onChange(item.key)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function CategoryChips({ category, onChange }) {
  return (
    <div className="factors-chips" role="tablist" aria-label="因子分类">
      {SIDEBAR_CATEGORY_CHIPS.map((chip) => (
        <button
          key={chip.key || "all"}
          type="button"
          className={`factors-chip${category === chip.key ? " factors-chip-active" : ""}`}
          onClick={() => onChange(chip.key)}
        >
          {chip.label}
        </button>
      ))}
    </div>
  );
}

function FactorTable({ factors, listTab, page, pageSize, selectedName, onSelect }) {
  const indexOffset = (page - 1) * pageSize;
  const showScore = listTab === "combo";
  return (
    <div className="factors-table-wrap">
      <table className={`factors-table factors-catalog-table${showScore ? " factors-catalog-table-scored" : ""}`}>
        <thead>
          <tr>
            <th>#</th>
            <th>中文因子</th>
            <th>分类</th>
            <th>方向</th>
            {showScore ? <th>评分</th> : null}
            {showScore ? <th>观察状态</th> : null}
            <th>来源</th>
          </tr>
        </thead>
        <tbody>
          {factors.map((factor, index) =>
            renderFactorRow(factor, indexOffset + index + 1, selectedName, onSelect, showScore),
          )}
        </tbody>
      </table>
      {!factors.length ? <p className="factors-empty">无匹配因子</p> : null}
    </div>
  );
}

function renderFactorRow(factor, rankIndex, selectedName, onSelect, showScore) {
  const direction = directionLabel(factor.direction);
  return (
    <tr
      key={factor.name}
      className={selectedName === factor.name ? "factors-row-selected" : ""}
      onClick={() => onSelect(factor.name)}
    >
      <td className="factors-index-cell">{rankIndex}</td>
      <td className="factors-factor-cell" title={`${factorTitle(factor)} (${factor.name})`}>
        <strong className="factors-name-cn">{factorTitle(factor)}</strong>
        <code className="factors-code">{factor.name}</code>
      </td>
      <td className="factors-category-cell">{factorTableCategoryLabel(factor)}</td>
      <td className={`factors-direction-cell${direction === "正向" ? " is-positive" : ""}`}>{direction}</td>
      {showScore ? <td className="factors-score-cell">{formatScore(factor.factorScore)}</td> : null}
      {showScore ? <td className="factors-observe-cell">{observeStatus(factor)}</td> : null}
      <td className="factors-source-cell">
        <span className={`factors-source-tag ${sourceTagClass(factor)}`} title={sourceLabel(factor)}>
          {sourceLabel(factor)}
        </span>
      </td>
    </tr>
  );
}

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(1) : "-";
}

function observeStatus(factor) {
  if (factor.walkForwardPassed === true || factor.paperLiveStatus === "backtest_passed") {
    return <span className="factors-observe-tag is-pass">通过</span>;
  }
  const reason = factor.walkForwardFailureReason || "回测未通过";
  return (
    <span className="factors-observe-tag is-observe" title={reason}>
      观察
    </span>
  );
}

function Pagination({ catalogTotal, listTotal, page, pageCount, onPageChange, onRefresh }) {
  const totalLabel = listTotal !== catalogTotal ? `${listTotal} / ${catalogTotal}` : catalogTotal;
  return (
    <nav className="factors-pagination" aria-label="因子列表分页">
      <span className="factors-page-total">共 {totalLabel} 条</span>
      <div className="factors-page-actions">
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(1)} aria-label="首页">
          «
        </button>
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="上一页">
          ‹
        </button>
        <strong className="factors-page-indicator">
          {page} / {pageCount}
        </strong>
        <button
          type="button"
          disabled={page >= pageCount}
          onClick={() => onPageChange(page + 1)}
          aria-label="下一页"
        >
          ›
        </button>
        <button
          type="button"
          disabled={page >= pageCount}
          onClick={() => onPageChange(pageCount)}
          aria-label="末页"
        >
          »
        </button>
      </div>
      <button type="button" className="factors-icon-btn factors-list-refresh" title="刷新列表" onClick={onRefresh}>
        ↻
      </button>
    </nav>
  );
}
