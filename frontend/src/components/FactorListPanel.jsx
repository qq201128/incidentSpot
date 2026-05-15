import { useEffect, useMemo, useState } from "react";
import { directionLabel, factorTitle } from "./factorDisplayUtils";
import "./FactorListPanel.css";

const PAGE_SIZE_OPTIONS = [12, 24, 48];
const LIST_TABS = [
  { key: "single", label: "单因子", title: "因子列表" },
  { key: "combo", label: "组合因子", title: "组合因子列表" },
];

export default function FactorListPanel({
  categories,
  category,
  comboFactors,
  comboTotal,
  factors,
  onCategoryChange,
  onQueryChange,
  onSelectFactor,
  query,
  selectedName,
  total,
}) {
  const [activeTab, setActiveTab] = useState(LIST_TABS[0].key);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0]);
  const activeFactors = activeTab === "combo" ? comboFactors : factors;
  const activeTotal = activeTab === "combo" ? comboTotal : total;
  const totalItems = activeFactors.length;
  const pageCount = Math.max(1, Math.ceil(totalItems / pageSize));
  const visibleFactors = useMemo(
    () => activeFactors.slice((page - 1) * pageSize, page * pageSize),
    [activeFactors, page, pageSize],
  );
  const activeTitle = listTitle(activeTab);

  useEffect(() => {
    setPage(1);
  }, [activeTab, category, query, pageSize]);

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  return (
    <section className="factors-list-panel card-surface">
      <div className="section-head factors-section-head">
        <div>
          <span className="section-kicker">筛选</span>
          <h2>{activeTitle}</h2>
        </div>
        <span className="factors-count">{totalItems} / {activeTotal} 项</span>
      </div>
      <ListTabs activeTab={activeTab} onChange={setActiveTab} />
      {activeTab === "single" ? (
        <CategoryChips categories={categories} category={category} onChange={onCategoryChange} />
      ) : null}
      <div className="factors-list-controls">
        <SearchBox query={query} onChange={onQueryChange} />
        <PageSizeSelect pageSize={pageSize} onChange={setPageSize} />
      </div>
      <FactorTable factors={visibleFactors} selectedName={selectedName} onSelect={onSelectFactor} />
      <Pagination
        page={page}
        pageCount={pageCount}
        pageSize={pageSize}
        totalItems={totalItems}
        onPageChange={setPage}
      />
    </section>
  );
}

function listTitle(activeTab) {
  return LIST_TABS.find((item) => item.key === activeTab)?.title || LIST_TABS[0].title;
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

function CategoryChips({ categories, category, onChange }) {
  return (
    <div className="factors-chips" role="tablist" aria-label="因子分类">
      <button
        type="button"
        className={`factors-chip${category === "" ? " factors-chip-active" : ""}`}
        onClick={() => onChange("")}
      >
        全部
      </button>
      {categories.map((item) => (
        <button
          key={item.key}
          type="button"
          className={`factors-chip${category === item.key ? " factors-chip-active" : ""}`}
          onClick={() => onChange(item.key)}
          title={`${item.count ?? 0} 个`}
        >
          {item.name}
          <span className="factors-chip-meta">{item.count ?? 0}</span>
        </button>
      ))}
    </div>
  );
}

function SearchBox({ query, onChange }) {
  return (
    <label className="factors-search">
      搜索
      <input value={query} onChange={(event) => onChange(event.target.value)} placeholder="名称或描述…" />
    </label>
  );
}

function PageSizeSelect({ pageSize, onChange }) {
  return (
    <label className="factors-page-size">
      每页
      <select value={pageSize} onChange={(event) => onChange(Number(event.target.value))}>
        {PAGE_SIZE_OPTIONS.map((size) => (
          <option key={size} value={size}>
            {size} 条
          </option>
        ))}
      </select>
    </label>
  );
}

function FactorTable({ factors, selectedName, onSelect }) {
  return (
    <div className="factors-table-wrap">
      <table className="factors-table">
        <thead>
          <tr>
            <th>中文因子</th>
            <th>分类</th>
            <th>方向</th>
          </tr>
        </thead>
        <tbody>{factors.map((factor) => renderFactorRow(factor, selectedName, onSelect))}</tbody>
      </table>
      {!factors.length ? <p className="factors-empty">无匹配因子</p> : null}
    </div>
  );
}

function renderFactorRow(factor, selectedName, onSelect) {
  return (
    <tr
      key={factor.name}
      className={selectedName === factor.name ? "factors-row-selected" : ""}
      onClick={() => onSelect(factor.name)}
    >
      <td>
        <strong className="factors-name-cn">{factorTitle(factor)}</strong>
        <code className="factors-code">{factor.name}</code>
      </td>
      <td>{factor.categoryName || factor.category}</td>
      <td>{directionLabel(factor.direction)}</td>
    </tr>
  );
}

function Pagination({ page, pageCount, pageSize, totalItems, onPageChange }) {
  const start = totalItems ? (page - 1) * pageSize + 1 : 0;
  const end = Math.min(page * pageSize, totalItems);
  return (
    <nav className="factors-pagination" aria-label="因子列表分页">
      <span className="factors-page-range">
        {start}-{end} / {totalItems}
      </span>
      <div className="factors-page-actions">
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(1)}>
          首页
        </button>
        <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
          上一页
        </button>
        <strong>
          {page} / {pageCount}
        </strong>
        <button type="button" disabled={page >= pageCount} onClick={() => onPageChange(page + 1)}>
          下一页
        </button>
        <button type="button" disabled={page >= pageCount} onClick={() => onPageChange(pageCount)}>
          末页
        </button>
      </div>
    </nav>
  );
}
