import { useEffect, useMemo, useState } from "react";
import { directionLabel, factorTitle } from "./factorDisplayUtils";
import "./FactorListPanel.css";

const PAGE_SIZE_OPTIONS = [12, 24, 48];

export default function FactorListPanel({
  categories,
  category,
  factors,
  onCategoryChange,
  onQueryChange,
  onSelectFactor,
  query,
  selectedName,
  total,
}) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0]);
  const totalItems = factors.length;
  const pageCount = Math.max(1, Math.ceil(totalItems / pageSize));
  const visibleFactors = useMemo(
    () => factors.slice((page - 1) * pageSize, page * pageSize),
    [factors, page, pageSize],
  );

  useEffect(() => {
    setPage(1);
  }, [category, query, pageSize]);

  useEffect(() => {
    if (page > pageCount) setPage(pageCount);
  }, [page, pageCount]);

  return (
    <section className="factors-list-panel card-surface">
      <div className="section-head factors-section-head">
        <div>
          <span className="section-kicker">筛选</span>
          <h2>因子列表</h2>
        </div>
        <span className="factors-count">{totalItems} / {total} 项</span>
      </div>
      <CategoryChips categories={categories} category={category} onChange={onCategoryChange} />
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
