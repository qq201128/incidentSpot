import { directionLabel, factorTitle } from "./factorDisplayUtils";

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
  return (
    <section className="factors-list-panel card-surface">
      <div className="section-head factors-section-head">
        <div>
          <span className="section-kicker">筛选</span>
          <h2>因子列表</h2>
        </div>
        <span className="factors-count">{total} 项</span>
      </div>
      <CategoryChips categories={categories} category={category} onChange={onCategoryChange} />
      <SearchBox query={query} onChange={onQueryChange} />
      <FactorTable factors={factors} selectedName={selectedName} onSelect={onSelectFactor} />
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
