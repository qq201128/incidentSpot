/**
 * 因子过滤面板组件
 */
import './FactorFilterPanel.css';

export default function FactorFilterPanel({ filters, onFilterChange, totalCount, filteredCount }) {
  const updateFilter = (key, value) => {
    onFilterChange?.(key, value);
  };

  return (
    <div className="factor-filter-panel">
      <div className="filter-search">
        <input
          type="text"
          placeholder="搜索因子名称、成员..."
          value={filters.search}
          onChange={(e) => updateFilter('search', e.target.value)}
          className="filter-search-input"
        />
        {filters.search && (
          <button
            type="button"
            className="filter-search-clear"
            onClick={() => updateFilter('search', '')}
            aria-label="清除搜索"
          >
            ×
          </button>
        )}
      </div>

      <div className="filter-controls">
        <div className="filter-group">
          <label className="filter-label">
            <span>最低胜率</span>
            <input
              type="range"
              min="45"
              max="70"
              step="1"
              value={filters.minWinRate}
              onChange={(e) => updateFilter('minWinRate', Number(e.target.value))}
              className="filter-range"
            />
            <strong className="filter-value">{filters.minWinRate}%</strong>
          </label>
        </div>

        <div className="filter-group">
          <label className="filter-label">
            <span>最低 IR</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={filters.minIR}
              onChange={(e) => updateFilter('minIR', Number(e.target.value))}
              className="filter-range"
            />
            <strong className="filter-value">{filters.minIR.toFixed(2)}</strong>
          </label>
        </div>

        <div className="filter-group">
          <label className="filter-label">
            <span>排序方式</span>
            <select
              value={filters.sortBy}
              onChange={(e) => updateFilter('sortBy', e.target.value)}
              className="filter-select"
            >
              <option value="ir">按 IR 排序</option>
              <option value="winRate">按胜率排序</option>
              <option value="sharpe">按夏普排序</option>
              <option value="trades">按交易次数排序</option>
            </select>
          </label>
        </div>
      </div>

      <div className="filter-stats">
        <span className="filter-stat">
          显示 <strong>{filteredCount}</strong> / {totalCount} 个因子
        </span>
        {filteredCount < totalCount && (
          <button
            type="button"
            className="filter-reset"
            onClick={() => {
              updateFilter('search', '');
              updateFilter('minWinRate', 50);
              updateFilter('minIR', 0.3);
            }}
          >
            重置过滤
          </button>
        )}
      </div>
    </div>
  );
}
