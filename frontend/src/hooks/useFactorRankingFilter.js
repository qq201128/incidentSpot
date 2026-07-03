/**
 * 因子排名过滤和排序 Hook
 */
import { useMemo, useState } from 'react';

export function useFactorRankingFilter(ranking) {
  const [filters, setFilters] = useState({
    search: '',
    minWinRate: 50,
    minIR: 0.3,
    sortBy: 'ir',
  });

  const filteredRanking = useMemo(() => {
    if (!ranking || !Array.isArray(ranking)) return [];

    let filtered = ranking;

    // 搜索过滤
    if (filters.search.trim()) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(
        (row) =>
          matchesSearch(row, searchLower)
      );
    }

    // 胜率过滤
    if (filters.minWinRate > 0) {
      filtered = filtered.filter((row) => getWinRate(row) >= filters.minWinRate);
    }

    // IR过滤
    if (filters.minIR > 0) {
      filtered = filtered.filter((row) => getIR(row) >= filters.minIR);
    }

    // 排序
    filtered = sortRanking(filtered, filters.sortBy);

    return filtered;
  }, [ranking, filters]);

  const updateFilter = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setFilters({
      search: '',
      minWinRate: 50,
      minIR: 0.3,
      sortBy: 'ir',
    });
  };

  return {
    filters,
    filteredRanking,
    updateFilter,
    resetFilters,
    totalCount: ranking?.length || 0,
    filteredCount: filteredRanking.length,
  };
}

function matchesSearch(row, search) {
  const searchable = [
    row.factorName || '',
    row.factorDisplayName || '',
    row.description || '',
    getMembersText(row.members),
  ]
    .join(' ')
    .toLowerCase();

  return searchable.includes(search);
}

function getMembersText(members) {
  if (!Array.isArray(members)) return '';
  return members.map((m) => `${m.name || ''} ${m.displayName || ''}`).join(' ');
}

function getWinRate(row) {
  return Number(row.winRate) || 0;
}

function getIR(row) {
  return Number(row.ir) || 0;
}

function getSharpe(row) {
  return Number(row.sharpe) || 0;
}

function getTrades(row) {
  return Number(row.trades) || 0;
}

function sortRanking(ranking, sortBy) {
  const sorters = {
    ir: (a, b) => getIR(b) - getIR(a),
    winRate: (a, b) => getWinRate(b) - getWinRate(a),
    sharpe: (a, b) => getSharpe(b) - getSharpe(a),
    trades: (a, b) => getTrades(b) - getTrades(a),
  };

  const sorter = sorters[sortBy] || sorters.ir;
  return [...ranking].sort(sorter);
}
