import { useEffect, useState } from "react";
import { fetchFactorsList } from "../api/client";

const LIST_DEBOUNCE_MS = 280;

export function useDebouncedValue(value, delayMs = LIST_DEBOUNCE_MS) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [delayMs, value]);
  return debounced;
}

export function useFactorsList({ category, duration, kind, listPage, listPageSize, query, reloadKey, symbol }) {
  const debouncedQuery = useDebouncedValue(query);
  const [state, setState] = useState(initialListState);

  useEffect(() => {
    let cancelled = false;
    setState((prev) => ({ ...prev, status: "加载中…" }));
    fetchFactorsList({
      category: category || undefined,
      duration,
      kind,
      q: debouncedQuery.trim() || undefined,
      page: listPage,
      pageSize: listPageSize,
      symbol,
    })
      .then((data) => {
        if (!cancelled) setState(listStateFromResponse(data, kind));
      })
      .catch((error) => {
        if (!cancelled) setState(listErrorState(error));
      });
    return () => {
      cancelled = true;
    };
  }, [category, debouncedQuery, duration, kind, listPage, listPageSize, reloadKey, symbol]);

  return state;
}

function initialListState() {
  return {
    categories: [],
    comboFactors: [],
    comboTotal: 0,
    factors: [],
    listTotal: 0,
    page: 1,
    pageCount: 1,
    sourceSummary: {},
    status: "加载中…",
    total: 0,
  };
}

function listStateFromResponse(data, kind) {
  const total = data.total ?? 0;
  const listTotal = data.total ?? data.listTotal ?? data.factors?.length ?? 0;
  return {
    categories: Array.isArray(data.categories) ? data.categories : [],
    comboFactors: Array.isArray(data.comboFactors) ? data.comboFactors : [],
    comboTotal: data.comboTotal ?? 0,
    factors: Array.isArray(data.factors) ? data.factors : [],
    listTotal,
    page: data.page ?? 1,
    pageCount: data.pageCount ?? 1,
    sourceSummary: data.sourceSummary ?? {},
    status: listStatusText(kind, listTotal),
    total: kind === "combo" ? data.singleTotal ?? 0 : data.unfilteredTotal ?? total,
  };
}

function listErrorState(error) {
  return {
    ...initialListState(),
    status: `列表失败：${error.message}`,
  };
}

function listStatusText(kind, listTotal) {
  if (kind === "combo") {
    return `已加载组合因子 ${listTotal} 条`;
  }
  return `已加载单因子 ${listTotal} 条`;
}
