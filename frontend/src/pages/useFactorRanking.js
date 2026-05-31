import { useCallback, useEffect, useRef, useState } from "react";
import { fetchFactorRanking, requestFactorRankingRefresh } from "../api/client";

const MIN_SYMBOL_LENGTH = 6;
const RANKING_DEBOUNCE_MS = 320;
const RANKING_PAGE_SIZE = 8;
const RANKING_REFRESH_DELAY_MS = 2500;

export function useFactorRanking({ category, duration, symbol }) {
  const [state, setState] = useState(initialRankingState);
  const abortRef = useRef(null);
  const seqRef = useRef(0);
  const loadRef = useRef(async () => {});
  const debouncedQuery = useDebouncedRankingQuery(state.query);

  useEffect(() => {
    setState((prev) => ({ ...prev, page: 1 }));
  }, [category, debouncedQuery, duration, symbol]);

  const load = useCallback(async () => {
    await loadRankingPage({
      abortRef,
      category,
      debouncedQuery,
      duration,
      page: state.page,
      seqRef,
      setState,
      symbol,
    });
  }, [category, debouncedQuery, duration, state.page, symbol]);

  loadRef.current = load;
  useDebouncedRankingLoad({ category, debouncedQuery, duration, loadRef, setState, state, symbol });
  const requestRefresh = useRankingRefresh({ loadRef, setState, symbol });
  return {
    ...state,
    requestRefresh,
    setPage: (page) => setState((prev) => ({ ...prev, page })),
    setQuery: (query) => setState((prev) => ({ ...prev, query })),
  };
}

function initialRankingState() {
  return {
    items: [],
    page: 1,
    pageCount: 1,
    query: "",
    status: "",
    total: 0,
    unfilteredTotal: 0,
  };
}

async function loadRankingPage(options) {
  const { abortRef, category, debouncedQuery, duration, page, seqRef, setState, symbol } = options;
  const seq = ++seqRef.current;
  const sym = normalizeSymbol(symbol);
  if (!validateRankingSymbol({ seq, seqRef, setState, sym })) return;
  abortRef.current?.abort();
  const ac = new AbortController();
  abortRef.current = ac;
  setState((prev) => ({ ...prev, items: [], status: "加载排名缓存…" }));
  try {
    const data = await requestRankingPage({ ac, category, debouncedQuery, duration, page, sym });
    if (seqRef.current !== seq || ac.signal.aborted) return;
    setState((prev) => rankingStateFromResponse({ data, duration, prev, sym }));
  } catch (error) {
    if (isAbortError(error, ac.signal) || seqRef.current !== seq) return;
    setState((prev) => ({ ...prev, status: `排名失败：${error.message}` }));
  }
}

function requestRankingPage({ ac, category, debouncedQuery, duration, page, sym }) {
  return fetchFactorRanking(sym, duration, category || undefined, {
    page,
    pageSize: RANKING_PAGE_SIZE,
    q: debouncedQuery.trim() || undefined,
    signal: ac.signal,
  });
}

function rankingStateFromResponse({ data, duration, prev, sym }) {
  return {
    items: rankingRowsWithDuration(data, duration),
    page: data.page ?? prev.page,
    pageCount: data.pageCount ?? 1,
    query: prev.query,
    status: formatRankingStatus(data, sym, duration),
    total: data.total ?? 0,
    unfilteredTotal: data.unfilteredTotal ?? data.total ?? 0,
  };
}

function rankingRowsWithDuration(data, duration) {
  return (Array.isArray(data.ranking) ? data.ranking : []).map((row) => ({
    ...row,
    duration: data.duration || duration,
  }));
}

function useDebouncedRankingQuery(query) {
  const [debounced, setDebounced] = useState(query);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(query), RANKING_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [query]);
  return debounced;
}

function useDebouncedRankingLoad({ category, debouncedQuery, duration, loadRef, setState, state, symbol }) {
  useEffect(() => {
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setState((prev) => ({ ...prev, ...emptyRankingState(), status: "请输入有效交易对（如 BTCUSDT）" }));
      return;
    }
    const timer = window.setTimeout(() => {
      void loadRef.current();
    }, RANKING_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [category, debouncedQuery, duration, loadRef, setState, state.page, symbol]);
}

function useRankingRefresh({ loadRef, setState, symbol }) {
  return useCallback(async () => {
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setState((prev) => ({ ...prev, status: "请输入有效交易对（如 BTCUSDT）" }));
      return;
    }
    try {
      setState((prev) => ({ ...prev, status: "已排队后台重算，请稍候…" }));
      await requestFactorRankingRefresh(sym);
      window.setTimeout(() => void loadRef.current(), RANKING_REFRESH_DELAY_MS);
    } catch (error) {
      setState((prev) => ({ ...prev, status: `排队刷新失败：${error.message}` }));
    }
  }, [loadRef, setState, symbol]);
}

function validateRankingSymbol({ seq, seqRef, setState, sym }) {
  if (sym.length >= MIN_SYMBOL_LENGTH) return true;
  if (seqRef.current === seq) {
    setState((prev) => ({ ...prev, ...emptyRankingState(), status: "请输入有效交易对（如 BTCUSDT）" }));
  }
  return false;
}

function emptyRankingState() {
  return {
    items: [],
    pageCount: 1,
    total: 0,
    unfilteredTotal: 0,
  };
}

function formatRankingStatus(data, sym, duration) {
  if (data.source === "none") {
    return formatEmptyRankingStatus(data, sym, duration);
  }
  const src = data.source === "cache" ? "后台缓存" : "无缓存";
  const when = data.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
  return `排名：${data.total ?? 0} 个因子（${sym} / ${duration} · ${src}${when}）`;
}

function formatEmptyRankingStatus(data, sym, duration) {
  const extra = Array.isArray(data.precomputedSymbols)
    ? `预计算交易对：${data.precomputedSymbols.join(", ")}。`
    : "";
  return `暂无该组合的排名缓存（${sym} / ${duration}）。${extra}`;
}

function isAbortError(error, signal) {
  return (
    error?.code === "ERR_CANCELED" ||
    error?.code === "ECONNABORTED" ||
    error?.name === "CanceledError" ||
    signal.aborted
  );
}

function normalizeSymbol(symbol) {
  return symbol.trim().toUpperCase();
}
