import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFactorBacktest,
  fetchFactorDetail,
  fetchFactorsList,
  fetchFactorRanking,
  requestFactorRankingRefresh,
} from "../api/client";

const DEFAULT_SYMBOL = "BTCUSDT";
const DEFAULT_DURATION = "10m";
const MIN_SYMBOL_LENGTH = 6;
const RANKING_DEBOUNCE_MS = 320;
const RANKING_REFRESH_DELAY_MS = 2500;

export function useFactorsPageData() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOL);
  const [duration, setDuration] = useState(DEFAULT_DURATION);
  const [category, setCategory] = useState("");
  const [query, setQuery] = useState("");
  const [selectedName, setSelectedName] = useState(null);
  const backtest = useBacktest({ duration, selectedName, symbol });
  const list = useFactorsList(category);
  const detail = useFactorDetail(selectedName, backtest.reset);
  const ranking = useFactorRanking({ category, duration, symbol });
  const filteredFactors = useFilteredFactors(list.factors, query);

  return {
    actions: {
      requestRankingRefresh: ranking.requestRefresh,
      runBacktest: backtest.run,
      setCategory,
      setDuration,
      setQuery,
      setSelectedName,
      setSymbol,
    },
    animationKeys: {
      backtestKey: `${backtest.loading}:${backtest.data?.factorScore ?? ""}:${backtest.error}`,
      detailKey: `${selectedName ?? ""}:${detail.data?.name ?? ""}:${detail.error}`,
      listKey: `${filteredFactors.length}:${query}:${category}`,
      rankingKey: `${ranking.items.length}:${ranking.status}`,
    },
    state: {
      backtest,
      categories: list.categories,
      category,
      detail,
      duration,
      filteredFactors,
      listStatus: list.status,
      query,
      ranking,
      selectedName,
      symbol,
      total: list.total,
    },
  };
}

function useFactorsList(category) {
  const [status, setStatus] = useState("加载中…");
  const [factors, setFactors] = useState([]);
  const [categories, setCategories] = useState([]);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    fetchFactorsList(category || undefined)
      .then((data) => {
        setFactors(Array.isArray(data.factors) ? data.factors : []);
        setCategories(Array.isArray(data.categories) ? data.categories : []);
        setTotal(data.total ?? 0);
        setStatus(`已加载 ${data.total ?? 0} 个因子`);
      })
      .catch((error) => {
        setStatus(`列表失败：${error.message}`);
        setFactors([]);
      });
  }, [category]);

  return { categories, factors, status, total };
}

function useBacktest({ duration, selectedName, symbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const reset = useCallback(() => {
    setData(null);
    setError("");
  }, []);

  const run = useCallback(async () => {
    if (!selectedName) return;
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setError("请输入有效交易对");
      return;
    }
    setLoading(true);
    setError("");
    setData(null);
    try {
      setData(await fetchFactorBacktest(selectedName, sym, duration));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [duration, selectedName, symbol]);

  return { data, error, loading, reset, run };
}

function useFactorDetail(selectedName, resetBacktest) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    resetBacktest();
    if (!selectedName) {
      setData(null);
      setError("");
      return;
    }
    let cancelled = false;
    setData(null);
    setError("");
    fetchFactorDetail(selectedName)
      .then((detail) => {
        if (!cancelled) setData(detail);
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError.message);
      });
    return () => {
      cancelled = true;
    };
  }, [resetBacktest, selectedName]);

  return { data, error };
}

function useFactorRanking({ category, duration, symbol }) {
  const [items, setItems] = useState([]);
  const [status, setStatus] = useState("");
  const abortRef = useRef(null);
  const seqRef = useRef(0);
  const loadRef = useRef(async () => {});

  const load = useCallback(async () => {
    const seq = ++seqRef.current;
    const sym = normalizeSymbol(symbol);
    if (!validateRankingSymbol({ seq, seqRef, setItems, setStatus, sym })) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setStatus("加载排名缓存…");
    setItems([]);
    try {
      const data = await fetchFactorRanking(sym, duration, category || undefined, { signal: ac.signal });
      if (seqRef.current !== seq || ac.signal.aborted) return;
      setItems(Array.isArray(data.ranking) ? data.ranking : []);
      setStatus(formatRankingStatus(data, sym, duration));
    } catch (error) {
      if (isAbortError(error, ac.signal) || seqRef.current !== seq) return;
      setStatus(`排名失败：${error.message}`);
    }
  }, [category, duration, symbol]);

  loadRef.current = load;
  useDebouncedRankingLoad({ category, duration, loadRef, setItems, setStatus, symbol });
  const requestRefresh = useRankingRefresh({ loadRef, setStatus, symbol });
  return { items, requestRefresh, status };
}

function useDebouncedRankingLoad({ category, duration, loadRef, setItems, setStatus, symbol }) {
  useEffect(() => {
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setStatus("请输入有效交易对（如 BTCUSDT）");
      setItems([]);
      return;
    }
    const timer = window.setTimeout(() => {
      void loadRef.current();
    }, RANKING_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
    };
  }, [category, duration, loadRef, setItems, setStatus, symbol]);
}

function useRankingRefresh({ loadRef, setStatus, symbol }) {
  return useCallback(async () => {
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setStatus("请输入有效交易对（如 BTCUSDT）");
      return;
    }
    try {
      setStatus("已排队后台重算，请稍候再点或等待自动加载…");
      await requestFactorRankingRefresh(sym);
      window.setTimeout(() => {
        void loadRef.current();
      }, RANKING_REFRESH_DELAY_MS);
    } catch (error) {
      setStatus(`排队刷新失败：${error.message}`);
    }
  }, [loadRef, setStatus, symbol]);
}

function useFilteredFactors(factors, query) {
  return useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return factors;
    return factors.filter((factor) => factorMatchesQuery(factor, q));
  }, [factors, query]);
}

function factorMatchesQuery(factor, query) {
  return (
    valueIncludes(factor.name, query) ||
    valueIncludes(factor.description, query) ||
    valueIncludes(factor.categoryName, query)
  );
}

function validateRankingSymbol({ seq, seqRef, setItems, setStatus, sym }) {
  if (sym.length >= MIN_SYMBOL_LENGTH) return true;
  if (seqRef.current === seq) {
    setStatus("请输入有效交易对（如 BTCUSDT）");
    setItems([]);
  }
  return false;
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
  return `暂无该组合的排名缓存（${sym} / ${duration}）。${extra}可使用「请求后台刷新」或调整 FACTOR_RANKING_SYMBOLS。`;
}

function isAbortError(error, signal) {
  return error?.code === "ERR_CANCELED" || error?.code === "ECONNABORTED" || error?.name === "CanceledError" || signal.aborted;
}

function normalizeSymbol(symbol) {
  return symbol.trim().toUpperCase();
}

function valueIncludes(value, query) {
  return value ? value.toLowerCase().includes(query) : false;
}
