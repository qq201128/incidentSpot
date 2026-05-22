import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFactorBacktest,
  fetchFactorDetail,
  fetchFactorsList,
  fetchFactorRanking,
  requestFactorRankingRefresh,
} from "../api/client";
import { fetchFactorPageOverview, fetchFactorPeriodScores } from "../api/factorPageClient";

const DEFAULT_SYMBOL = "BTCUSDT";
const DEFAULT_DURATION = "10m";
const MIN_SYMBOL_LENGTH = 6;
const RANKING_DEBOUNCE_MS = 320;
const RANKING_REFRESH_DELAY_MS = 2500;
const LIST_DEBOUNCE_MS = 280;

export function useFactorsPageData() {
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOL);
  const [duration, setDuration] = useState(DEFAULT_DURATION);
  const [previewDuration, setPreviewDuration] = useState(DEFAULT_DURATION);
  const [category, setCategory] = useState("");
  const [query, setQuery] = useState("");
  const [listTab, setListTab] = useState("single");
  const [listPage, setListPage] = useState(1);
  const [listPageSize, setListPageSize] = useState(20);
  const [selectedName, setSelectedName] = useState(null);
  const [overview, setOverview] = useState(null);
  const [periodScoresState, setPeriodScoresState] = useState({ factorName: null, scores: [] });
  const backtest = useBacktest({ duration, selectedName, symbol });
  const [listReloadKey, setListReloadKey] = useState(0);
  const list = useFactorsList({
    category,
    kind: listTab,
    listPage,
    listPageSize,
    query,
    reloadKey: listReloadKey,
  });
  const detail = useFactorDetail(selectedName, symbol, duration);
  const previewMetrics = usePreviewMetrics(selectedName, symbol, previewDuration);
  const ranking = useFactorRanking({ category, duration, symbol });
  const debouncedQuery = useDebouncedValue(query, LIST_DEBOUNCE_MS);

  const selectedFactor = useMemo(() => {
    if (!selectedName) return null;
    const fromList = list.factors.find((row) => row.name === selectedName);
    if (fromList) return fromList;
    const fromRanking = ranking.items.find(
      (row) => row.factorName === selectedName || row.name === selectedName,
    );
    return fromRanking ? rankingRowToFactorSnapshot(fromRanking) : null;
  }, [list.factors, ranking.items, selectedName]);

  useEffect(() => {
    setPreviewDuration(defaultPreviewDuration(selectedFactor, duration));
  }, [duration, selectedFactor, selectedName]);

  useEffect(() => {
    setListPage(1);
  }, [category, debouncedQuery, listTab, listPageSize]);

  useEffect(() => {
    const first = list.factors[0]?.name;
    if (!first) {
      setSelectedName(null);
      return;
    }
    if (!selectedName) setSelectedName(first);
  }, [list.factors, listTab, selectedName]);

  useEffect(() => {
    let cancelled = false;
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setOverview(null);
      return undefined;
    }
    fetchFactorPageOverview(sym, duration, category || undefined)
      .then((data) => {
        if (!cancelled) setOverview(data);
      })
      .catch(() => {
        if (!cancelled) setOverview(null);
      });
    return () => {
      cancelled = true;
    };
  }, [symbol, duration, category]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedName) {
      setPeriodScoresState({ factorName: null, scores: [] });
      return undefined;
    }
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setPeriodScoresState({ factorName: null, scores: [] });
      return undefined;
    }
    const factorName = selectedName;
    fetchFactorPeriodScores(factorName, sym)
      .then((data) => {
        if (cancelled) return;
        setPeriodScoresState({
          factorName,
          scores: Array.isArray(data.scores) ? data.scores : [],
        });
      })
      .catch(() => {
        if (!cancelled) {
          setPeriodScoresState((prev) =>
            prev.factorName === factorName ? { factorName, scores: [] } : prev,
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedName, symbol]);

  const periodScores = periodScoresState.scores;
  const periodScoresPending = Boolean(
    selectedName && periodScoresState.factorName && periodScoresState.factorName !== selectedName,
  );

  const cachedMetrics = useMemo(() => {
    if (!selectedName) return null;
    return ranking.items.find((row) => row.factorName === selectedName || row.name === selectedName) || null;
  }, [ranking.items, selectedName]);

  const displayMetrics = useMemo(() => {
    if (metricsCanDisplay(backtest.data, selectedName)) return backtest.data;
    if (metricsCanDisplay(previewMetrics, selectedName)) return previewMetrics;
    if (metricsCanDisplay(detail.data, selectedName)) return detail.data;
    return cachedMetrics;
  }, [backtest.data, cachedMetrics, detail.data, previewMetrics, selectedName]);

  const rankingTotal = overview?.rankingTotal ?? ranking.items.length;

  return {
    actions: {
      reloadList: () => setListReloadKey((value) => value + 1),
      requestRankingRefresh: ranking.requestRefresh,
      runBacktest: backtest.run,
      setCategory,
      setDuration,
      setListPage,
      setListPageSize,
      setListTab,
      setPreviewDuration,
      setQuery,
      setSelectedName,
      setSymbol,
    },
    animationKeys: {
      listKey: `${list.factors.length}:${listTab}:${listPage}:${debouncedQuery}:${category}`,
      rankingKey: `${ranking.items.length}:${ranking.status}`,
    },
    state: {
      alerts: overview?.alerts ?? [],
      backtest,
      cachedMetrics,
      categories: list.categories,
      category,
      comboTotal: list.comboTotal,
      detail,
      displayMetrics,
      duration,
      filteredComboFactors: list.comboFactors,
      filteredFactors: list.factors,
      highWinrateCombo: overview?.highWinrateCombo ?? null,
      listPage,
      listPageCount: list.pageCount,
      listPageSize,
      listTab,
      listTotal: list.listTotal,
      overview,
      periodScores,
      periodScoresPending,
      previewDuration,
      query,
      ranking,
      rankingTotal,
      selectedFactor,
      selectedName,
      sourceSummary: overview?.sourceSummary ?? list.sourceSummary ?? {},
      sourceSummaryGlobal: overview?.sourceSummaryGlobal ?? {},
      symbol,
      total: list.total,
    },
  };
}

function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [delayMs, value]);
  return debounced;
}

function useFactorsList({ category, kind, listPage, listPageSize, query, reloadKey }) {
  const debouncedQuery = useDebouncedValue(query, LIST_DEBOUNCE_MS);
  const [status, setStatus] = useState("加载中…");
  const [factors, setFactors] = useState([]);
  const [comboFactors, setComboFactors] = useState([]);
  const [categories, setCategories] = useState([]);
  const [total, setTotal] = useState(0);
  const [comboTotal, setComboTotal] = useState(0);
  const [listTotal, setListTotal] = useState(0);
  const [pageCount, setPageCount] = useState(1);
  const [sourceSummary, setSourceSummary] = useState({});

  useEffect(() => {
    let cancelled = false;
    setStatus("加载中…");
    fetchFactorsList({
      category: category || undefined,
      kind,
      q: debouncedQuery.trim() || undefined,
      page: listPage,
      pageSize: listPageSize,
    })
      .then((data) => {
        if (cancelled) return;
        setFactors(Array.isArray(data.factors) ? data.factors : []);
        setComboFactors(Array.isArray(data.comboFactors) ? data.comboFactors : []);
        setCategories(Array.isArray(data.categories) ? data.categories : []);
        setTotal(data.total ?? 0);
        setComboTotal(data.comboTotal ?? 0);
        setListTotal(data.listTotal ?? data.factors?.length ?? 0);
        setPageCount(data.pageCount ?? 1);
        setSourceSummary(data.sourceSummary ?? {});
        setStatus(
          kind === "combo"
            ? `已加载组合因子 ${data.listTotal ?? 0} 条`
            : `已加载单因子 ${data.total ?? 0} 条`,
        );
      })
      .catch((error) => {
        if (cancelled) return;
        setStatus(`列表失败：${error.message}`);
        setCategories([]);
        setFactors([]);
        setComboFactors([]);
        setTotal(0);
        setComboTotal(0);
        setListTotal(0);
        setPageCount(1);
        setSourceSummary({});
      });
    return () => {
      cancelled = true;
    };
  }, [category, debouncedQuery, kind, listPage, listPageSize, reloadKey]);

  return {
    categories,
    comboFactors,
    comboTotal,
    factors,
    listTotal,
    pageCount,
    sourceSummary,
    status,
    total,
  };
}

function usePreviewMetrics(selectedName, symbol, previewDuration) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    if (!selectedName) {
      setMetrics(null);
      return undefined;
    }
    const sym = normalizeSymbol(symbol);
    if (sym.length < MIN_SYMBOL_LENGTH) {
      setMetrics(null);
      return undefined;
    }
    const factorName = selectedName;
    let cancelled = false;
    fetchFactorDetail(factorName, sym, previewDuration)
      .then((detail) => {
        if (cancelled || detail?.name !== factorName) return;
        setMetrics(hasRankingMetrics(detail) ? detail : null);
      })
      .catch(() => {
        if (!cancelled) setMetrics((prev) => (metricsMatchFactor(prev, factorName) ? prev : null));
      });
    return () => {
      cancelled = true;
    };
  }, [previewDuration, selectedName, symbol]);

  return metrics;
}

function hasRankingMetrics(row) {
  return row && (row.factorScore != null || row.winRate != null || row.icMean != null);
}

function useBacktest({ duration, selectedName, symbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setData(null);
    setError("");
  }, [selectedName]);

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

  return { data, error, loading, run };
}

function useFactorDetail(selectedName, symbol, duration) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedName) {
      setData(null);
      setError("");
      setLoading(false);
      return undefined;
    }
    const factorName = selectedName;
    let cancelled = false;
    setLoading(true);
    setError("");
    const sym = normalizeSymbol(symbol);
    fetchFactorDetail(factorName, sym.length >= MIN_SYMBOL_LENGTH ? sym : undefined, duration)
      .then((detail) => {
        if (cancelled || detail?.name !== factorName) return;
        setData(detail);
        setLoading(false);
      })
      .catch((requestError) => {
        if (cancelled) return;
        setError(requestError.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [duration, selectedName, symbol]);

  return { data, error, loading };
}

function metricsMatchFactor(metrics, factorName) {
  if (!metrics || !factorName) return false;
  const name = metrics.factorName || metrics.name;
  return name === factorName;
}

function metricsCanDisplay(metrics, factorName) {
  return metricsMatchFactor(metrics, factorName) && hasRankingMetrics(metrics);
}

function defaultPreviewDuration(factor, fallback) {
  const duration = factor?.duration || factor?.timeframes?.[0];
  return duration || fallback;
}

function rankingRowToFactorSnapshot(row) {
  const name = row.factorName || row.name;
  if (!name) return null;
  return {
    ...row,
    name,
    displayName: row.displayName || row.factorDisplayName || row.description,
  };
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
      setItems(
        (Array.isArray(data.ranking) ? data.ranking : []).map((row) => ({
          ...row,
          duration: data.duration || duration,
        })),
      );
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
      setStatus("已排队后台重算，请稍候…");
      await requestFactorRankingRefresh(sym);
      window.setTimeout(() => {
        void loadRef.current();
      }, RANKING_REFRESH_DELAY_MS);
    } catch (error) {
      setStatus(`排队刷新失败：${error.message}`);
    }
  }, [loadRef, setStatus, symbol]);
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
  return `暂无该组合的排名缓存（${sym} / ${duration}）。${extra}`;
}

function isAbortError(error, signal) {
  return error?.code === "ERR_CANCELED" || error?.code === "ECONNABORTED" || error?.name === "CanceledError" || signal.aborted;
}

function normalizeSymbol(symbol) {
  return symbol.trim().toUpperCase();
}
