import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchFactorBacktest,
  fetchFactorDetail,
} from "../api/client";
import { fetchFactorPageOverview, fetchFactorPeriodScores } from "../api/factorPageClient";
import { useDebouncedValue, useFactorsList } from "./useFactorList";
import { useFactorRanking } from "./useFactorRanking";

const DEFAULT_SYMBOL = "BTCUSDT";
const DEFAULT_DURATION = "10m";
const MIN_SYMBOL_LENGTH = 6;

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
    duration,
    kind: listTab,
    listPage,
    listPageSize,
    query,
    reloadKey: listReloadKey,
    symbol: normalizeSymbol(symbol),
  });
  const detail = useFactorDetail(selectedName, symbol, duration);
  const previewMetrics = usePreviewMetrics(selectedName, symbol, previewDuration);
  const ranking = useFactorRanking({ category, duration, symbol });
  const debouncedQuery = useDebouncedValue(query);

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
    if (list.page !== listPage) setListPage(list.page);
  }, [list.page, listPage]);

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
      setRankingPage: ranking.setPage,
      setRankingQuery: ranking.setQuery,
      setSelectedName,
      setSymbol,
    },
    animationKeys: {
      listKey: `${list.factors.length}:${listTab}:${listPage}:${debouncedQuery}:${category}`,
      rankingKey: `${ranking.items.length}:${ranking.status}:${ranking.page}:${ranking.query}`,
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

function normalizeSymbol(symbol) {
  return symbol.trim().toUpperCase();
}
