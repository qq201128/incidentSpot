import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFactorBacktest,
  fetchFactorDetail,
  fetchFactorsList,
  fetchFactorRanking,
  requestFactorRankingRefresh,
} from "../api/client";
import FactorCombinationPanel from "../components/FactorCombinationPanel";
import FactorDetailPanel from "../components/FactorDetailPanel";
import FactorLearningPanel from "../components/FactorLearningPanel";
import FactorListPanel from "../components/FactorListPanel";
import "./FactorsPage.css";

const DURATIONS = [
  { value: "10m", label: "10 分钟" },
  { value: "30m", label: "30 分钟" },
  { value: "60m", label: "60 分钟" },
  { value: "1d", label: "1 天" },
];

export default function FactorsPage() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [duration, setDuration] = useState("10m");
  const [category, setCategory] = useState("");
  const [listStatus, setListStatus] = useState("加载中…");
  const [factors, setFactors] = useState([]);
  const [categories, setCategories] = useState([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [selectedName, setSelectedName] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState("");
  const [ranking, setRanking] = useState([]);
  const [rankStatus, setRankStatus] = useState("");
  const [backtest, setBacktest] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState("");

  const loadList = useCallback(async () => {
    setListStatus("加载中…");
    try {
      const data = await fetchFactorsList(category || undefined);
      setFactors(Array.isArray(data.factors) ? data.factors : []);
      setCategories(Array.isArray(data.categories) ? data.categories : []);
      setTotal(data.total ?? 0);
      setListStatus(`已加载 ${data.total ?? 0} 个因子`);
    } catch (e) {
      setListStatus(`列表失败：${e.message}`);
      setFactors([]);
    }
  }, [category]);

  useEffect(() => {
    loadList();
  }, [loadList]);

  useEffect(() => {
    if (!selectedName) {
      setDetail(null);
      setDetailError("");
      setBacktest(null);
      setBacktestError("");
      return;
    }
    let cancelled = false;
    setDetail(null);
    setDetailError("");
    setBacktest(null);
    setBacktestError("");
    fetchFactorDetail(selectedName)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e) => {
        if (!cancelled) setDetailError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedName]);

  const rankingAbortRef = useRef(null);
  /** 防止旧请求/被取消的请求在「正在计算…」之后不写回状态，导致文案卡死。 */
  const rankingSeqRef = useRef(0);
  const loadRankingRef = useRef(async () => {});

  const loadRanking = useCallback(async () => {
    const seq = ++rankingSeqRef.current;
    const sym = symbol.trim().toUpperCase();
    if (sym.length < 6) {
      if (rankingSeqRef.current === seq) {
        setRankStatus("请输入有效交易对（如 BTCUSDT）");
        setRanking([]);
      }
      return;
    }
    rankingAbortRef.current?.abort();
    const ac = new AbortController();
    rankingAbortRef.current = ac;

    if (rankingSeqRef.current !== seq) {
      return;
    }
    setRankStatus("加载排名缓存…");
    setRanking([]);
    try {
      const data = await fetchFactorRanking(sym, duration, category || undefined, {
        signal: ac.signal,
      });
      if (rankingSeqRef.current !== seq) {
        return;
      }
      if (ac.signal.aborted) {
        return;
      }
      setRanking(Array.isArray(data.ranking) ? data.ranking : []);
      const src = data.source === "cache" ? "后台缓存" : "无缓存";
      const when = data.updatedAt ? ` · 更新 ${data.updatedAt}` : "";
      const total = data.total ?? 0;
      if (data.source === "none") {
        const extra = Array.isArray(data.precomputedSymbols)
          ? `预计算交易对：${data.precomputedSymbols.join(", ")}。`
          : "";
        setRankStatus(
          `暂无该组合的排名缓存（${sym} / ${duration}）。${extra}可使用「请求后台刷新」或调整 FACTOR_RANKING_SYMBOLS。`,
        );
      } else {
        setRankStatus(`排名：${total} 个因子（${sym} / ${duration} · ${src}${when}）`);
      }
    } catch (e) {
      const aborted =
        e?.code === "ERR_CANCELED" ||
        e?.code === "ECONNABORTED" ||
        e?.name === "CanceledError" ||
        ac.signal.aborted;
      if (aborted) {
        return;
      }
      if (rankingSeqRef.current !== seq) {
        return;
      }
      setRankStatus(`排名失败：${e.message}`);
    }
  }, [symbol, duration, category]);

  loadRankingRef.current = loadRanking;

  /**
   * 交易对输入会高频改 symbol：防抖后再请求。
   * 通过 ref 调用最新 loadRanking，依赖仅 symbol/duration/category，避免 effect 因函数引用抖动反复清定时器。
   */
  useEffect(() => {
    const sym = symbol.trim().toUpperCase();
    if (sym.length < 6) {
      setRankStatus("请输入有效交易对（如 BTCUSDT）");
      setRanking([]);
      return;
    }
    const delayMs = 320;
    const t = window.setTimeout(() => {
      void loadRankingRef.current();
    }, delayMs);
    return () => {
      window.clearTimeout(t);
    };
  }, [symbol, duration, category]);

  const filteredFactors = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return factors;
    return factors.filter(
      (f) =>
        (f.name && f.name.toLowerCase().includes(q)) ||
        (f.description && f.description.toLowerCase().includes(q)) ||
        (f.categoryName && f.categoryName.toLowerCase().includes(q)),
    );
  }, [factors, query]);

  async function runBacktest() {
    if (!selectedName) return;
    const sym = symbol.trim().toUpperCase();
    if (sym.length < 6) {
      setBacktestError("请输入有效交易对");
      return;
    }
    setBacktestLoading(true);
    setBacktestError("");
    setBacktest(null);
    try {
      const data = await fetchFactorBacktest(selectedName, sym, duration);
      setBacktest(data);
    } catch (e) {
      setBacktestError(e.message);
    } finally {
      setBacktestLoading(false);
    }
  }

  return (
    <main className="factors-page layout">
      <header className="factors-topbar topbar">
        <div>
          <span className="eyebrow">因子库</span>
          <h1>量化因子目录与回测</h1>
        </div>
        <p className="status-pill factors-status-pill">{listStatus}</p>
      </header>

      <section className="factors-toolbar card-surface">
        <div className="factors-toolbar-row">
          <label>
            交易对
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="BTCUSDT"
            />
          </label>
          <label>
            规则周期
            <select value={duration} onChange={(e) => setDuration(e.target.value)}>
              {DURATIONS.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="factors-btn-secondary"
            onClick={async () => {
              const sym = symbol.trim().toUpperCase();
              if (sym.length < 6) {
                setRankStatus("请输入有效交易对（如 BTCUSDT）");
                return;
              }
              try {
                setRankStatus("已排队后台重算，请稍候再点或等待自动加载…");
                await requestFactorRankingRefresh(sym);
                window.setTimeout(() => {
                  void loadRankingRef.current();
                }, 2500);
              } catch (e) {
                setRankStatus(`排队刷新失败：${e.message}`);
              }
            }}
          >
            请求后台刷新排名
          </button>
        </div>
        <p className="factors-rank-hint">{rankStatus}</p>
      </section>

      <FactorCombinationPanel symbol={symbol} duration={duration} />
      <FactorLearningPanel symbol={symbol} duration={duration} />

      <div className="factors-grid">
        <FactorListPanel
          categories={categories}
          category={category}
          factors={filteredFactors}
          onCategoryChange={setCategory}
          onQueryChange={setQuery}
          onSelectFactor={setSelectedName}
          query={query}
          selectedName={selectedName}
          total={total}
        />
        <FactorDetailPanel
          backtest={backtest}
          backtestError={backtestError}
          backtestLoading={backtestLoading}
          detail={detail}
          detailError={detailError}
          onRunBacktest={runBacktest}
          onSelectFactor={setSelectedName}
          ranking={ranking}
          selectedName={selectedName}
        />
      </div>
    </main>
  );
}
