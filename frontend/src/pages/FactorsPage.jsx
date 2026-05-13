import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchFactorBacktest,
  fetchFactorDetail,
  fetchFactorsList,
  fetchFactorRanking,
  requestFactorRankingRefresh,
} from "../api/client";
import FactorCombinationPanel from "../components/FactorCombinationPanel";
import FactorLearningPanel from "../components/FactorLearningPanel";
import "./FactorsPage.css";

const DURATIONS = [
  { value: "10m", label: "10 分钟" },
  { value: "30m", label: "30 分钟" },
  { value: "60m", label: "60 分钟" },
  { value: "1d", label: "1 天" },
];

function directionLabel(v) {
  if (v === "higher_better") return "高优";
  if (v === "lower_better") return "低优";
  return "中性";
}

function formatNum(n, digits = 4) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}

function formatPct(n, digits = 1) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return `${(Number(n) * 100).toFixed(digits)}%`;
}

function factorTitle(f) {
  return f?.displayName || f?.factorDisplayName || f?.description || f?.name || "未命名因子";
}

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
        <section className="factors-list-panel card-surface">
          <div className="section-head factors-section-head">
            <div>
              <span className="section-kicker">筛选</span>
              <h2>因子列表</h2>
            </div>
            <span className="factors-count">{total} 项</span>
          </div>
          <div className="factors-chips" role="tablist" aria-label="因子分类">
            <button
              type="button"
              className={`factors-chip${category === "" ? " factors-chip-active" : ""}`}
              onClick={() => setCategory("")}
            >
              全部
            </button>
            {categories.map((c) => (
              <button
                key={c.key}
                type="button"
                className={`factors-chip${category === c.key ? " factors-chip-active" : ""}`}
                onClick={() => setCategory(c.key)}
                title={`${c.count ?? 0} 个`}
              >
                {c.name}
                <span className="factors-chip-meta">{c.count ?? 0}</span>
              </button>
            ))}
          </div>
          <label className="factors-search">
            搜索
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="名称或描述…"
            />
          </label>
          <div className="factors-table-wrap">
            <table className="factors-table">
              <thead>
                <tr>
                  <th>中文因子</th>
                  <th>分类</th>
                  <th>方向</th>
                </tr>
              </thead>
              <tbody>
                {filteredFactors.map((f) => (
                  <tr
                    key={f.name}
                    className={selectedName === f.name ? "factors-row-selected" : ""}
                    onClick={() => setSelectedName(f.name)}
                  >
                    <td>
                      <strong className="factors-name-cn">{factorTitle(f)}</strong>
                      <code className="factors-code">{f.name}</code>
                    </td>
                    <td>{f.categoryName || f.category}</td>
                    <td>{directionLabel(f.direction)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!filteredFactors.length ? (
              <p className="factors-empty">无匹配因子</p>
            ) : null}
          </div>
        </section>

        <section className="factors-detail-panel card-surface">
          <div className="section-head factors-section-head">
            <div>
              <span className="section-kicker">详情</span>
              <h2>{detail ? factorTitle(detail) : selectedName || "请选择因子"}</h2>
            </div>
          </div>
          {detailError ? <p className="factors-error">{detailError}</p> : null}
          {detail ? (
            <dl className="factors-dl">
              <div>
                <dt>英文/字段名</dt>
                <dd>
                  <code className="factors-code">{detail.name}</code>
                </dd>
              </div>
              <div>
                <dt>说明</dt>
                <dd>{detail.description}</dd>
              </div>
              <div>
                <dt>公式</dt>
                <dd>
                  <code className="factors-formula">{detail.formula}</code>
                </dd>
              </div>
              <div>
                <dt>周期</dt>
                <dd>{(detail.timeframes || []).join(", ") || "—"}</dd>
              </div>
              <div>
                <dt>方向</dt>
                <dd>{directionLabel(detail.direction)}</dd>
              </div>
              <div>
                <dt>源码</dt>
                <dd>
                  <code className="factors-code">{detail.sourceFile}</code>
                </dd>
              </div>
            </dl>
          ) : null}
          {!selectedName ? (
            <p className="factors-placeholder">在左侧表格中点击一行查看定义与回测。</p>
          ) : null}

          {selectedName ? (
            <div className="factors-backtest-block">
              <h3 className="factors-subhead">单因子回测</h3>
              <button type="button" disabled={backtestLoading} onClick={runBacktest}>
                {backtestLoading ? "计算中…" : "运行回测"}
              </button>
              {backtestError ? <p className="factors-error">{backtestError}</p> : null}
              {backtest ? (
                <dl className="factors-metrics">
                  <div>
                    <dt>样本期数</dt>
                    <dd>{backtest.totalPeriods}</dd>
                  </div>
                  <div>
                    <dt>IC 均值</dt>
                    <dd>{formatNum(backtest.icMean, 6)}</dd>
                  </div>
                  <div>
                    <dt>IC 标准差</dt>
                    <dd>{formatNum(backtest.icStd, 6)}</dd>
                  </div>
                  <div>
                    <dt>IR</dt>
                    <dd>{formatNum(backtest.ir, 4)}</dd>
                  </div>
                  <div>
                    <dt>IC&gt;0 占比</dt>
                    <dd>{formatNum(backtest.icPositiveRate, 4)}</dd>
                  </div>
                  <div>
                    <dt>多空收益</dt>
                    <dd>{formatNum(backtest.longShortReturn, 6)}</dd>
                  </div>
                  <div>
                    <dt>因子夏普</dt>
                    <dd>{formatNum(backtest.sharpe, 4)}</dd>
                  </div>
                  <div>
                    <dt>胜率</dt>
                    <dd>{formatPct(backtest.winRate, 1)}</dd>
                  </div>
                  <div>
                    <dt>最大回撤</dt>
                    <dd>{formatPct(backtest.maxDrawdown, 2)}</dd>
                  </div>
                  <div>
                    <dt>盈亏比</dt>
                    <dd>{formatNum(backtest.profitFactor, 4)}</dd>
                  </div>
                  <div>
                    <dt>换手</dt>
                    <dd>{formatNum(backtest.turnover, 4)}</dd>
                  </div>
                  <div>
                    <dt>t 统计量</dt>
                    <dd>{formatNum(backtest.tStat, 4)}</dd>
                  </div>
                  <div>
                    <dt>p 值</dt>
                    <dd>{formatNum(backtest.pValue, 6)}</dd>
                  </div>
                </dl>
              ) : null}
            </div>
          ) : null}

          <div className="factors-ranking-block">
            <h3 className="factors-subhead">按 IR 排序（后台缓存 · 当前筛选与交易对）</h3>
            <div className="factors-ranking-wrap">
              <table className="factors-table factors-ranking-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>中文因子</th>
                    <th>类别</th>
                    <th>IR</th>
                    <th>贡献</th>
                    <th>夏普</th>
                    <th>胜率</th>
                    <th>IC 均值</th>
                    <th>多空</th>
                  </tr>
                </thead>
                <tbody>
                  {ranking.slice(0, 40).map((row, i) => (
                    <tr
                      key={row.factorName}
                      className={selectedName === row.factorName ? "factors-row-selected" : ""}
                      onClick={() => setSelectedName(row.factorName)}
                    >
                      <td>{i + 1}</td>
                      <td>
                        <strong className="factors-name-cn">{factorTitle(row)}</strong>
                        <code className="factors-code">{row.factorName}</code>
                      </td>
                      <td>{row.categoryName || row.category}</td>
                      <td>{formatNum(row.ir, 4)}</td>
                      <td>{formatPct(row.contribution, 1)}</td>
                      <td>{formatNum(row.sharpe, 2)}</td>
                      <td>{formatPct(row.winRate, 0)}</td>
                      <td>{formatNum(row.icMean, 4)}</td>
                      <td>{formatNum(row.longShortReturn, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!ranking.length ? <p className="factors-empty factors-ranking-empty">暂无排名数据</p> : null}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
