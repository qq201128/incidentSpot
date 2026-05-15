#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.env_file import load_backend_env_file
from app.services.factor_duration_alignment import duration_entry_rows, live_duration_entry_index
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, utc_now
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS
from app.services.rule_config import horizon_minutes_for_duration

TARGET_WIN_RATE = 0.70
TARGET_COUNT = 5
ZSCORE_CLIP = 4.0
SIGNAL_THRESHOLDS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
EXCLUDED_COLUMNS = frozenset({"open_time", "open", "high", "low", "close", "volume", "fwd_ret"})
REPORT_PATH = BACKEND_ROOT / "reports" / "factor_backtests" / "high_winrate_factor_combo_goal.json"
LIBRARY_PATH = BACKEND_ROOT / "models" / "factor_learning" / "high_winrate_factor_combo_goal_library.json"

ONLINE_RESEARCH_SOURCES = (
    {
        "title": "Explainable Patterns in Cryptocurrency Microstructure",
        "url": "https://arxiv.org/pdf/2602.00776",
        "factorFamilies": ["order flow imbalance", "spread", "VWAP-to-mid pressure"],
    },
    {
        "title": "Order flow and cryptocurrency returns",
        "url": "https://www.sciencedirect.com/science/article/pii/S1386418126000029",
        "factorFamilies": ["signed order flow", "buyer/seller initiated volume"],
    },
    {
        "title": "The Crypto Signal Compendium",
        "url": "https://the-algotrading-book-website.vercel.app/chapters/01-foundations/024-crypto-signal-compendium/",
        "factorFamilies": ["open interest", "long/short ratio", "taker buy/sell", "sentiment"],
    },
)


@dataclass(frozen=True)
class ComboHit:
    members: tuple[str, str]
    threshold: float
    win_rate: float
    profit_factor: float
    trades: int
    avg_return: float
    score: pd.Series


def run_goal(
    symbol: str,
    duration: str,
    target_count: int,
    output: Path,
    library: Path,
) -> dict[str, Any]:
    load_backend_env_file()
    frame = load_factor_frame(symbol, duration)
    search_frame = _search_frame(frame, duration)
    scores = _oriented_scores(search_frame)
    hits = _ranked_hits(search_frame, scores)
    selected = _selected_hits(hits, target_count)
    payload = _report_payload(symbol, duration, target_count, search_frame, scores, selected)
    _write_json(output, payload)
    _write_json(library, _library_payload(payload))
    if len(selected) < target_count:
        raise RuntimeError(f"only found {len(selected)} combos with winRate >= {TARGET_WIN_RATE}")
    return payload


def _search_frame(frame: pd.DataFrame, duration: str) -> pd.DataFrame:
    horizon = horizon_minutes_for_duration(duration)
    out = duration_entry_rows(frame.copy(), duration)
    out["fwd_ret"] = out["close"].shift(-horizon) / out["close"] - 1.0
    return out


def _oriented_scores(frame: pd.DataFrame) -> dict[str, pd.Series]:
    scores: dict[str, pd.Series] = {}
    for name in _numeric_factor_columns(frame):
        series = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if _usable_pair_count(series, frame["fwd_ret"]) < BACKTEST_MIN_PERIODS:
            continue
        orientation = _orientation(series, frame["fwd_ret"])
        scores[name] = _expanding_zscore(series) * orientation
    return scores


def _numeric_factor_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame.columns
        if name not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(frame[name])
    ]


def _usable_pair_count(series: pd.Series, fwd_ret: pd.Series) -> int:
    return len(pd.concat([series, fwd_ret], axis=1).dropna())


def _orientation(series: pd.Series, fwd_ret: pd.Series) -> int:
    valid = pd.concat([series, fwd_ret], axis=1).dropna()
    if valid.iloc[:, 0].nunique(dropna=True) < 2 or valid.iloc[:, 1].nunique(dropna=True) < 2:
        return 1
    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")
    return 1 if corr is not None and math.isfinite(float(corr)) and corr >= 0 else -1


def _expanding_zscore(series: pd.Series) -> pd.Series:
    mean = series.expanding(min_periods=BACKTEST_MIN_PERIODS).mean().shift(1)
    std = series.expanding(min_periods=BACKTEST_MIN_PERIODS).std().shift(1)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-ZSCORE_CLIP, ZSCORE_CLIP)


def _ranked_hits(frame: pd.DataFrame, scores: dict[str, pd.Series]) -> list[ComboHit]:
    hits: dict[tuple[str, str], ComboHit] = {}
    for left, right in combinations(scores, 2):
        best = _best_pair_hit(frame, left, right, scores)
        if best is not None:
            hits[best.members] = best
    rows = list(hits.values())
    rows.sort(key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), reverse=True)
    return rows


def _best_pair_hit(
    frame: pd.DataFrame,
    left: str,
    right: str,
    scores: dict[str, pd.Series],
) -> ComboHit | None:
    combo_score = (scores[left] + scores[right]) / 2.0
    candidates = [_combo_hit(frame, (left, right), combo_score, threshold) for threshold in SIGNAL_THRESHOLDS]
    valid = [row for row in candidates if row is not None]
    return max(valid, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def _combo_hit(
    frame: pd.DataFrame,
    members: tuple[str, str],
    score: pd.Series,
    threshold: float,
) -> ComboHit | None:
    signal = pd.Series(np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, np.nan)), index=frame.index)
    returns = (signal * frame["fwd_ret"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < BACKTEST_MIN_PERIODS:
        return None
    win_rate = float((returns > 0).mean())
    profit_factor = _profit_factor(returns)
    if win_rate < TARGET_WIN_RATE or profit_factor < SUCCESS_PROFIT_FACTOR_MIN:
        return None
    return ComboHit(members, threshold, win_rate, profit_factor, len(returns), float(returns.mean()), score)


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    return gains / losses if losses > 0 else math.inf


def _selected_hits(hits: list[ComboHit], target_count: int) -> list[ComboHit]:
    return hits[:target_count]


def _report_payload(
    symbol: str,
    duration: str,
    target_count: int,
    frame: pd.DataFrame,
    scores: dict[str, pd.Series],
    selected: list[ComboHit],
) -> dict[str, Any]:
    return {
        "version": "high_winrate_factor_combo_goal_v1",
        "updatedAt": utc_now(),
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "target": _target_payload(target_count),
        "onlineResearchSources": list(ONLINE_RESEARCH_SOURCES),
        "search": _search_payload(frame, scores),
        "ranking": [_ranking_row(index, row) for index, row in enumerate(selected, start=1)],
        "paperLiveSimulation": [_paper_signal(frame, index, row, duration) for index, row in enumerate(selected, start=1)],
    }


def _target_payload(target_count: int) -> dict[str, Any]:
    return {
        "targetCount": target_count,
        "minWinRate": TARGET_WIN_RATE,
        "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN,
        "minTrades": BACKTEST_MIN_PERIODS,
        "thresholds": list(SIGNAL_THRESHOLDS),
        "method": "oriented_expanding_zscore_pair_threshold_v1",
    }


def _search_payload(frame: pd.DataFrame, scores: dict[str, pd.Series]) -> dict[str, Any]:
    pair_count = len(scores) * (len(scores) - 1) // 2
    return {"entryRows": len(frame), "candidateFactors": len(scores), "testedPairs": pair_count}


def _ranking_row(rank: int, hit: ComboHit) -> dict[str, Any]:
    return {
        "rank": rank,
        "factorName": f"goal_combo__{hit.members[0]}__{hit.members[1]}",
        "members": list(hit.members),
        "threshold": hit.threshold,
        "winRate": round(hit.win_rate, 4),
        "profitFactor": round(hit.profit_factor, 4),
        "trades": hit.trades,
        "avgReturn": round(hit.avg_return, 8),
    }


def _paper_signal(frame: pd.DataFrame, rank: int, hit: ComboHit, duration: str) -> dict[str, Any]:
    index = live_duration_entry_index(frame, duration)
    score = float(hit.score.loc[index])
    direction = "up" if score >= hit.threshold else "down" if score <= -hit.threshold else "wait"
    return {
        **_ranking_row(rank, hit),
        "simulationMode": "paper_live",
        "simulationStrategyKey": f"high_winrate_factor_combo_goal_top{rank}",
        "sourceOpenTime": int(frame.at[index, "open_time"]),
        "entryPrice": float(frame.at[index, "close"]),
        "score": round(score, 6),
        "direction": direction,
        "qualityPassed": direction != "wait",
    }


def _library_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": report["version"],
        "updatedAt": report["updatedAt"],
        "symbol": report["symbol"],
        "duration": report["duration"],
        "target": report["target"],
        "factors": report["ranking"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find high-win-rate factor combos and paper-live signals.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration", default="30m")
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_goal(args.symbol, args.duration, args.target_count, args.output, args.library)
    print(json.dumps({"output": str(args.output), "library": str(args.library), "ranking": report["ranking"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
