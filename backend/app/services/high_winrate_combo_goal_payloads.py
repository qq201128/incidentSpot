from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.factor_duration_alignment import live_duration_entry_index
from app.services.factor_learning_common import utc_now
from app.services import high_winrate_combo_goal_search as search

ONLINE_RESEARCH_SOURCES = (
    {"title": "Explainable Patterns in Cryptocurrency Microstructure", "url": "https://arxiv.org/pdf/2602.00776", "factorFamilies": ["order flow imbalance", "spread", "VWAP-to-mid pressure"]},
    {"title": "Order flow and cryptocurrency returns", "url": "https://www.sciencedirect.com/science/article/pii/S1386418126000029", "factorFamilies": ["signed order flow", "buyer/seller initiated volume"]},
    {"title": "The Crypto Signal Compendium", "url": "https://the-algotrading-book-website.vercel.app/chapters/01-foundations/024-crypto-signal-compendium/", "factorFamilies": ["open interest", "long/short ratio", "taker buy/sell", "sentiment"]},
)


def report_payload(
    symbol: str,
    duration: str,
    target_count: int,
    frame: pd.DataFrame,
    score_search: search.ScoreSearch,
    ranked_search: search.RankedSearch,
    selected: list[search.ComboHit],
    validation_gate: dict[str, Any] | None = None,
    search_config: search.GoalSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = search.validated_search_config(search_config)
    ranking = [ranking_row(index, row, cfg) for index, row in enumerate(selected, start=1)]
    return {
        "version": "high_winrate_factor_combo_goal_v1",
        "updatedAt": utc_now(),
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "target": target_payload(target_count, cfg),
        "onlineResearchSources": list(ONLINE_RESEARCH_SOURCES),
        "search": search_payload(frame, score_search.scores, ranked_search.diagnostics),
        "candidateDiagnostics": score_search.diagnostics,
        "rankingDiagnostics": ranked_search.diagnostics,
        "validationGate": validation_gate,
        "rankingFailure": ranking_failure_payload(ranking, score_search, ranked_search, validation_gate),
        "ranking": ranking,
        "paperLiveSimulation": [paper_signal(frame, index, row, duration, cfg) for index, row in enumerate(selected, start=1)],
    }


def target_payload(
    target_count: int,
    search_config: search.GoalSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = search.validated_search_config(search_config)
    return {
        "targetCount": target_count,
        "minWinRate": search.TARGET_WIN_RATE,
        "minProfitFactor": search.TARGET_PROFIT_FACTOR,
        "minTrades": cfg.min_trades,
        "thresholds": list(cfg.signal_thresholds),
        "searchCandidateLimit": cfg.candidate_limit,
        "method": "oriented_expanding_zscore_pair_threshold_v1",
    }


def search_payload(
    frame: pd.DataFrame,
    scores: dict[str, search.OrientedScore],
    ranking_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    pair_count = len(scores) * (len(scores) - 1) // 2
    return {
        "entryRows": len(frame),
        "candidateFactors": len(scores),
        "testedPairs": pair_count,
        "candidatePairCount": pair_count,
        "selectedCandidateFactors": ranking_diagnostics["selectedCandidateFactors"],
        "testedCombinations": ranking_diagnostics["testedCombinations"],
        "testedThresholdEvaluations": ranking_diagnostics["testedThresholdEvaluations"],
    }


def ranking_failure_payload(
    ranking: list[dict[str, Any]],
    score_search: search.ScoreSearch,
    ranked_search: search.RankedSearch,
    validation_gate: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if ranking:
        return None
    if _validation_gate_failed(validation_gate):
        return {
            "stage": "validation_gate",
            "reason": validation_gate["failureReason"],
            "details": validation_gate,
        }
    candidate_reason = score_search.diagnostics.get("reason")
    if candidate_reason is not None:
        return {
            "stage": "candidate_factor_filter",
            "reason": candidate_reason,
            "details": score_search.diagnostics,
        }
    return {
        "stage": "combo_threshold_gates",
        "reason": ranked_search.diagnostics["failureReason"],
        "details": ranked_search.diagnostics,
    }


def _validation_gate_failed(validation_gate: dict[str, Any] | None) -> bool:
    return bool(
        validation_gate
        and validation_gate.get("status") == "failed"
        and validation_gate.get("failureReason")
    )


def ranking_row(
    rank: int,
    hit: search.ComboHit,
    search_config: search.GoalSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = search.validated_search_config(search_config)
    member_names = "__".join(hit.members)
    name = f"goal_combo__{member_names}"
    members = [member_payload(member, orientation) for member, orientation in zip(hit.members, hit.orientations)]
    display_name = combo_display_name(members)
    reported_win_rate = round(hit.win_rate, 4)
    return {
        "rank": rank,
        "factorName": name,
        "factorDisplayName": display_name,
        "description": display_name,
        "formula": f"oriented_zscore_pair_threshold_v1({', '.join(hit.members)})",
        "method": "oriented_expanding_zscore_pair_threshold_v1",
        "members": members,
        "comboSize": len(hit.members),
        "threshold": hit.threshold,
        "winRate": reported_win_rate,
        "backtestWinRate": reported_win_rate,
        "oosWinRate": None,
        "walkForwardResult": None,
        "recentRollingResult": None,
        "paperLiveWinRate": None,
        "paperLiveStatus": "backtest_candidate",
        "profitFactor": round(hit.profit_factor, 4),
        "trades": hit.trades,
        "totalPeriods": hit.trades,
        "minTrades": cfg.min_trades,
        "avgReturn": round(hit.avg_return, 8),
    }


def member_payload(name: str, orientation: int) -> dict[str, Any]:
    return {"name": name, "displayName": name, "category": "unknown", "orientation": orientation}


def combo_display_name(members: list[dict[str, Any]]) -> str:
    from app.services.factor_combo_display import combo_display_name as build_combo_display_name

    return build_combo_display_name(members)


def paper_signal(
    frame: pd.DataFrame,
    rank: int,
    hit: search.ComboHit,
    duration: str,
    search_config: search.GoalSearchConfig | None = None,
) -> dict[str, Any]:
    index = live_duration_entry_index(frame, duration)
    score = float(hit.score.loc[index])
    direction = "up" if score >= hit.threshold else "down" if score <= -hit.threshold else "wait"
    return {
        **ranking_row(rank, hit, search_config),
        "simulationMode": "paper_live",
        "simulationStrategyKey": f"high_winrate_factor_combo_goal_top{rank}",
        "sourceOpenTime": int(frame.at[index, "open_time"]),
        "entryPrice": float(frame.at[index, "close"]),
        "score": round(score, 6),
        "direction": direction,
        "qualityPassed": direction != "wait",
    }


def library_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": report["version"],
        "updatedAt": report["updatedAt"],
        "symbol": report["symbol"],
        "duration": report["duration"],
        "target": report["target"],
        "factors": report["ranking"],
    }
