from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import traceback
from typing import Any, Callable

from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.services.factor_combo_batch_predictions import offline_candidate_screening_report
from app.services.factor_combination_background import refresh_symbol_combination_rankings
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.forward_validation_service import settle_due_predictions
from app.services.model_family_daily_candidates import model_family_daily_candidate_report
from app.services.paper_live_candidate_service import (
    OBSERVATION_POOL_LIMIT,
    paper_live_candidate_report,
    refresh_paper_live_candidate_states,
)
from app.services.paper_live_report_cache import store_paper_live_report_cache
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperLiveDailyLoopDeps:
    symbols: Callable[[], list[str]] = factor_ranking_precomputed_symbols
    refresh_candidates: Callable[..., None] = refresh_symbol_combination_rankings
    settle_predictions: Callable[[str, str], dict[str, int]] = settle_due_predictions
    refresh_states: Callable[[str, str], dict[str, Any]] = refresh_paper_live_candidate_states
    candidate_report: Callable[[str, str], dict[str, Any]] = paper_live_candidate_report
    offline_screening: Callable[[str, str], dict[str, Any]] = offline_candidate_screening_report
    model_candidates: Callable[[str, str], dict[str, Any]] = model_family_daily_candidate_report


@dataclass(frozen=True)
class CandidateReportContext:
    stages: list[dict[str, Any]]
    symbol: str
    duration: str
    deps: PaperLiveDailyLoopDeps


def run_paper_live_daily_closed_loop(
    *,
    symbols: list[str] | None = None,
    durations: list[str] | None = None,
    deps: PaperLiveDailyLoopDeps | None = None,
) -> dict[str, Any]:
    active_deps = deps or PaperLiveDailyLoopDeps()
    selected_symbols = _symbols(symbols, active_deps)
    selected_durations = _durations(durations)
    results = [
        _run_symbol_duration(symbol, duration, active_deps)
        for symbol in selected_symbols
        for duration in selected_durations
    ]
    return _summary_payload(results, selected_symbols, selected_durations)


def _run_symbol_duration(symbol: str, duration: str, deps: PaperLiveDailyLoopDeps) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    sym = symbol.strip().upper()
    _append_stage(stages, "market_and_offline_candidates", lambda: _refresh_candidates(sym, duration, deps))
    _append_stage(stages, "settle_due_predictions", lambda: deps.settle_predictions(sym, duration))
    _append_stage(stages, "paper_live_lifecycle", lambda: deps.refresh_states(sym, duration))
    report = _candidate_report_stage(CandidateReportContext(stages, sym, duration, deps))
    return {
        "symbol": sym,
        "duration": duration,
        "status": _result_status(stages),
        "stages": stages,
        "dailyChecklist": _daily_checklist(stages, report),
        "candidatePool": report,
        "realTimePredictionExecutor": _prediction_executor_payload(),
    }


def _refresh_candidates(symbol: str, duration: str, deps: PaperLiveDailyLoopDeps) -> dict[str, Any]:
    deps.refresh_candidates(symbol, duration, run_learning_agent=True)
    screening = deps.offline_screening(symbol, duration)
    models = deps.model_candidates(symbol, duration)
    return {
        "offlineScreeningRole": "candidate_prefilter_only",
        "paperLiveQueueRole": "qualified_candidates_enabled_for_observation",
        "offlineScreening": _offline_screening_summary(screening),
        "modelCandidates": _model_candidate_summary(models),
        "realTradingEnabled": False,
    }


def _candidate_report_stage(context: CandidateReportContext) -> dict[str, Any]:
    box: dict[str, Any] = {}

    def load_report() -> dict[str, Any]:
        report = context.deps.candidate_report(context.symbol, context.duration)
        box["report"] = report
        store_paper_live_report_cache(context.symbol, context.duration, report)
        return _candidate_summary(report)

    _append_stage(context.stages, "candidate_pool_report", load_report)
    return box.get("report") or {}


def _append_stage(stages: list[dict[str, Any]], name: str, action: Callable[[], dict[str, Any]]) -> None:
    try:
        payload = action()
        stages.append({"stage": name, "status": "passed", "payload": payload})
    except Exception as exc:
        logger.exception("paper-live daily stage failed: %s", name)
        stages.append(_failed_stage(name, exc))


def _candidate_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "observationPoolLimit": report.get("observationPoolLimit"),
        "stableCount": len(report.get("stable") or []),
        "collectingCount": len(report.get("collecting") or []),
        "failedCount": len(report.get("failed") or []),
        "predictionFailureCount": len(report.get("predictionFailures") or []),
        "avoidNextSearch": report.get("avoidNextSearch") or [],
    }


def _offline_screening_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "rankingPolicy": report.get("rankingPolicy") or [],
        "observationPoolLimit": report.get("observationPoolLimit"),
        "focusedCount": report.get("focusedCount"),
        "candidateCount": report.get("candidateCount"),
        "rejectedCount": report.get("rejectedCount"),
        "rejectedReasons": report.get("rejectedReasons") or [],
        "reasonCounts": report.get("reasonCounts") or {},
    }


def _model_candidate_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "familyCount": report.get("familyCount"),
        "paperLiveReadyCount": report.get("paperLiveReadyCount"),
        "failures": report.get("failures") or [],
        "models": report.get("models") or [],
        "realTradingEnabled": False,
    }


def _prediction_executor_payload() -> dict[str, Any]:
    return {
        "stage": "generate_real_time_predictions",
        "status": "delegated_to_auto_predict_loop",
        "reason": "predictions must be created at real entry windows, not by looking ahead during daily review",
        "writesTo": ["predictions", "simulation_events"],
        "realTradingEnabled": False,
    }


def _summary_payload(results: list[dict[str, Any]], symbols: list[str], durations: list[str]) -> dict[str, Any]:
    return {
        "version": "paper_live_daily_closed_loop_v1",
        "runAt": _utc_now(),
        "status": "passed" if all(row["status"] == "passed" for row in results) else "failed",
        "symbols": symbols,
        "durations": durations,
        "realTradingEnabled": False,
        "observationPoolLimit": OBSERVATION_POOL_LIMIT,
        "dailyTaskCount": 11,
        "results": results,
    }


def _daily_checklist(stages: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    stage_status = {stage["stage"]: stage["status"] for stage in stages}
    candidate_report_status = stage_status.get("candidate_pool_report")
    return [
        _check("refresh_market_data", stage_status.get("market_and_offline_candidates")),
        _check("update_factor_and_model_candidates", stage_status.get("market_and_offline_candidates")),
        _check("offline_oos_walk_forward_rolling_prefilter", stage_status.get("market_and_offline_candidates")),
        _check("enqueue_qualified_candidates_for_paper_live", stage_status.get("market_and_offline_candidates")),
        _check("generate_real_time_predictions", "delegated", _prediction_executor_payload()["reason"]),
        _check("settle_due_predictions", stage_status.get("settle_due_predictions")),
        _check("update_paper_live_metrics", stage_status.get("paper_live_lifecycle")),
        _check("update_candidate_lifecycle_status", stage_status.get("paper_live_lifecycle")),
        _reported_candidate_check("eliminate_failed_candidates", candidate_report_status, report, key="failed"),
        _reported_candidate_check("retain_stable_candidates", candidate_report_status, report, key="stable"),
        _check("output_failure_reasons_and_next_search_direction", candidate_report_status),
    ]


def _reported_candidate_check(
    name: str,
    candidate_report_status: str | None,
    report: dict[str, Any],
    *,
    key: str,
) -> dict[str, Any]:
    if candidate_report_status == "passed":
        return _check(name, "reported", f"{key}={len(report.get(key) or [])}")
    reason = "candidate_pool_report_failed" if candidate_report_status == "failed" else "candidate_pool_report_not_run"
    return _check(name, candidate_report_status, reason)


def _check(name: str, status: str | None, reason: str | None = None) -> dict[str, Any]:
    return {
        "task": name,
        "status": status or "not_run",
        "reason": reason,
        "realTradingEnabled": False,
    }


def _failed_stage(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "stage": name,
        "status": "failed",
        "reason": str(exc),
        "exceptionType": type(exc).__name__,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _result_status(stages: list[dict[str, Any]]) -> str:
    return "failed" if any(stage["status"] == "failed" for stage in stages) else "passed"


def _symbols(symbols: list[str] | None, deps: PaperLiveDailyLoopDeps) -> list[str]:
    selected = symbols if symbols is not None else deps.symbols()
    return list(dict.fromkeys(symbol.strip().upper() for symbol in selected if symbol.strip()))


def _durations(durations: list[str] | None) -> list[str]:
    raw = durations if durations is not None else list(BACKTEST_DURATION_ORDER)
    selected = [duration for duration in raw if duration in SUPPORTED_RULE_DURATIONS]
    if not selected:
        raise ValueError("no supported paper-live durations selected")
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
