from __future__ import annotations

from itertools import combinations, islice
from typing import Any

from app.services.factor_cache_metadata import assert_cache_usable_for_live_signal
from app.services.factor_combination_cache_service import get_cached_combination_ranking, save_cached_combination_ranking
from app.services.factor_combination_ranker import combination_result_for_member_rows
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_ranking_cache_service import get_cached_ranking


DEFAULT_BATCH_SIZE = 120
DEFAULT_LOOKBACK_DAYS = 365


def refresh_incremental_combination_cache(
    symbol: str,
    duration: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    safe_lookback_days = _positive_int(lookback_days, "lookback_days")
    frame = load_factor_frame(sym, duration, lookback_days=safe_lookback_days)
    source, missing_count = _source_rows(sym, duration, frame)
    existing = get_cached_combination_ranking(sym, duration) or _empty_report(sym, duration)
    evaluated = list(existing.get("evaluatedRanking") or existing.get("ranking") or [])
    seen = {str(row.get("factorName")) for row in evaluated if isinstance(row, dict)}
    rows, failures = _evaluate_batch(frame, sym, duration, source, seen, int(batch_size))
    evaluated.extend(rows)
    ranking = _passed_rows([*list(existing.get("ranking") or []), *rows])
    report = {
        **existing,
        "symbol": sym,
        "duration": duration,
        "ranking": ranking,
        "evaluatedRanking": evaluated,
        "total": len(ranking),
        "evaluatedTotal": len(evaluated),
        "testedCombinationCount": len(evaluated),
        "lookbackDays": safe_lookback_days,
        "failureCount": len(list(existing.get("failures") or [])) + len(failures),
        "failures": [*list(existing.get("failures") or []), *failures][-50:],
        "searchDiagnostics": _diagnostics(
            existing, source, evaluated, rows, failures, missing_count, safe_lookback_days
        ),
    }
    save_cached_combination_ranking(report)
    return report


def _source_rows(symbol: str, duration: str, frame: Any) -> tuple[list[dict[str, Any]], int]:
    cache = get_cached_ranking(symbol, duration)
    if cache is None:
        raise ValueError(f"factor ranking cache missing for {symbol} {duration}")
    assert_cache_usable_for_live_signal(cache, f"factor ranking {symbol} {duration}")
    rows = [dict(row) for row in cache.get("ranking") or [] if isinstance(row, dict)]
    if len(rows) < 2:
        raise ValueError(f"factor ranking cache has fewer than 2 rows for {symbol} {duration}")
    available = {str(column) for column in frame.columns}
    filtered = [row for row in rows if str(row.get("factorName") or "") in available]
    if len(filtered) < 2:
        raise ValueError(f"factor ranking cache has fewer than 2 computable rows for {symbol} {duration}")
    return filtered, len(rows) - len(filtered)


def _evaluate_batch(
    frame: Any,
    symbol: str,
    duration: str,
    source: list[dict[str, Any]],
    seen: set[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for left, right in islice(_pending_pairs(source, seen), max(0, batch_size)):
        combo_name = f"combo__{left['factorName']}__{right['factorName']}"
        try:
            rows.append(
                combination_result_for_member_rows(
                    frame,
                    symbol=symbol,
                    duration=duration,
                    combo_name=combo_name,
                    member_rows=[_member(left), _member(right)],
                )
            )
        except Exception as exc:
            failures.append({"factorName": combo_name, "stage": "incremental_combination", "error": str(exc)})
    return rows, failures


def _pending_pairs(source: list[dict[str, Any]], seen: set[str]):
    for left, right in combinations(source, 2):
        combo_name = f"combo__{left['factorName']}__{right['factorName']}"
        if combo_name not in seen:
            yield left, right


def _member(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(row["factorName"]),
        "displayName": str(row.get("factorDisplayName") or row["factorName"]),
        "category": str(row.get("category") or "unknown"),
        "orientation": 1,
        "singleWinRate": row.get("winRate"),
        "singleIr": row.get("ir"),
        "singleSharpe": row.get("sharpe"),
        "singleScore": row.get("factorScore"),
    }


def _passed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {str(row.get("factorName")): row for row in rows if isinstance(row, dict)}
    passed = [row for row in unique.values() if row.get("walkForwardPassed") is True]
    passed.sort(key=lambda row: float(row.get("factorScore") or 0.0), reverse=True)
    return passed


def _diagnostics(
    existing: dict[str, Any],
    source: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    missing_source_count: int,
    lookback_days: int,
) -> dict[str, Any]:
    previous = existing.get("searchDiagnostics") if isinstance(existing.get("searchDiagnostics"), dict) else {}
    return {
        **previous,
        "mode": "incremental_pairwise_display_refresh_v1",
        "lookbackDays": int(lookback_days),
        "baseCandidateCount": len(source),
        "missingSourceFactorCount": int(missing_source_count),
        "evaluatedCombinationCount": len(evaluated),
        "lastBatchEvaluatedCount": len(rows),
        "lastBatchFailureCount": len(failures),
        "walkForwardPassedCount": sum(1 for row in evaluated if row.get("walkForwardPassed") is True),
    }


def _positive_int(value: int, name: str) -> int:
    selected = int(value)
    if selected <= 0:
        raise ValueError(f"{name} must be positive")
    return selected


def _empty_report(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "ranking": [],
        "evaluatedRanking": [],
        "searchConfig": {"source": "incremental_pairwise_display_refresh_v1"},
        "baseFactors": [],
        "baseFactorCount": 0,
        "minedFactorSourceCount": 0,
        "minedFactorUsedCount": 0,
        "agentMinedFactorUsedCount": 0,
        "testedCombinationCount": 0,
        "failures": [],
    }
