from __future__ import annotations

from typing import Any

from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_mined_library import mined_factor_rows_for_duration
from app.services.factor_metric_enrichment import factor_score
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


def combo_metrics_for_factor(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    row = cached_combo_row_for_factor(symbol, duration, factor_name)
    if row is not None:
        return row
    return library_combo_metrics_for_factor(symbol, duration, factor_name)


def combo_period_scores(symbol: str, factor_name: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    scores = [_combo_score_row(sym, duration, factor_name) for duration in SUPPORTED_RULE_DURATIONS]
    return {"symbol": sym, "factorName": factor_name, "scores": scores}


def cached_combo_row_for_factor(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    cached = get_cached_combination_ranking(symbol.strip().upper(), duration)
    if not cache_is_usable(cached):
        return None
    for row in cached.get("ranking") or []:
        if row.get("factorName") == factor_name or row.get("name") == factor_name:
            return _normalize_combo_metrics(row, duration)
    return None


def library_combo_metrics_for_factor(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    for row in mined_factor_rows_for_duration(symbol, duration):
        if row.get("factorName") == factor_name:
            return _normalize_library_metrics(row, duration)
    return None


def _combo_score_row(symbol: str, duration: str, factor_name: str) -> dict[str, Any]:
    row = combo_metrics_for_factor(symbol, duration, factor_name)
    return {
        "duration": duration,
        "factorScore": row.get("factorScore") if row else None,
        "available": row is not None,
        "totalPeriods": row.get("totalPeriods") if row else None,
    }


def _normalize_combo_metrics(row: dict[str, Any], duration: str) -> dict[str, Any]:
    factor_name = str(row.get("factorName") or row.get("name") or "")
    display = str(row.get("factorDisplayName") or row.get("displayName") or row.get("description") or factor_name)
    return {
        **row,
        "name": factor_name,
        "factorName": factor_name,
        "displayName": display,
        "factorDisplayName": display,
        "description": display,
        "duration": str(row.get("duration") or duration),
    }


def _normalize_library_metrics(row: dict[str, Any], duration: str) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    merged = {**row, **metrics}
    merged.setdefault("factorName", row.get("name"))
    merged.setdefault("factorScore", row.get("score") or factor_score(merged))
    merged.setdefault("duration", duration)
    return _normalize_combo_metrics(merged, duration)
