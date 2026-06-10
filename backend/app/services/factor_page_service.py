from __future__ import annotations

from typing import Any

from app.services.factor_catalog import FactorCategory, list_single_factor_categories
from app.services.agent_mined_factor_library import agent_factor_rows_for_duration
from app.services.factor_catalog_summaries import (
    AGENT_FACTOR_SOURCE_FILE,
    list_combo_factor_summaries,
    list_single_factor_summaries,
)
from app.services.factor_metric_enrichment import backtest_validity
from app.services.factor_page_combo_rows import (
    combo_cache_total,
    combo_list_rows,
    sorted_combo_list_rows,
)
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.factor_page_alerts import ALERT_ERROR, ALERT_WARNING, build_alerts as _build_alerts
from app.services.factor_page_high_winrate import high_winrate_card as _high_winrate_card
from app.services.factor_page_list import (
    SOURCE_AGENT,
    SOURCE_COMBO,
    SOURCE_LOCAL,
    SOURCE_LSTM,
    classify_factor_source,
    filter_factor_rows,
    paginate_rows,
    source_label,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

ALERT_WARNING = "warning"
ALERT_ERROR = "error"

def build_factor_list_page(
    *,
    category: str | None,
    kind: str,
    symbol: str | None = None,
    duration: str | None = None,
    query: str | None,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    singles = [enrich_factor_summary(row) for row in list_single_factor_summaries(category)]
    combos = [enrich_factor_summary(row) for row in combo_list_rows(symbol, duration, list_combo_factor_summaries())]
    overview = build_factor_overview(category)
    if kind == "combo" and symbol and duration:
        overview = {**overview, "comboTotal": len(combos)}
    if kind == "combo":
        base_rows = sorted_combo_list_rows(combos)
        rows = filter_factor_rows(base_rows, query)
        paginated = paginate_rows(rows, page, page_size)
        return {
            **overview,
            "kind": "combo",
            "factors": paginated["items"],
            "comboFactors": combos,
            "categories": list_single_factor_categories(),
            "total": paginated["total"],
            "comboTotal": overview["comboTotal"],
            "listTotal": paginated["total"],
            "unfilteredTotal": len(base_rows),
            "page": paginated["page"],
            "pageSize": paginated["pageSize"],
            "pageCount": paginated["pageCount"],
            "query": (query or "").strip(),
            "sourceSummary": overview["sourceSummary"],
        }
    base_rows = list(singles)
    rows = filter_factor_rows(base_rows, query)
    paginated = paginate_rows(rows, page, page_size)
    return {
        **overview,
        "kind": "single",
        "factors": paginated["items"],
        "comboFactors": combos,
        "categories": list_single_factor_categories(),
        "total": paginated["total"],
        "comboTotal": overview["comboTotal"],
        "listTotal": paginated["total"],
        "unfilteredTotal": len(base_rows),
        "page": paginated["page"],
        "pageSize": paginated["pageSize"],
        "pageCount": paginated["pageCount"],
        "query": (query or "").strip(),
        "sourceSummary": overview["sourceSummary"],
    }

def build_factor_page_bundle(
    symbol: str,
    duration: str,
    *,
    category: str | None = None,
    kind: str = "single",
    query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    list_payload = build_factor_list_page(
        category=category,
        kind=kind,
        symbol=sym,
        duration=duration,
        query=query,
        page=page,
        page_size=page_size,
    )
    context = build_factor_page_context(sym, duration, category=category)
    cached = get_cached_ranking(sym, duration)
    ranking_rows: list[dict[str, Any]] = []
    if cached is not None:
        ranking_rows = list(cached.get("ranking") or [])
        if category:
            ranking_rows = [row for row in ranking_rows if row.get("category") == category]
        ranking_rows.sort(
            key=lambda row: (float(row.get("factorScore") or 0.0), abs(float(row.get("ir") or 0.0))),
            reverse=True,
        )
    return {
        **list_payload,
        **context,
        "rankingPageTotal": len(ranking_rows),
    }

def build_factor_overview(category: str | None = None) -> dict[str, Any]:
    singles = list_single_factor_summaries(category)
    combos = list_combo_factor_summaries()
    source_summary = _source_summary(singles, combos)
    return {
        "singleTotal": len(singles),
        "comboTotal": len(combos),
        "sourceSummary": source_summary,
    }

def build_factor_page_context(
    symbol: str,
    duration: str,
    *,
    category: str | None = None,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    overview = build_factor_overview(category)
    combo_total = combo_cache_total(sym, duration)
    overview = {
        **overview,
        "comboTotal": combo_total if combo_total is not None else overview["comboTotal"],
    }
    global_summary = overview["sourceSummary"]
    display_summary = _display_source_summary(sym, duration, category, global_summary)
    ranking_payload = _ranking_payload(sym, duration, category)
    combo_card = _high_winrate_card(sym, duration)
    alerts = build_factor_alerts(sym, duration, ranking_payload=ranking_payload)
    return {
        **overview,
        "symbol": sym,
        "duration": duration,
        "category": category,
        "sourceSummary": display_summary,
        "sourceSummaryGlobal": global_summary,
        "rankingStatus": ranking_payload.get("status"),
        "rankingUpdatedAt": ranking_payload.get("updatedAt"),
        "rankingTotal": ranking_payload.get("total"),
        "rankingSource": ranking_payload.get("source"),
        "highWinrateCombo": combo_card,
        "alerts": alerts,
    }

def build_factor_period_scores(symbol: str, factor_name: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    scores: list[dict[str, Any]] = []
    for duration in SUPPORTED_RULE_DURATIONS:
        row = _ranking_row_for_factor(sym, duration, factor_name)
        scores.append(
            {
                "duration": duration,
                "factorScore": row.get("factorScore") if row else None,
                "available": row is not None,
                "totalPeriods": row.get("totalPeriods") if row else None,
            }
        )
    return {"symbol": sym, "factorName": factor_name, "scores": scores}

def build_factor_alerts(
    symbol: str,
    duration: str,
    *,
    ranking_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sym = symbol.strip().upper()
    ranking_payload = ranking_payload or _ranking_payload(sym, duration, None)
    return _build_alerts(sym, duration, ranking_payload)

def enrich_factor_summary(row: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(row)
    payload["sourceKind"] = classify_factor_source(row.get("sourceFile"), row.get("name"))
    payload["sourceLabel"] = source_label(payload["sourceKind"])
    merged = {**payload, **(metrics or {})}
    merged["canStore"] = _can_store(merged)
    return merged

def _source_summary(singles: list[dict[str, Any]], combos: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        SOURCE_LOCAL: 0,
        SOURCE_AGENT: 0,
        SOURCE_LSTM: 0,
        SOURCE_COMBO: len(combos),
    }
    for row in singles:
        kind = classify_factor_source(row.get("sourceFile"), row.get("name"))
        if kind == SOURCE_COMBO:
            continue
        summary[kind] = summary.get(kind, 0) + 1
    return summary

def _includes_agent_category(category: str | None) -> bool:
    if not category:
        return True
    return category == FactorCategory.STATISTIC.value

def _display_source_summary(
    symbol: str,
    duration: str,
    category: str | None,
    global_summary: dict[str, int],
) -> dict[str, int]:
    """顶栏来源状态：Agent 数按当前交易对+周期统计，其余保持全库口径。"""
    summary = dict(global_summary)
    if _includes_agent_category(category):
        summary[SOURCE_AGENT] = len(agent_factor_rows_for_duration(symbol, duration))
    else:
        summary[SOURCE_AGENT] = 0
    return summary

def _ranking_payload(symbol: str, duration: str, category: str | None) -> dict[str, Any]:
    cached = get_cached_ranking(symbol, duration)
    if cached is None:
        return {
            "source": "none",
            "updatedAt": None,
            "total": 0,
            "status": f"暂无排名缓存（{symbol} / {duration}）",
            "rankingFailures": [],
        }
    ranking = list(cached.get("ranking") or [])
    if category:
        ranking = [row for row in ranking if row.get("category") == category]
    ranking.sort(
        key=lambda row: (float(row.get("factorScore") or 0.0), abs(float(row.get("ir") or 0.0))),
        reverse=True,
    )
    updated = cached.get("updatedAt")
    return {
        "source": "cache",
        "updatedAt": updated,
        "total": len(ranking),
        "status": f"排名缓存已刷新 · {symbol} / {duration}" + (f" · 更新 {updated}" if updated else ""),
        "rankingFailures": list(cached.get("rankingFailures") or []),
        "rankingDiagnostics": dict(cached.get("rankingDiagnostics") or {}),
    }

def ranking_metrics_for_factor(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    return _ranking_row_for_factor(symbol, duration, factor_name)

def _ranking_row_for_factor(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    cached = get_cached_ranking(symbol, duration)
    if cached is None:
        return None
    for row in cached.get("ranking") or []:
        if row.get("factorName") == factor_name or row.get("name") == factor_name:
            return dict(row)
    return None

def _can_store(row: dict[str, Any]) -> bool:
    if row.get("qualityPassed") is True:
        return True
    validity = backtest_validity(row)
    return bool(validity.get("valid"))
