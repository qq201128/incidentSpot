from __future__ import annotations

from typing import Any

from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report
from app.services.factor_catalog import FactorCategory, list_single_factor_categories
from app.services.agent_mined_factor_library import agent_factor_rows_for_duration
from app.services.factor_catalog_summaries import (
    AGENT_FACTOR_SOURCE_FILE,
    list_combo_factor_summaries,
    list_single_factor_summaries,
)
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_metric_enrichment import backtest_validity
from app.services.factor_mined_library import MINED_FACTOR_SOURCE_FILE
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.high_winrate_combo_view import build_high_winrate_combo_view
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

SOURCE_LOCAL = "local_definition"
SOURCE_AGENT = "agent_candidate"
SOURCE_LSTM = "lstm_shadow"
SOURCE_COMBO = "composite_cache"

ALERT_WARNING = "warning"
ALERT_ERROR = "error"


def filter_factor_rows(rows: list[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return list(rows)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        haystack = " ".join(
            str(row.get(key) or "")
            for key in ("name", "displayName", "description", "formula", "categoryName", "sourceLabel")
        ).lower()
        if q in haystack or q in source_label(classify_factor_source(row.get("sourceFile"), row.get("name"))).lower():
            filtered.append(row)
    return filtered


def paginate_rows(rows: list[dict[str, Any]], page: int, page_size: int) -> dict[str, Any]:
    total = len(rows)
    size = max(1, int(page_size))
    current = max(1, int(page))
    page_count = max(1, (total + size - 1) // size) if total else 1
    if current > page_count:
        current = page_count
    start = (current - 1) * size
    end = start + size
    return {
        "items": rows[start:end],
        "total": total,
        "page": current,
        "pageSize": size,
        "pageCount": page_count,
    }


def build_factor_list_page(
    *,
    category: str | None,
    kind: str,
    query: str | None,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    singles = [enrich_factor_summary(row) for row in list_single_factor_summaries(category)]
    combos = [enrich_factor_summary(row) for row in list_combo_factor_summaries()]
    overview = build_factor_overview(category)
    if kind == "combo":
        rows = filter_factor_rows(combos, query)
        paginated = paginate_rows(rows, page, page_size)
        return {
            **overview,
            "kind": "combo",
            "factors": paginated["items"],
            "comboFactors": combos,
            "categories": list_single_factor_categories(),
            "total": overview["singleTotal"],
            "comboTotal": overview["comboTotal"],
            "listTotal": paginated["total"],
            "page": paginated["page"],
            "pageSize": paginated["pageSize"],
            "pageCount": paginated["pageCount"],
            "sourceSummary": overview["sourceSummary"],
        }
    rows = filter_factor_rows(singles, query)
    paginated = paginate_rows(rows, page, page_size)
    return {
        **overview,
        "kind": "single",
        "factors": paginated["items"],
        "comboFactors": combos,
        "categories": list_single_factor_categories(),
        "total": overview["singleTotal"],
        "comboTotal": overview["comboTotal"],
        "listTotal": paginated["total"],
        "page": paginated["page"],
        "pageSize": paginated["pageSize"],
        "pageCount": paginated["pageCount"],
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
        "ranking": ranking_rows,
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
    alerts: list[dict[str, Any]] = []
    alerts.extend(_coverage_alerts(sym, duration))
    alerts.extend(_ranking_failure_alerts(ranking_payload))
    alerts.extend(_combination_failure_alerts(sym, duration))
    return alerts


def enrich_factor_summary(row: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(row)
    payload["sourceKind"] = classify_factor_source(row.get("sourceFile"), row.get("name"))
    payload["sourceLabel"] = source_label(payload["sourceKind"])
    merged = {**payload, **(metrics or {})}
    payload["canStore"] = _can_store(merged)
    return payload


def classify_factor_source(source_file: object, name: object) -> str:
    sf = str(source_file or "")
    nm = str(name or "").lower()
    if sf == AGENT_FACTOR_SOURCE_FILE or nm.startswith("agent__"):
        return SOURCE_AGENT
    if sf == MINED_FACTOR_SOURCE_FILE:
        return SOURCE_COMBO
    if "lstm" in nm or "lstm" in sf.lower():
        return SOURCE_LSTM
    return SOURCE_LOCAL


def source_label(kind: str) -> str:
    labels = {
        SOURCE_LOCAL: "本地定义",
        SOURCE_AGENT: "Agent候选",
        SOURCE_LSTM: "LSTM影子",
        SOURCE_COMBO: "组合缓存",
    }
    return labels.get(kind, kind)


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


def _high_winrate_card(symbol: str, duration: str) -> dict[str, Any]:
    cached = get_cached_high_winrate_combo_ranking(symbol, duration)
    view = build_high_winrate_combo_view(cached, duration)
    ranking = list(view.get("highWinrateRanking") or [])
    top = dict(ranking[0]) if ranking else None
    if top is None:
        return {
            "available": False,
            "updatedAt": view.get("highWinrateUpdatedAt"),
            "total": 0,
        }
    members = top.get("members") if isinstance(top.get("members"), list) else []
    return {
        "available": True,
        "updatedAt": view.get("highWinrateUpdatedAt"),
        "total": int(view.get("highWinrateTotal") or len(ranking)),
        "factorName": top.get("factorName"),
        "displayName": top.get("factorDisplayName") or top.get("displayName") or top.get("factorName"),
        "members": [
            {
                "name": member.get("name") if isinstance(member, dict) else member,
                "displayName": member.get("displayName") if isinstance(member, dict) else member,
            }
            for member in members
        ],
        "winRate": top.get("winRate"),
        "avgTradesPerDay": top.get("avgTradesPerDay"),
        "factorScore": top.get("factorScore"),
        "profitFactor": top.get("profitFactor"),
        "maxDrawdown": top.get("maxDrawdown"),
        "totalPeriods": top.get("totalPeriods") or top.get("trades"),
        "sampleDays": top.get("sampleDays"),
    }


def _coverage_alerts(symbol: str, duration: str) -> list[dict[str, Any]]:
    report = build_data_coverage_report(CoverageOptions(symbol=symbol, interval=duration))
    gaps: list[dict[str, Any]] = []
    for table in report.get("tables") or []:
        table_name = str(table.get("table") or "")
        for row_index, row in enumerate(table.get("rows") or []):
            status = row.get("status")
            if status in {"healthy", None}:
                continue
            reason = str(row.get("missingReason") or status)
            gaps.append(
                {
                    "table": table_name,
                    "status": status,
                    "missingReason": reason,
                    "group": str(row.get("group") or ""),
                    "coveragePct": row.get("coveragePct"),
                    "rowIndex": row_index,
                }
            )
    if not gaps:
        return []
    sample = gaps[0]
    sample_reason = sample.get("missingReason") or "unknown"
    sample_table = sample.get("table") or "—"
    extra = f"等共 {len(gaps)} 项" if len(gaps) > 1 else ""
    return [
        {
            "id": "data_coverage_gap:summary",
            "level": ALERT_WARNING,
            "code": "data_coverage_gap",
            "title": "回测依赖缺失会阻断刷新",
            "message": (
                f"缺少成交额等依赖数据（{sample_table} · {sample_reason}{extra}）。"
                "缺失数据或回测失败时将显示具体原因，并可能阻塞排名刷新。"
            ),
            "detail": {"symbol": symbol, "duration": duration, "total": len(gaps), "items": gaps[:20]},
        }
    ]


def _ranking_failure_alerts(ranking_payload: dict[str, Any]) -> list[dict[str, Any]]:
    failures = ranking_payload.get("rankingFailures") or []
    if not failures:
        return []
    samples = failures[:5]
    reasons = sorted({str(item.get("error") or "unknown") for item in failures})
    detail = "; ".join(reasons[:3])
    return [
        {
            "id": "factor_ranking_failures",
            "level": ALERT_WARNING,
            "code": "factor_ranking_failures",
            "title": "部分单因子排名计算失败",
            "message": (
                f"本轮排名有 {len(failures)} 个因子未计入（示例：{detail}）。"
                "请检查特征列或数据覆盖后重新刷新排名。"
            ),
            "detail": {"failureCount": len(failures), "samples": samples},
        }
    ]


def _combination_failure_alerts(symbol: str, duration: str) -> list[dict[str, Any]]:
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        return []
    search = cached.get("search") if isinstance(cached.get("search"), dict) else {}
    diagnostics = cached.get("searchDiagnostics") if isinstance(cached.get("searchDiagnostics"), dict) else {}
    failure_counts = diagnostics.get("failureReasonCounts") if isinstance(diagnostics, dict) else {}
    failures = cached.get("failures") if isinstance(cached.get("failures"), list) else []
    if not failure_counts and not search and not failures:
        return []
    entry_rows = search.get("entryRows") or diagnostics.get("entryRows")
    min_sample = diagnostics.get("targetCriteria", {}).get("validationMinSampleCount")
    message_parts = []
    if entry_rows is not None and min_sample is not None and int(entry_rows) < int(min_sample):
        message_parts.append(
            f"组合回测样本不足（{entry_rows} < {min_sample}），最近一轮高胜率组合搜索未产出可用结果。"
        )
    if failure_counts:
        top_reason = next(iter(failure_counts))
        message_parts.append(f"失败原因统计：{top_reason} × {failure_counts[top_reason]}。")
    if failures and not message_parts:
        message_parts.append(f"组合回测失败 {len(failures)} 条，请查看详情。")
    if not message_parts:
        return []
    return [
        {
            "id": "combo_backtest_failed",
            "level": ALERT_ERROR,
            "code": "combo_backtest_failed",
            "title": "组合因子回测未通过",
            "message": " ".join(message_parts),
            "detail": {
                "entryRows": entry_rows,
                "failureReasonCounts": failure_counts,
                "search": search,
            },
        }
    ]


def _can_store(row: dict[str, Any]) -> bool:
    if row.get("qualityPassed") is True:
        return True
    validity = backtest_validity(row)
    return bool(validity.get("valid"))
