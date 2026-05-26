from __future__ import annotations

from typing import Any

from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report
from app.services.factor_combination_cache_service import get_cached_combination_ranking

ALERT_WARNING = "warning"
ALERT_ERROR = "error"
# Sparse or auxiliary tables: partial coverage is normal; only "missing" blocks research.
SPARSE_COVERAGE_TABLES = frozenset({
    "funding_features",
    "futures_positioning_features",
    "market_sentiment_features",
    "onchain_features",
    "klines_multi",
})
MIN_ORDERBOOK_COVERAGE_PCT = 80.0


def build_alerts(symbol: str, duration: str, ranking_payload: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    alerts.extend(coverage_alerts(symbol, duration))
    alerts.extend(ranking_failure_alerts(ranking_payload))
    alerts.extend(combination_failure_alerts(symbol, duration))
    return alerts


def coverage_alerts(symbol: str, duration: str) -> list[dict[str, Any]]:
    report = build_data_coverage_report(CoverageOptions(symbol=symbol, interval=duration))
    gaps = coverage_gaps(report, primary_interval=duration)
    if not gaps:
        return []
    sample = gaps[0]
    sample_reason = sample.get("missingReason") or "unknown"
    sample_table = sample.get("table") or "—"
    extra = f"等共 {len(gaps)} 项" if len(gaps) > 1 else ""
    return [{
        "id": "data_coverage_gap:summary",
        "level": ALERT_WARNING,
        "code": "data_coverage_gap",
        "title": "回测依赖缺失会阻断刷新",
        "message": (
            f"缺少成交额等依赖数据（{sample_table} · {sample_reason}{extra}）。"
            "缺失数据或回测失败时将显示具体原因，并可能阻塞排名刷新。"
        ),
        "detail": {"symbol": symbol, "duration": duration, "total": len(gaps), "items": gaps[:20]},
    }]


def coverage_gaps(report: dict[str, Any], *, primary_interval: str | None = None) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    gaps.extend(_main_kline_gaps(report))
    for table in report.get("tables") or []:
        table_name = str(table.get("table") or "")
        for row_index, row in enumerate(table.get("rows") or []):
            if _skip_coverage_gap(table_name, row, primary_interval=primary_interval):
                continue
            status = row.get("status")
            if status in {"healthy", None}:
                continue
            if table_name in SPARSE_COVERAGE_TABLES and status != "missing":
                continue
            if table_name == "orderbook_features" and _orderbook_coverage_ok(row):
                continue
            gaps.append({
                "table": table_name,
                "status": status,
                "missingReason": str(row.get("missingReason") or status),
                "group": _coverage_group_label(row),
                "coveragePct": row.get("coveragePct"),
                "rowIndex": row_index,
            })
    return gaps


def _main_kline_gaps(report: dict[str, Any]) -> list[dict[str, Any]]:
    main = report.get("mainRange") if isinstance(report.get("mainRange"), dict) else {}
    row_count = int(main.get("rowCount") or 0)
    status = str(main.get("status") or "")
    if row_count > 0 and status in {"healthy", "partial"}:
        return []
    return [{
        "table": "klines",
        "status": status or "missing",
        "missingReason": str(main.get("missingReason") or "no_rows"),
        "group": str(report.get("interval") or ""),
        "coveragePct": None,
        "rowIndex": 0,
    }]


def _orderbook_coverage_ok(row: dict[str, Any]) -> bool:
    pct = row.get("coveragePct")
    if pct is None:
        return False
    return float(pct) >= MIN_ORDERBOOK_COVERAGE_PCT


def _skip_coverage_gap(table_name: str, row: dict[str, Any], *, primary_interval: str | None) -> bool:
    if table_name == "orderbook_ticks":
        return True
    if table_name == "klines" and primary_interval:
        row_interval = row.get("interval")
        if row_interval and str(row_interval) != str(primary_interval):
            return True
    return False


def _coverage_group_label(row: dict[str, Any]) -> str:
    parts = [str(row.get(key) or "") for key in ("symbol", "interval", "source") if row.get(key)]
    return " · ".join(parts)


def ranking_failure_alerts(ranking_payload: dict[str, Any]) -> list[dict[str, Any]]:
    failures = ranking_payload.get("rankingFailures") or []
    if not failures:
        return []
    reasons = sorted({str(item.get("error") or "unknown") for item in failures})
    detail = "; ".join(reasons[:3])
    return [{
        "id": "factor_ranking_failures",
        "level": ALERT_WARNING,
        "code": "factor_ranking_failures",
        "title": "部分单因子排名计算失败",
        "message": f"本轮排名有 {len(failures)} 个因子未计入（示例：{detail}）。请检查特征列或数据覆盖后重新刷新排名。",
        "detail": {"failureCount": len(failures), "samples": failures[:5]},
    }]


def combination_failure_alerts(symbol: str, duration: str) -> list[dict[str, Any]]:
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        return []
    search = cached.get("search") if isinstance(cached.get("search"), dict) else {}
    diagnostics = cached.get("searchDiagnostics") if isinstance(cached.get("searchDiagnostics"), dict) else {}
    failure_counts = diagnostics.get("failureReasonCounts") if isinstance(diagnostics, dict) else {}
    failures = cached.get("failures") if isinstance(cached.get("failures"), list) else []
    if not failure_counts and not search and not failures:
        return []
    message = combination_failure_message(search, diagnostics, failure_counts, failures)
    return [] if not message else [{
        "id": "combo_backtest_failed",
        "level": ALERT_ERROR,
        "code": "combo_backtest_failed",
        "title": "组合因子回测未通过",
        "message": message,
        "detail": {
            "entryRows": search.get("entryRows") or diagnostics.get("entryRows"),
            "failureReasonCounts": failure_counts,
            "search": search,
        },
    }]


def combination_failure_message(search: dict, diagnostics: dict, failure_counts: dict, failures: list) -> str:
    entry_rows = search.get("entryRows") or diagnostics.get("entryRows")
    min_sample = diagnostics.get("targetCriteria", {}).get("validationMinSampleCount")
    parts = []
    if entry_rows is not None and min_sample is not None and int(entry_rows) < int(min_sample):
        parts.append(f"组合回测样本不足（{entry_rows} < {min_sample}），最近一轮高胜率组合搜索未产出可用结果。")
    if failure_counts:
        top_reason = next(iter(failure_counts))
        top_count = failure_counts[top_reason]
        parts.append(f"失败原因统计：{top_reason} × {top_count}。")
        criteria = diagnostics.get("targetCriteria") if isinstance(diagnostics, dict) else {}
        min_wr = criteria.get("validationMinWinRate") if isinstance(criteria, dict) else None
        if top_reason == "validation_win_rate_below_min" and min_wr is not None:
            parts.append(
                f"验证集胜率未达门槛（≥ {float(min_wr):.1%}）。"
                "请重新触发组合刷新；亦可查看「高胜率目标组合」结果。"
            )
    if failures and not parts:
        parts.append(f"组合回测失败 {len(failures)} 条，请查看详情。")
    return " ".join(parts)
