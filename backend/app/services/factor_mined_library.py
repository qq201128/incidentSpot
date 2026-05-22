from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.services.factor_combo_display import combo_display_name
from app.services.factor_learning_common import (
    SUCCESS_PROFIT_FACTOR_MIN,
    SUCCESS_WIN_RATE_MIN,
    edge_score,
    finite,
    round_metric,
    utc_now,
)
from app.services.factor_learning_memory_store import FACTOR_LEARNING_DIR
from app.services.json_atomic_io import load_json_object, save_json_object
from app.services.factor_metric_enrichment import factor_score
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection

MINED_FACTOR_LIBRARY_VERSION = "mined_factor_library_v1"
MINED_FACTOR_LIBRARY_PATH = FACTOR_LEARNING_DIR / "mined_factor_library.json"
MINED_FACTOR_SOURCE = "factor_combo_ranking"
MINED_FACTOR_SOURCE_FILE = "mined_factor_library.json"
SUMMARY_FACTOR_LIMIT = 12


def load_mined_factor_library(path: Path | None = None) -> dict[str, Any]:
    target = path or MINED_FACTOR_LIBRARY_PATH
    if not target.exists():
        return _empty_library()
    payload = load_json_object(target)
    if not isinstance(payload, dict):
        raise ValueError(f"mined factor library is not an object: {target}")
    return payload


def mined_factor_rows_for_duration(symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = load_mined_factor_library().get("factors") or []
    return [
        deepcopy(row)
        for row in rows
        if _row_symbol(row) == symbol.strip().upper() and str(row.get("duration")) == duration
    ]


def mined_factor_rows() -> list[dict[str, Any]]:
    return deepcopy(load_mined_factor_library().get("factors") or [])


def mined_factor_definition(row: dict[str, Any]) -> FactorDefinition:
    duration = str(row.get("duration") or "")
    return FactorDefinition(
        name=str(row["factorName"]),
        category=FactorCategory.PERFORMANCE,
        description=str(row.get("factorDisplayName") or row["factorName"]),
        formula=str(row.get("formula") or row["factorName"]),
        source_file=MINED_FACTOR_SOURCE_FILE,
        timeframes=(duration,) if duration else (),
        direction=FactorDirection.HIGHER_BETTER,
        parameters=_mined_factor_parameters(row),
    )


def mined_factor_payload(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    display_name = _combo_display_name_for_row(row)
    return {
        "name": str(row["factorName"]),
        "category": "performance",
        "categoryName": "绩效因子",
        "displayName": display_name,
        "description": display_name,
        "formula": str(row.get("formula") or row["factorName"]),
        "sourceFile": MINED_FACTOR_SOURCE_FILE,
        "timeframes": [str(row.get("duration"))] if row.get("duration") else [],
        "direction": FactorDirection.HIGHER_BETTER.value,
        "parameters": _mined_factor_parameters(row),
        "symbol": _row_symbol(row),
        "duration": str(row.get("duration") or ""),
        "promotionCount": int(row.get("promotionCount") or 0),
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
    }


def mined_factor_library_summary(symbol: str, duration: str) -> dict[str, Any]:
    rows = mined_factor_rows_for_duration(symbol, duration)
    rows.sort(key=_library_score, reverse=True)
    return enrich_mined_factor_library_summary(
        {
            "version": MINED_FACTOR_LIBRARY_VERSION,
            "symbol": symbol.strip().upper(),
            "duration": duration,
            "total": len(rows),
            "thresholds": _threshold_payload(),
            "factors": rows[:SUMMARY_FACTOR_LIMIT],
        }
    )


def enrich_mined_factor_library_summary(library: dict[str, Any]) -> dict[str, Any]:
    if not library:
        return library
    enriched = deepcopy(library)
    factors = enriched.get("factors")
    if isinstance(factors, list):
        enriched["factors"] = [_enriched_summary_row(row) for row in factors]
    return enriched


def regular_library_combination_rows_for_duration(
    symbol: str,
    duration: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        _library_combination_row(row)
        for row in mined_factor_rows_for_duration(symbol, duration)
        if _is_regular_combination_name(str(row.get("factorName") or ""))
    ]
    selected = [row for row in rows if row is not None]
    selected.sort(key=_library_combination_rank_key, reverse=True)
    return selected[:limit]


def upsert_good_combinations(report: dict[str, Any]) -> dict[str, Any]:
    candidates = [_library_row(report, row) for row in report.get("ranking") or [] if _is_good_combo(row)]
    candidates = [row for row in candidates if row is not None]
    library = load_mined_factor_library()
    if not candidates:
        return _promotion_report(report, promoted=0, total=len(library.get("factors") or []))
    merged = _merged_rows(library.get("factors") or [], candidates)
    payload = {
        "version": MINED_FACTOR_LIBRARY_VERSION,
        "updatedAt": utc_now(),
        "thresholds": _threshold_payload(),
        "factors": merged,
    }
    _save_library(payload)
    return _promotion_report(report, promoted=len(candidates), total=len(merged))


def _library_row(report: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    factor_name = str(row.get("factorName") or "")
    members = row.get("members")
    if not factor_name or not isinstance(members, list) or not members:
        return None
    now = utc_now()
    members_payload = [_member_payload(member) for member in members]
    display_name = combo_display_name(members_payload)
    return {
        "symbol": str(report["symbol"]).strip().upper(),
        "duration": str(report["duration"]),
        "factorName": factor_name,
        "factorDisplayName": display_name,
        "description": display_name,
        "formula": str(row.get("formula") or factor_name),
        "method": str(row.get("method") or ""),
        "category": "performance",
        "source": MINED_FACTOR_SOURCE,
        "members": members_payload,
        "threshold": finite(row.get("threshold")),
        "minTrades": int(row.get("minTrades") or row.get("totalPeriods") or 0),
        "metrics": _metric_payload(row),
        "score": _standard_score(row),
        "searchScore": _row_score(row),
        "searchProfitFactor": finite(row.get("profitFactor")),
        "firstSeenAt": now,
        "lastSeenAt": now,
        "promotionCount": 1,
    }


def _library_combination_row(row: dict[str, Any]) -> dict[str, Any] | None:
    factor_name = str(row.get("factorName") or "")
    members = row.get("members")
    if not factor_name or not isinstance(members, list) or not members:
        return None
    metrics = row.get("metrics") or {}
    validation = metrics.get("validation") if isinstance(metrics, dict) else None
    members_payload = [_member_payload(member) for member in members]
    display_name = combo_display_name(members_payload)
    return {
        "factorName": factor_name,
        "factorDisplayName": display_name,
        "description": display_name,
        "formula": str(row.get("formula") or factor_name),
        "method": str(row.get("method") or ""),
        "members": members_payload,
        "threshold": finite(row.get("threshold")),
        "minTrades": int(row.get("minTrades") or metrics.get("totalPeriods") or 0),
        "winRate": finite(metrics.get("winRate")),
        "profitFactor": finite(metrics.get("profitFactor")),
        "sharpe": finite(metrics.get("sharpe")),
        "ir": finite(metrics.get("ir")),
        "totalPeriods": int(metrics.get("totalPeriods") or 0),
        "contribution": finite(metrics.get("contribution")),
        "factorScore": _display_score(row),
        "searchScore": finite(row.get("searchScore")),
        "searchProfitFactor": finite(row.get("searchProfitFactor")),
        "source": str(row.get("source") or MINED_FACTOR_SOURCE),
        "walkForward": validation if isinstance(validation, dict) else None,
        "walkForwardPassed": _validation_passed(validation),
    }


def _merged_rows(existing: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_row_key(row): deepcopy(row) for row in existing}
    for row in candidates:
        key = _row_key(row)
        previous = by_key.get(key)
        if previous is not None:
            row["firstSeenAt"] = previous.get("firstSeenAt") or row["firstSeenAt"]
            row["promotionCount"] = int(previous.get("promotionCount") or 0) + 1
        by_key[key] = row
    rows = list(by_key.values())
    rows.sort(key=_library_score, reverse=True)
    return rows


def _is_good_combo(row: dict[str, Any]) -> bool:
    win_rate = finite(row.get("winRate"))
    profit_factor = finite(row.get("profitFactor"))
    return (
        win_rate is not None
        and profit_factor is not None
        and win_rate >= SUCCESS_WIN_RATE_MIN
        and profit_factor >= SUCCESS_PROFIT_FACTOR_MIN
    )


def _combo_display_name_for_row(row: dict[str, Any]) -> str:
    members = row.get("members")
    if isinstance(members, list) and len(members) >= 2:
        return combo_display_name([_member_payload(member) for member in members])
    return str(row.get("factorDisplayName") or row.get("description") or row["factorName"])


def _enriched_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(row)
    display_name = _combo_display_name_for_row(enriched)
    enriched["factorDisplayName"] = display_name
    enriched["description"] = display_name
    return enriched


def _member_payload(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(member["name"]),
        "displayName": str(member.get("displayName") or member["name"]),
        "category": str(member.get("category") or "unknown"),
        "orientation": int(member.get("orientation") or 1),
    }


def _metric_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "winRate": finite(row.get("winRate")),
        "profitFactor": finite(row.get("profitFactor")),
        "sharpe": finite(row.get("sharpe")),
        "ir": finite(row.get("ir")),
        "totalPeriods": int(row.get("totalPeriods") or 0),
        "contribution": finite(row.get("contribution")),
    }


def _promotion_report(report: dict[str, Any], *, promoted: int, total: int) -> dict[str, Any]:
    return {
        "symbol": str(report.get("symbol") or "").strip().upper(),
        "duration": str(report.get("duration") or ""),
        "promoted": promoted,
        "libraryTotal": total,
        "thresholds": _threshold_payload(),
    }


def _threshold_payload() -> dict[str, float]:
    return {
        "minWinRate": SUCCESS_WIN_RATE_MIN,
        "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN,
    }


def _mined_factor_parameters(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "members": [
            str(member["name"])
            for member in row.get("members") or []
            if isinstance(member, dict) and member.get("name")
        ],
        "symbol": _row_symbol(row),
        "duration": str(row.get("duration") or ""),
    }


def _empty_library() -> dict[str, Any]:
    return {
        "version": MINED_FACTOR_LIBRARY_VERSION,
        "thresholds": _threshold_payload(),
        "factors": [],
    }


def _save_library(payload: dict[str, Any]) -> None:
    save_json_object(MINED_FACTOR_LIBRARY_PATH, payload)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_row_symbol(row), str(row.get("duration")), str(row.get("factorName")))


def _is_regular_combination_name(name: str) -> bool:
    return bool(name) and not name.startswith("goal_combo__")


def _library_combination_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _score_num(row.get("factorScore")),
        _score_num(row.get("winRate")),
        _score_num(row.get("profitFactor")),
        _score_num(row.get("sharpe")),
    )


def _validation_passed(validation: Any) -> bool:
    if not isinstance(validation, dict):
        return True
    return str(validation.get("status") or "") == "passed"


def _row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()


def _library_score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or row
    if _is_stability_rejected(row):
        return 0.0
    return _standard_score(metrics)


def _row_score(row: dict[str, Any]) -> float:
    return round_metric(edge_score(row), 6)


def _standard_score(row: dict[str, Any]) -> float:
    return factor_score(row)


def _display_score(row: dict[str, Any]) -> float:
    if _is_stability_rejected(row):
        return 0.0
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    return _standard_score(metrics)


def _is_stability_rejected(row: dict[str, Any]) -> bool:
    return str(row.get("stabilityStatus") or "") == "rejected"


def _score_num(value: Any) -> float:
    number = finite(value)
    return number if number is not None else float("-inf")
