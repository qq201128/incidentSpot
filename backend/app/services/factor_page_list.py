from __future__ import annotations

from typing import Any

from app.services.factor_catalog_summaries import AGENT_FACTOR_SOURCE_FILE
from app.services.factor_mined_library import MINED_FACTOR_SOURCE_FILE

SOURCE_LOCAL = "local_definition"
SOURCE_AGENT = "agent_candidate"
SOURCE_LSTM = "lstm_shadow"
SOURCE_COMBO = "composite_cache"


def filter_factor_rows(rows: list[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return list(rows)
    return [row for row in rows if row_matches_query(row, q)]


def row_matches_query(row: dict[str, Any], query: str) -> bool:
    haystack = " ".join(
        str(row.get(key) or "")
        for key in ("name", "displayName", "description", "formula", "categoryName", "sourceLabel")
    ).lower()
    return query in haystack or query in source_label(classify_factor_source(row.get("sourceFile"), row.get("name"))).lower()


def paginate_rows(rows: list[dict[str, Any]], page: int, page_size: int) -> dict[str, Any]:
    total = len(rows)
    size = max(1, int(page_size))
    current = max(1, int(page))
    page_count = max(1, (total + size - 1) // size) if total else 1
    current = min(current, page_count)
    start = (current - 1) * size
    return {"items": rows[start:start + size], "total": total, "page": current, "pageSize": size, "pageCount": page_count}


def sort_combo_rows_by_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=combo_list_sort_key, reverse=True)


def classify_factor_source(source_file: object, name: object) -> str:
    source_file_text = str(source_file or "")
    name_text = str(name or "").lower()
    if source_file_text == AGENT_FACTOR_SOURCE_FILE or name_text.startswith("agent__"):
        return SOURCE_AGENT
    if source_file_text == MINED_FACTOR_SOURCE_FILE:
        return SOURCE_COMBO
    if "lstm" in name_text or "lstm" in source_file_text.lower():
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


def combo_list_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(row.get("factorScore") or 0.0),
        abs(float(row.get("ir") or 0.0)),
        str(row.get("name") or ""),
    )
