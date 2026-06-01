from __future__ import annotations

from typing import Any

from app.services.factor_ranking_page import MAX_RANKING_PAGE_SIZE


def build_combination_ranking_page(
    ranking: list[dict[str, Any]],
    query: str | None,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    filtered = _filter_ranking_rows(ranking, query)
    size = max(1, min(int(page_size), MAX_RANKING_PAGE_SIZE))
    page_count = max(1, (len(filtered) + size - 1) // size) if filtered else 1
    current = min(max(1, int(page)), page_count)
    start = (current - 1) * size
    return {
        "ranking": filtered[start:start + size],
        "total": len(filtered),
        "unfilteredTotal": len(ranking),
        "page": current,
        "pageSize": size,
        "pageCount": page_count,
        "query": (query or "").strip(),
    }


def _filter_ranking_rows(ranking: list[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return list(ranking)
    return [row for row in ranking if _ranking_row_matches(row, q)]


def _ranking_row_matches(row: dict[str, Any], query: str) -> bool:
    haystack = " ".join(
        [
            str(row.get("factorName") or ""),
            str(row.get("factorDisplayName") or ""),
            str(row.get("description") or ""),
            _member_search_text(row.get("members")),
        ]
    ).lower()
    return query in haystack


def _member_search_text(members: object) -> str:
    if not isinstance(members, list):
        return ""
    return " ".join(
        f"{member.get('name', '')} {member.get('displayName', '')}"
        for member in members
        if isinstance(member, dict)
    )
