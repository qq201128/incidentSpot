from __future__ import annotations

DEFAULT_RANKING_PAGE_SIZE = 8
MAX_RANKING_PAGE_SIZE = 100

SEARCH_FIELDS = (
    "factorName",
    "name",
    "displayName",
    "factorDisplayName",
    "description",
    "category",
    "categoryName",
    "sourceLabel",
    "sourceFile",
    "duration",
)


def build_ranking_page(
    ranking: list[dict],
    query: str | None,
    page: int,
    page_size: int,
) -> dict:
    filtered = _filter_ranking_rows(ranking, query)
    size = max(1, min(int(page_size), MAX_RANKING_PAGE_SIZE))
    page_count = _page_count(len(filtered), size)
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


def _page_count(total: int, page_size: int) -> int:
    if total <= 0:
        return 1
    return (total + page_size - 1) // page_size


def _filter_ranking_rows(ranking: list[dict], query: str | None) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return list(ranking)
    return [row for row in ranking if _ranking_row_matches(row, q)]


def _ranking_row_matches(row: dict, query: str) -> bool:
    return query in _ranking_row_search_text(row)


def _ranking_row_search_text(row: dict) -> str:
    values = [str(row.get(field) or "") for field in SEARCH_FIELDS]
    return " ".join(values).lower()
