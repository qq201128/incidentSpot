from __future__ import annotations

from typing import Any, Callable

AGENT_FACTOR_SOURCE_FILE = "agent_mined_factor_library.json"
MINED_FACTOR_SOURCE_FILE = "mined_factor_library.json"


def select_base_candidates(
    candidates: list[Any],
    config: Any,
    *,
    rank_key: Callable[[Any], tuple],
) -> list[Any]:
    ranked = sorted(candidates, key=rank_key, reverse=True)
    selected: list[Any] = []
    selected_ids: set[str] = set()
    for source, limit in _source_limits(config):
        _append_source_candidates(selected, selected_ids, ranked, source, limit, config.base_factor_limit)
    _append_remaining_candidates(selected, selected_ids, ranked, config.base_factor_limit)
    return selected


def _source_limits(config: Any) -> tuple[tuple[str, int], ...]:
    return (
        ("native", config.native_factor_limit),
        ("mined", config.mined_factor_limit),
        ("agent", config.agent_factor_limit),
    )


def _append_source_candidates(
    selected: list[Any],
    selected_ids: set[str],
    ranked: list[Any],
    source: str,
    limit: int,
    total_limit: int,
) -> None:
    source_count = 0
    for candidate in ranked:
        if len(selected) >= total_limit or source_count >= limit:
            return
        if _source_group(candidate) != source:
            continue
        source_count += _append_candidate(selected, selected_ids, candidate)


def _append_remaining_candidates(
    selected: list[Any],
    selected_ids: set[str],
    ranked: list[Any],
    total_limit: int,
) -> None:
    for candidate in ranked:
        if len(selected) >= total_limit:
            return
        _append_candidate(selected, selected_ids, candidate)


def _append_candidate(
    selected: list[Any],
    selected_ids: set[str],
    candidate: Any,
) -> int:
    if candidate.factor.name in selected_ids:
        return 0
    selected.append(candidate)
    selected_ids.add(candidate.factor.name)
    return 1


def _source_group(candidate: Any) -> str:
    source = candidate.factor.source_file
    if source == AGENT_FACTOR_SOURCE_FILE:
        return "agent"
    if source == MINED_FACTOR_SOURCE_FILE:
        return "mined"
    return "native"
