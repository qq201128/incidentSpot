from __future__ import annotations

from pathlib import Path

from app.services.model_family_candidates import read_model_candidate_library


def recorded_model_search_keys(
    family: str,
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> frozenset[str]:
    library = read_model_candidate_library(family, symbol, duration, artifact_root=artifact_root)
    return frozenset(
        str(row.get("searchKey"))
        for row in library["records"]
        if row.get("searchKey")
    )
