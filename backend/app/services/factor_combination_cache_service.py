from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.cache_payloads import CachePayloadDecodeError, decode_cache_payload
from app.services.factor_cache_metadata import cache_status, ranking_cache_metadata

logger = logging.getLogger("uvicorn.error")


def get_cached_combination_ranking(symbol: str, duration: str) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    row = _cache_row(symbol, duration)
    if row is None:
        return None
    payload = decode_cache_payload(
        row["payload"],
        cache_name="factor_combo_ranking_cache",
        identity={"symbol": sym, "duration": duration},
    )
    if not isinstance(payload, dict):
        return None
    cache_meta = payload.get("cacheMeta")
    return {
        **payload,
        "updatedAt": str(row["updated_at"]),
        "cacheStatus": cache_status(cache_meta, sym),
    }


def save_cached_combination_ranking(report: dict[str, Any]) -> None:
    symbol = str(report["symbol"]).strip().upper()
    duration = str(report["duration"])
    ranking = report.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("combination ranking report must contain a ranking list")
    if not ranking and not _has_search_diagnostics(report):
        existing = get_cached_combination_ranking(symbol, duration)
        if _cache_has_rows(existing):
            logger.warning(
                "skip overwriting non-empty factor combo cache with legacy empty ranking: %s %s existing=%s",
                symbol,
                duration,
                len(existing.get("ranking") or []),
            )
            return
    persisted = {**report, "cacheMeta": ranking_cache_metadata(symbol, duration)}
    payload = json.dumps(persisted, ensure_ascii=False)
    config = json.dumps(persisted.get("searchConfig") or {}, ensure_ascii=False)
    updated_at = _utc_now()
    ranking_total = len(ranking)

    def _persist() -> None:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO factor_combo_ranking_cache(symbol, duration, updated_at, total, search_config, payload)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, duration) DO UPDATE SET
                  updated_at = excluded.updated_at,
                  total = excluded.total,
                  search_config = excluded.search_config,
                  payload = excluded.payload
                """,
                (symbol, duration, updated_at, ranking_total, config, payload),
            )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_persist)


def _cache_row(symbol: str, duration: str) -> Any | None:
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT payload, total, updated_at
            FROM factor_combo_ranking_cache
            WHERE symbol = ? AND duration = ?
            """,
            (symbol.strip().upper(), duration),
        ).fetchone()
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_has_rows(cache: dict[str, Any] | None) -> bool:
    rows = None if cache is None else cache.get("ranking")
    return bool(isinstance(rows, list) and rows)


def _has_search_diagnostics(report: dict[str, Any]) -> bool:
    return isinstance(report.get("searchDiagnostics"), dict)
