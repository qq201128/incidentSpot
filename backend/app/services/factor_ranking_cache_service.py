from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.factor_cache_metadata import cache_status, ranking_cache_metadata
from app.services.auto_trade_service import list_auto_trade_settings

logger = logging.getLogger("uvicorn.error")


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def get_cached_ranking(symbol: str, duration: str) -> dict[str, Any] | None:
    """Return ranking cache payload or None if no row."""
    sym = _norm_symbol(symbol)
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT payload, total, updated_at
            FROM factor_ranking_cache
            WHERE symbol = ? AND duration = ?
            """,
            (sym, duration),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        logger.warning("factor_ranking_cache corrupt JSON for %s %s", sym, duration)
        return None
    ranking, cache_meta, diagnostics, failures = _ranking_payload(payload)
    if not isinstance(ranking, list):
        return None
    return {
        "ranking": ranking,
        "total": int(row["total"]),
        "updatedAt": str(row["updated_at"]),
        "cacheMeta": cache_meta,
        "cacheStatus": cache_status(cache_meta, sym),
        "rankingDiagnostics": diagnostics,
        "rankingFailures": failures,
    }


def save_cached_ranking(
    symbol: str,
    duration: str,
    ranking: list[dict[str, Any]],
    *,
    diagnostics: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> None:
    sym = _norm_symbol(symbol)
    payload = json.dumps(
        {
            "ranking": ranking,
            "cacheMeta": ranking_cache_metadata(sym, duration),
            "rankingDiagnostics": diagnostics or {},
            "rankingFailures": failures or [],
        },
        ensure_ascii=False,
    )
    total = len(ranking)
    ts = datetime.now(timezone.utc).isoformat()

    def _persist() -> None:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO factor_ranking_cache(symbol, duration, updated_at, total, payload)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(symbol, duration) DO UPDATE SET
                  updated_at = excluded.updated_at,
                  total = excluded.total,
                  payload = excluded.payload
                """,
                (sym, duration, ts, total, payload),
            )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_persist)


def factor_ranking_precomputed_symbols() -> list[str]:
    raw = os.getenv("FACTOR_RANKING_SYMBOLS", "BTCUSDT").strip()
    base = [p.strip().upper() for p in raw.split(",") if p.strip()] if raw else ["BTCUSDT"]
    enabled = [settings.symbol.strip().upper() for settings in list_auto_trade_settings() if settings.enabled]
    symbols = list(dict.fromkeys(base + enabled))
    return symbols or ["BTCUSDT"]


def _ranking_payload(payload: Any) -> tuple[Any, dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    if isinstance(payload, list):
        return payload, None, {}, []
    if not isinstance(payload, dict):
        return None, None, {}, []
    diagnostics = payload.get("rankingDiagnostics")
    failures = payload.get("rankingFailures")
    return (
        payload.get("ranking"),
        payload.get("cacheMeta"),
        diagnostics if isinstance(diagnostics, dict) else {},
        failures if isinstance(failures, list) else [],
    )
