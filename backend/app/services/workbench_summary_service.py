from __future__ import annotations

from datetime import datetime, timezone

from app.db.session import get_conn
from app.services.combo_event_governance import combo_event_monitoring
from app.services.event_ai_history import ai_history_success
from app.services.workbench_summary_cache import get_cached_workbench_summary


def build_workbench_summary(symbol: str, duration: str) -> dict:
    safe_symbol = symbol.strip().upper()
    conn = get_conn()
    try:
        counts = _event_counts(conn, safe_symbol)
        event_total = sum(counts.values())
        monitoring = combo_event_monitoring(safe_symbol, duration)
        return {
            "symbol": safe_symbol,
            "duration": duration,
            "dataSource": "Binance Index",
            "serverTime": datetime.now(timezone.utc).isoformat(),
            "eventCounts": counts,
            "eventTotal": event_total,
            "hasOpenPosition": _has_open_position(conn, safe_symbol),
            "aiHistorySuccess": ai_history_success(conn, safe_symbol),
            "shadowEventDeviation": monitoring["shadowEventDeviation"],
            "batchComboEventDemotion": monitoring["batchComboDemotion"],
            "factorCandidateEventDemotion": monitoring.get("factorCandidateDemotion"),
            "simulationObservation": monitoring.get("simulationObservation"),
            "governanceWarming": bool(monitoring.get("warming")),
        }
    finally:
        conn.close()


def get_workbench_summary(symbol: str, duration: str) -> dict:
    return get_cached_workbench_summary(
        symbol,
        duration,
        build=build_workbench_summary,
    )


def _has_open_position(conn, symbol: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM events e
        INNER JOIN orders o ON o.event_id = e.id
        WHERE e.symbol = ? AND e.status = 'OPEN'
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    return row is not None


def _event_counts(conn, symbol: str) -> dict:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS total
        FROM events
        WHERE symbol = ?
        GROUP BY status
        """,
        (symbol,),
    ).fetchall()
    counts = {"OPEN": 0, "SETTLED": 0, "FAILED": 0}
    for row in rows:
        counts[row["status"]] = int(row["total"])
    return counts
