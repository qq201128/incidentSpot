from __future__ import annotations

import json
from typing import Any

from app.db.session import get_conn


def latest_event_final_decision(symbol: str, duration: str) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT symbol, duration, open_time, decision, direction,
                   probability_up, confidence, final_score, regime_label,
                   candidate_count, reason_codes, settled_at, decision_correct
            FROM event_final_decisions
            WHERE symbol = ? AND duration = ?
            ORDER BY open_time DESC
            LIMIT 1
            """,
            (sym, duration),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return _latest_payload(dict(row))


def event_final_decision_summary(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT decision, direction, confidence, regime_label, decision_correct
            FROM event_final_decisions
            WHERE symbol = ? AND duration = ?
            """,
            (sym, duration),
        ).fetchall()
    finally:
        conn.close()
    items = [dict(row) for row in rows]
    return {
        "symbol": sym,
        "duration": duration,
        "overall": _summary_stats(items),
        "byDecision": _grouped_stats(items, "decision"),
        "byRegime": _grouped_stats(items, "regime_label"),
    }


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row.get("decision_correct") is not None]
    wins = sum(int(row["decision_correct"]) for row in settled)
    rate = None if not settled else wins / len(settled)
    return {"count": len(rows), "settled": len(settled), "wins": wins, "winRate": rate}


def _grouped_stats(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return [{"key": name, **_summary_stats(items)} for name, items in sorted(groups.items())]


def _latest_payload(row: dict[str, Any]) -> dict[str, Any]:
    regime = _parse_regime_label(row.get("regime_label"))
    reasons = _parse_reason_codes(row.get("reason_codes"))
    decision = str(row.get("decision") or "SKIP").upper()
    settled = row.get("decision_correct")
    return {
        "symbol": row.get("symbol"),
        "duration": row.get("duration"),
        "openTime": int(row.get("open_time") or 0),
        "decision": decision,
        "direction": row.get("direction"),
        "probabilityUp": row.get("probability_up"),
        "confidence": row.get("confidence"),
        "finalScore": row.get("final_score"),
        "regimeLabel": row.get("regime_label"),
        "trendRegime": regime["trend"],
        "volRegime": regime["vol"],
        "candidateCount": int(row.get("candidate_count") or 0),
        "reasonCodes": reasons,
        "skipReason": _skip_reason(decision, reasons),
        "settledAt": row.get("settled_at"),
        "decisionCorrect": settled if settled is None else bool(int(settled)),
    }


def _parse_regime_label(label: Any) -> dict[str, str]:
    raw = str(label or "unknown:unknown")
    trend, _, vol = raw.partition(":")
    return {"trend": trend or "unknown", "vol": vol or "unknown"}


def _parse_reason_codes(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [str(raw)] if str(raw).strip() else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _skip_reason(decision: str, reasons: list[str]) -> str | None:
    if decision != "SKIP":
        return None
    return reasons[0] if reasons else "未给出原因"
