from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

STATUS_ACTIVE = "active"
STATUS_COLLECTING = "collecting"
STATUS_DEMOTED = "demoted"
STATUS_PAUSED = "paused"
REASON_OFFLINE_PROMOTION = "offline_promotion"
RECENT_SAMPLE_LIMIT = 30
ACTIVE_SAMPLE_COUNT = 20
ACTIVE_WIN_RATE_MIN = 0.70
MIN_PROFIT_FACTOR = 1.0
LOSS_STREAK_LIMIT = 5
DEFAULT_QTY = 5.0


def promote_high_winrate_strategy(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    metrics = _empty_metrics()
    payload = _status_payload(STATUS_COLLECTING, REASON_OFFLINE_PROMOTION, metrics)
    conn = get_conn()
    try:
        _ensure_table(conn)
        _set_strategy_slot(conn, sym, duration, enabled=True, live_trading_enabled=False)
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    return {"symbol": sym, "duration": duration, **payload}


def evaluate_high_winrate_demotion(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        _ensure_table(conn)
        rows = _settled_rows(conn, sym, duration)
        current = _current_status(conn, sym, duration)
        metrics = _metrics(rows)
        decision = _decision(metrics)
        payload = _evaluation_payload(decision, current, metrics)
        _sync_strategy_slot_for_status(conn, sym, duration, payload["status"])
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    return {"symbol": sym, "duration": duration, **payload}


def high_winrate_demotion_status(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        _ensure_table(conn)
        row = _current_status(conn, sym, duration)
    finally:
        conn.close()
    return _public_status(row, sym, duration)


def _settled_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT open_time, prediction_correct, actual_return, high_winrate_rule
        FROM predictions
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
          AND settled_at IS NOT NULL
        ORDER BY open_time DESC
        LIMIT ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, symbol, duration, RECENT_SAMPLE_LIMIT),
    ).fetchall()
    return [dict(row) for row in rows]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["actual_return"]) for row in rows if row.get("actual_return") is not None]
    wins = sum(1 for row in rows if bool(row.get("prediction_correct")))
    return {
        "sampleCount": len(rows),
        "winRate": _ratio(wins, len(rows)),
        "profitFactor": _profit_factor(returns),
        "consecutiveLosses": _consecutive_losses(rows),
        "latestRule": str(rows[0].get("high_winrate_rule") or "") if rows else None,
    }


def _decision(metrics: dict[str, Any]) -> dict[str, str]:
    if metrics["consecutiveLosses"] >= LOSS_STREAK_LIMIT:
        return {"status": STATUS_DEMOTED, "reason": "consecutive_losses"}
    if metrics["sampleCount"] < ACTIVE_SAMPLE_COUNT:
        return {"status": STATUS_COLLECTING, "reason": "insufficient_settled_samples"}
    if _lt(metrics["winRate"], ACTIVE_WIN_RATE_MIN):
        return {"status": STATUS_DEMOTED, "reason": "live_win_rate_below_target"}
    if _lt(metrics["profitFactor"], MIN_PROFIT_FACTOR):
        return {"status": STATUS_DEMOTED, "reason": "profit_factor_below_one"}
    return {"status": STATUS_ACTIVE, "reason": "stable_live_target_met"}


def _evaluation_payload(
    decision: dict[str, str],
    current: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if current.get("status") == STATUS_PAUSED:
        return _status_payload(STATUS_PAUSED, str(current.get("reason") or "paused"), metrics)
    return _status_payload(decision["status"], decision["reason"], metrics)


def _status_payload(status: str, reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "metrics": metrics,
        "thresholds": _thresholds(),
        "evaluatedAt": _utc_now(),
    }


def _write_status(conn: Any, symbol: str, duration: str, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    conn.execute(
        """
        INSERT INTO high_winrate_strategy_status(
          strategy_key, symbol, duration, status, reason, details_json,
          sample_count, win_rate, profit_factor, consecutive_losses, evaluated_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_key, symbol, duration) DO UPDATE SET
          status = excluded.status, reason = excluded.reason, details_json = excluded.details_json,
          sample_count = excluded.sample_count, win_rate = excluded.win_rate,
          profit_factor = excluded.profit_factor, consecutive_losses = excluded.consecutive_losses,
          evaluated_at = excluded.evaluated_at, updated_at = excluded.updated_at
        """,
        (
            HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            symbol,
            duration,
            payload["status"],
            payload["reason"],
            json.dumps(payload, ensure_ascii=False),
            int(metrics["sampleCount"]),
            metrics["winRate"],
            metrics["profitFactor"],
            int(metrics["consecutiveLosses"]),
            payload["evaluatedAt"],
            _utc_now(),
        ),
    )


def _current_status(conn: Any, symbol: str, duration: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT status, reason, details_json, evaluated_at
        FROM high_winrate_strategy_status
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, symbol, duration),
    ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _public_status(row: dict[str, Any], symbol: str, duration: str) -> dict[str, Any]:
    if not row:
        return {"symbol": symbol, "duration": duration, "status": "unknown", "reason": "not_evaluated"}
    details = json.loads(row["details_json"]) if row.get("details_json") else {}
    return {
        "symbol": symbol,
        "duration": duration,
        "status": details.get("status") or row.get("status"),
        "reason": details.get("reason") or row.get("reason"),
        **details,
    }


def _set_strategy_slot(
    conn: Any,
    symbol: str,
    duration: str,
    *,
    enabled: bool,
    live_trading_enabled: bool | None,
) -> None:
    row = _strategy_slot(conn, duration)
    live_enabled = _live_trading_enabled(row, live_trading_enabled)
    conn.execute(
        """
        INSERT OR REPLACE INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            duration,
            int(enabled),
            int(live_enabled),
            symbol,
            int(DURATION_TO_MINUTES[duration]),
            float(row["qty"]) if row else DEFAULT_QTY,
            _utc_now(),
        ),
    )


def _sync_strategy_slot_for_status(conn: Any, symbol: str, duration: str, status: str) -> None:
    if status == STATUS_PAUSED:
        _set_strategy_slot(conn, symbol, duration, enabled=False, live_trading_enabled=False)
        return
    live_override = False if status == STATUS_DEMOTED else None
    _set_strategy_slot(conn, symbol, duration, enabled=True, live_trading_enabled=live_override)


def _strategy_slot(conn: Any, duration: str) -> Any | None:
    return conn.execute(
        """
        SELECT qty, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ? AND duration = ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration),
    ).fetchone()


def _ensure_table(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS high_winrate_strategy_status (
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          sample_count INTEGER NOT NULL,
          win_rate REAL,
          profit_factor REAL,
          consecutive_losses INTEGER NOT NULL,
          evaluated_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (strategy_key, symbol, duration)
        )
        """
    )


def _thresholds() -> dict[str, Any]:
    return {
        "activeSampleCount": ACTIVE_SAMPLE_COUNT,
        "activeWinRateMin": ACTIVE_WIN_RATE_MIN,
        "minProfitFactor": MIN_PROFIT_FACTOR,
        "lossStreakLimit": LOSS_STREAK_LIMIT,
    }


def _empty_metrics() -> dict[str, Any]:
    return {"sampleCount": 0, "winRate": None, "profitFactor": None, "consecutiveLosses": 0, "latestRule": None}


def _profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if not values or losses == 0:
        return None
    return round(gains / losses, 4)


def _consecutive_losses(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if bool(row.get("prediction_correct")):
            break
        count += 1
    return count


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 4)


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _live_trading_enabled(row: Any | None, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    return bool(row["live_trading_enabled"]) if row else False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
