from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.paper_live_stage_log import log_settlement_pending, log_settlement_success
from app.services.event_final_decision_service import settle_due_final_decisions
from app.services.trading_costs import ROUNDTRIP_COST_RATE

DAY_MS = 86_400_000
TARGET_WIN_RATE = 0.90
MIN_MONITOR_TRADES = 20
OVERFIT_DROP_LIMIT = 0.10
DURATION_MINUTES = {"10m": 10, "30m": 30, "60m": 60, "1d": 1440}
WINDOW_DAYS = (1, 3, 7, 30)


def settle_due_predictions(symbol: str, duration: str) -> dict[str, int]:
    def _settle() -> dict[str, int]:
        conn = get_conn()
        try:
            rows = _due_prediction_rows(conn, symbol, duration)
            settled = sum(1 for row in rows if _settle_prediction(conn, row, duration))
            decision_settled = settle_due_final_decisions(conn, symbol, duration, _max_open_time(conn, symbol))
            conn.commit()
            return {
                "checked": len(rows),
                "settled": settled,
                "pendingData": len(rows) - settled,
                "finalDecisionsSettled": decision_settled,
            }
        finally:
            conn.close()

    return run_db_write_with_retry(_settle)


def forward_validation_summary(symbol: str, duration: str, meta: dict[str, Any]) -> dict[str, Any]:
    conn = get_conn()
    try:
        rows = _settled_prediction_rows(conn, symbol, duration)
    finally:
        conn.close()
    gated_rows = [row for row in rows if bool(row["high_winrate_gate_passed"])]
    backtest = (meta.get("production_gate_backtest") or {}).get("win_rate")
    windows = {f"{days}d": _stats(_window_rows(gated_rows, days), days) for days in WINDOW_DAYS}
    return {
        "allPredictions": _stats(rows, None),
        "productionGate": _stats(gated_rows, None),
        "windows": windows,
        "perRule": _per_rule_stats(gated_rows),
        "overfitMonitor": _overfit_monitor(_stats(gated_rows, None), windows, backtest),
    }


def _due_prediction_rows(conn, symbol: str, duration: str) -> list[dict[str, Any]]:
    latest = _max_open_time(conn, symbol)
    if latest is None:
        return []
    max_open_time = int(latest) - _duration_ms(duration)
    rows = conn.execute(
        """
        SELECT id, signal_key, strategy_key, symbol, duration, open_time, direction, entry_price
        FROM predictions
        WHERE symbol = ? AND duration = ? AND settled_at IS NULL AND open_time <= ?
        ORDER BY open_time
        """,
        (symbol.upper(), duration, max_open_time),
    ).fetchall()
    return [dict(row) for row in rows]


def _max_open_time(conn, symbol: str) -> int | None:
    latest = conn.execute(
        "SELECT MAX(open_time) AS value FROM klines WHERE symbol = ? AND interval = '1m'",
        (symbol.upper(),),
    ).fetchone()
    return None if latest is None or latest["value"] is None else int(latest["value"])


def _settle_prediction(conn, row: dict[str, Any], duration: str) -> bool:
    entry_price = row["entry_price"] or _kline_close(conn, row["symbol"], int(row["open_time"]))
    exit_open_time = int(row["open_time"]) + _duration_ms(duration)
    exit_price = _kline_close(conn, row["symbol"], exit_open_time)
    if entry_price is None or exit_price is None:
        log_settlement_pending(
            conn,
            row,
            entry_price=entry_price,
            exit_open_time=exit_open_time,
            exit_price=exit_price,
        )
        return False
    actual_return = _directional_return(float(entry_price), float(exit_price), row["direction"])
    prediction_correct = _direction_correct(float(entry_price), float(exit_price), row["direction"])
    conn.execute(
        """
        UPDATE predictions
        SET entry_price = ?, exit_price = ?, actual_return = ?,
            prediction_correct = ?, settled_at = ?
        WHERE id = ?
        """,
        (entry_price, exit_price, actual_return, int(prediction_correct), _utc_now(), row["id"]),
    )
    log_settlement_success(
        conn,
        row,
        entry_price=float(entry_price),
        exit_open_time=exit_open_time,
        exit_price=float(exit_price),
        actual_return=actual_return,
        prediction_correct=prediction_correct,
    )
    return True


def _settled_prediction_rows(conn, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        WITH first_rows AS (
          SELECT MIN(id) AS id
          FROM predictions
          WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
          GROUP BY open_time, direction, COALESCE(high_winrate_rule, high_winrate_gate, '')
        )
        SELECT p.open_time, p.high_winrate_gate_passed, p.high_winrate_rule,
               p.high_winrate_gate, p.prediction_correct, p.actual_return
        FROM predictions p
        JOIN first_rows f ON p.id = f.id
        ORDER BY p.open_time
        """,
        (symbol.upper(), duration),
    ).fetchall()
    return [dict(row) for row in rows]


def _stats(rows: list[dict[str, Any]], fixed_days: int | None) -> dict[str, Any]:
    trades = len(rows)
    if trades == 0:
        return {"trades": 0, "wins": 0, "winRate": None, "tradesPerDay": 0.0, "avgReturn": None}
    wins = sum(int(row["prediction_correct"]) for row in rows)
    avg_return = sum(float(row["actual_return"]) for row in rows) / trades
    days = float(fixed_days) if fixed_days else _span_days(rows)
    return {
        "trades": trades,
        "wins": wins,
        "winRate": wins / trades,
        "tradesPerDay": trades / days,
        "avgReturn": avg_return,
    }


def _per_rule_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rule = row["high_winrate_rule"] or row["high_winrate_gate"] or "unknown"
        grouped.setdefault(rule, []).append(row)
    return [{"rule": key, **_stats(value, None)} for key, value in sorted(grouped.items())]


def _overfit_monitor(gated: dict[str, Any], windows: dict[str, dict[str, Any]], backtest: Any) -> dict[str, Any]:
    sample = windows["7d"] if windows["7d"]["trades"] >= MIN_MONITOR_TRADES else gated
    if sample["trades"] < MIN_MONITOR_TRADES:
        return {"status": "collecting", "sampleTrades": sample["trades"], "minimumTrades": MIN_MONITOR_TRADES}
    warnings = _monitor_warnings(sample, None if backtest is None else float(backtest))
    return {
        "status": "warn" if warnings else "ok",
        "sampleTrades": sample["trades"],
        "minimumTrades": MIN_MONITOR_TRADES,
        "warnings": warnings,
    }


def _monitor_warnings(sample: dict[str, Any], backtest_win_rate: float | None) -> list[str]:
    warnings = []
    if sample["winRate"] is not None and sample["winRate"] < TARGET_WIN_RATE:
        warnings.append("forward_win_rate_below_target")
    if backtest_win_rate is not None and backtest_win_rate - float(sample["winRate"]) > OVERFIT_DROP_LIMIT:
        warnings.append("forward_backtest_gap_too_wide")
    return warnings


def _window_rows(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    cutoff = int(datetime.now(timezone.utc).timestamp() * 1000) - days * DAY_MS
    return [row for row in rows if int(row["open_time"]) >= cutoff]


def _span_days(rows: list[dict[str, Any]]) -> float:
    start = min(int(row["open_time"]) for row in rows)
    end = max(int(row["open_time"]) for row in rows)
    return max((end - start + 1) / DAY_MS, 1 / 1440)


def _kline_close(conn, symbol: str, open_time: int) -> float | None:
    row = conn.execute(
        """
        SELECT close FROM klines
        WHERE symbol = ? AND interval = '1m' AND open_time >= ?
        ORDER BY open_time
        LIMIT 1
        """,
        (symbol.upper(), open_time),
    ).fetchone()
    return None if row is None else float(row["close"])


def _directional_return(entry_price: float, exit_price: float, direction: str) -> float:
    gross = exit_price / entry_price - 1.0
    signed = gross if direction == "up" else -gross
    return signed - ROUNDTRIP_COST_RATE


def _direction_correct(entry_price: float, exit_price: float, direction: str) -> bool:
    if direction == "up":
        return exit_price > entry_price
    return exit_price < entry_price


def _duration_ms(duration: str) -> int:
    if duration not in DURATION_MINUTES:
        raise ValueError(f"unsupported forward validation duration: {duration}")
    return DURATION_MINUTES[duration] * 60_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
