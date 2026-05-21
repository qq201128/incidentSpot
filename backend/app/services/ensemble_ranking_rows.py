from __future__ import annotations

from typing import Any

from app.services.ensemble_judge_constants import ENSEMBLE_RANKER_STRATEGY_KEY
from app.services.ensemble_judge_metrics import unscored_candidate_row


def ensemble_candidate_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in scored_rows(conn, symbol, duration)]
    _attach_signal_labels(conn, symbol, duration, rows)
    scored_keys = {str(row["signal_key"]) for row in rows}
    rows.extend(_unscored_signal_rows(conn, symbol, duration, scored_keys))
    return rows


def scored_rows(conn: Any, symbol: str, duration: str) -> list[Any]:
    return conn.execute(
        """
        SELECT s.*,
               COALESCE(labels.signal_label, s.signal_key) AS signal_label
        FROM ensemble_signal_scores s
        LEFT JOIN (
          SELECT signal_key,
                 MAX(COALESCE(NULLIF(high_winrate_rule, ''), NULLIF(model_version, ''), signal_key)) AS signal_label
          FROM predictions
          WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
            AND signal_key != ?
          GROUP BY signal_key
        ) labels ON labels.signal_key = s.signal_key
        WHERE s.symbol = ? AND s.duration = ?
        """,
        (symbol, duration, ENSEMBLE_RANKER_STRATEGY_KEY, symbol, duration),
    ).fetchall()


def _attach_signal_labels(conn: Any, symbol: str, duration: str, rows: list[dict[str, Any]]) -> None:
    labels = _signal_label_rows(conn, symbol, duration)
    mapping = {str(row["signal_key"]): str(row["signal_label"] or row["signal_key"]) for row in labels}
    for row in rows:
        row["signal_label"] = mapping.get(str(row["signal_key"]), str(row["signal_key"]))


def _unscored_signal_rows(
    conn: Any,
    symbol: str,
    duration: str,
    scored_keys: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for row in _pending_signal_rows(conn, symbol, duration):
        if str(row["signal_key"]) not in scored_keys:
            rows.append(unscored_candidate_row(dict(row)))
    return rows


def _signal_label_rows(conn: Any, symbol: str, duration: str) -> list[Any]:
    return conn.execute(
        """
        SELECT signal_key,
               MAX(COALESCE(NULLIF(high_winrate_rule, ''), NULLIF(model_version, ''), signal_key)) AS signal_label
        FROM predictions
        WHERE symbol = ? AND duration = ?
          AND signal_key != ?
        GROUP BY signal_key
        """,
        (symbol, duration, ENSEMBLE_RANKER_STRATEGY_KEY),
    ).fetchall()


def _pending_signal_rows(conn: Any, symbol: str, duration: str) -> list[Any]:
    return conn.execute(
        """
        SELECT signal_key,
               MAX(COALESCE(NULLIF(high_winrate_rule, ''), NULLIF(model_version, ''), signal_key)) AS signal_label,
               MAX(high_winrate_rule) AS high_winrate_rule,
               MAX(model_version) AS model_version,
               COUNT(*) AS pending_count
        FROM predictions
        WHERE symbol = ? AND duration = ? AND settled_at IS NULL
          AND signal_key != ?
        GROUP BY signal_key
        """,
        (symbol, duration, ENSEMBLE_RANKER_STRATEGY_KEY),
    ).fetchall()
