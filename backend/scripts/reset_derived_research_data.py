from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import get_conn  # noqa: E402

DERIVED_TABLES = (
    "settlements",
    "orders",
    "events",
    "predictions",
    "ensemble_stage_status",
    "ensemble_signal_scores",
    "paper_live_candidate_status",
    "paper_live_candidate_status_history",
    "paper_live_prediction_failures",
    "paper_live_prediction_stage_log",
    "factor_ranking_cache",
    "factor_combo_ranking_cache",
    "factor_combo_signal_cache",
    "factor_combo_backtest_cache",
    "factor_combo_feature_snapshots",
    "high_winrate_combo_ranking_cache",
    "high_winrate_strategy_status",
    "event_market_regimes",
    "event_final_decisions",
)
RESET_SETTINGS_SQL = """
UPDATE auto_trade_settings
SET enabled = 0, live_trading_enabled = 0, updated_at = datetime('now')
WHERE id = 1
"""
MODEL_ARTIFACT_DIRS = (
    ROOT / "models" / "ml",
    ROOT / "models" / "lstm",
    ROOT / "models" / "factor_learning",
)


def main() -> None:
    args = _parse_args()
    report = reset_derived_research_data(confirm=args.confirm)
    _print_report(report)


def reset_derived_research_data(*, confirm: bool = False) -> dict[str, Any]:
    conn = get_conn()
    try:
        tables = _table_counts(conn)
        artifact_dirs = _artifact_dirs()
        if confirm:
            _delete_tables(conn, tables)
            _reset_strategy_slots(conn)
            conn.commit()
            _delete_artifacts(artifact_dirs)
        return {"mode": "delete" if confirm else "dry-run", "tables": tables, "artifactDirs": artifact_dirs}
    finally:
        conn.close()


def _table_counts(conn: Any) -> list[dict[str, Any]]:
    rows = []
    for table in DERIVED_TABLES:
        if not _table_exists(conn, table):
            rows.append({"table": table, "exists": False, "rows": 0})
            continue
        count = conn.execute(f"SELECT COUNT(*) AS value FROM {table}").fetchone()["value"]
        rows.append({"table": table, "exists": True, "rows": int(count)})
    if _table_exists(conn, "auto_trade_strategies"):
        count = conn.execute("SELECT COUNT(*) AS value FROM auto_trade_strategies").fetchone()["value"]
        rows.append({"table": "auto_trade_strategies", "exists": True, "rows": int(count), "action": "delete"})
    return rows


def _delete_tables(conn: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        table = str(row["table"])
        if not row.get("exists"):
            continue
        if table == "auto_trade_strategies":
            conn.execute("DELETE FROM auto_trade_strategies")
            continue
        conn.execute(f"DELETE FROM {table}")


def _reset_strategy_slots(conn: Any) -> None:
    if _table_exists(conn, "auto_trade_settings"):
        conn.execute(RESET_SETTINGS_SQL)


def _artifact_dirs() -> list[dict[str, Any]]:
    return [
        {"path": str(path), "exists": path.exists()}
        for path in MODEL_ARTIFACT_DIRS
        if _inside_backend_models(path)
    ]


def _delete_artifacts(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = Path(str(row["path"]))
        if row["exists"] and _inside_backend_models(path):
            shutil.rmtree(path)


def _inside_backend_models(path: Path) -> bool:
    models = (ROOT / "models").resolve()
    return path.resolve().is_relative_to(models)


def _table_exists(conn: Any, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _print_report(report: dict[str, Any]) -> None:
    print(f"mode: {report['mode']}")
    for row in report["tables"]:
        print(f"table {row['table']}: exists={row['exists']} rows={row['rows']}")
    for row in report["artifactDirs"]:
        print(f"artifact {row['path']}: exists={row['exists']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset derived event-contract research data.")
    parser.add_argument("--confirm", action="store_true", help="delete derived rows and old model artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    main()
