from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import reset_derived_research_data as reset_script  # noqa: E402


def test_reset_derived_research_data_dry_run_keeps_rows(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "reset.db"
    _init_db(db_path)
    monkeypatch.setattr(reset_script, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(reset_script, "MODEL_ARTIFACT_DIRS", (tmp_path / "models" / "ml",))

    report = reset_script.reset_derived_research_data(confirm=False)

    assert report["mode"] == "dry-run"
    assert _count(db_path, "predictions") == 1


def test_reset_derived_research_data_confirm_deletes_derived_only(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "reset-confirm.db"
    model_dir = tmp_path / "models" / "ml"
    model_dir.mkdir(parents=True)
    (model_dir / "old.txt").write_text("old", encoding="utf-8")
    _init_db(db_path)
    monkeypatch.setattr(reset_script, "ROOT", tmp_path)
    monkeypatch.setattr(reset_script, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(reset_script, "MODEL_ARTIFACT_DIRS", (model_dir,))

    report = reset_script.reset_derived_research_data(confirm=True)

    assert report["mode"] == "delete"
    assert _count(db_path, "predictions") == 0
    assert _count(db_path, "klines") == 1
    assert not model_dir.exists()


def _init_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE klines(id INTEGER);
        CREATE TABLE predictions(id INTEGER);
        CREATE TABLE events(id INTEGER);
        CREATE TABLE orders(id INTEGER);
        CREATE TABLE settlements(id INTEGER);
        CREATE TABLE auto_trade_settings(id INTEGER, enabled INTEGER, live_trading_enabled INTEGER, updated_at TEXT);
        CREATE TABLE auto_trade_strategies(strategy_key TEXT);
        INSERT INTO klines VALUES(1);
        INSERT INTO predictions VALUES(1);
        INSERT INTO events VALUES(1);
        INSERT INTO orders VALUES(1);
        INSERT INTO settlements VALUES(1);
        INSERT INTO auto_trade_settings VALUES(1, 1, 1, 'old');
        INSERT INTO auto_trade_strategies VALUES('old');
        """
    )
    conn.commit()
    conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _count(path: Path, table: str) -> int:
    conn = _connect(path)
    value = int(conn.execute(f"SELECT COUNT(*) AS value FROM {table}").fetchone()["value"])
    conn.close()
    return value
