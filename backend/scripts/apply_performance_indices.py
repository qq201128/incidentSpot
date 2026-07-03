#!/usr/bin/env python3
"""
Apply performance optimization indices to the database.
Run this script to add new indices for high-frequency queries.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data.db"

# New performance indices from schema.sql optimization
# Adjusted based on actual schema (no 'settled' column in events)
PERFORMANCE_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_events_status_symbol ON events(status, symbol, event_interval)",
    "CREATE INDEX IF NOT EXISTS idx_klines_query_optimized ON klines(symbol, interval, open_time DESC)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_strategy_symbol ON predictions(strategy_key, symbol, duration, open_time DESC)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_execution_status ON predictions(execution_status, symbol, duration, execution_checked_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_prediction_lookup ON events(prediction_open_time, symbol, event_interval) WHERE prediction_open_time IS NOT NULL",
]


def main():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH, timeout=30.0)

    try:
        print("Applying performance indices...")
        for i, sql in enumerate(PERFORMANCE_INDICES, 1):
            print(f"  [{i}/{len(PERFORMANCE_INDICES)}] {sql[:80]}...")
            conn.execute(sql)

        conn.commit()
        print(f"\nSuccessfully applied {len(PERFORMANCE_INDICES)} performance indices")

        # Verify indices were created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND (name LIKE 'idx_%_optimized' OR name LIKE 'idx_%_strategy_symbol' OR name LIKE 'idx_%_execution_status')"
        )
        created = [row[0] for row in cursor.fetchall()]
        if created:
            print(f"\nCreated indices: {', '.join(created)}")

    except sqlite3.Error as exc:
        print(f"\nDatabase error: {exc}")
        conn.rollback()
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
