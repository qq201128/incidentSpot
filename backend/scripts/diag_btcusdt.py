"""One-off diagnostic: BTCUSDT klines, ranking cache, factor frame readiness."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_cache_metadata import cache_is_usable_for_live_signal
from app.services.factor_candidate_signal_service import (
    CANDIDATE_SCORE_LOOKBACK_BARS,
    MIN_DURATION_KLINE_ROWS,
)
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.kline_timing import current_rule_entry_open_time_for_duration

DB = BACKEND / "data.db"
SYM = "BTCUSDT"
MIN_KLINE = MIN_DURATION_KLINE_ROWS
DURATIONS = ("10m", "30m", "60m", "1d")
INTERVALS_EXTRA = ("1m", "5m", "15m", "1h", "4h")
SAMPLE_FACTORS = ("spread_bps", "imbalance", "funding_rate", "tf_5m_ret_5", "close")


def ts(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> None:
    print("=== DB:", DB.resolve(), "exists:", DB.exists(), "===")
    if not DB.exists():
        raise SystemExit("no data.db")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    print("\n--- Klines (BTCUSDT) ---")
    rows = conn.execute(
        """
        SELECT interval, COUNT(*) AS c, MIN(open_time) AS mn, MAX(open_time) AS mx
        FROM klines WHERE symbol = ? GROUP BY interval ORDER BY interval
        """,
        (SYM,),
    ).fetchall()
    by_iv = {r["interval"]: r for r in rows}
    for iv in sorted(set(DURATIONS + INTERVALS_EXTRA + tuple(by_iv))):
        r = by_iv.get(iv)
        if r:
            if r["c"] >= MIN_KLINE:
                flag = "OK"
            elif r["c"] >= BACKTEST_MIN_PERIODS:
                flag = "LOW"
            else:
                flag = "CRIT"
            print(
                f"  {iv:6} count={r['c']:5}  need>={MIN_KLINE} [{flag}]"
                f"  oldest={ts(r['mn'])}  newest={ts(r['mx'])}"
            )
        else:
            print(f"  {iv:6} count=    0  need>={MIN_KLINE} [MISSING]")

    print("\n--- Market context feature rows (BTCUSDT) ---")
    for table in (
        "orderbook_features",
        "funding_features",
        "futures_positioning_features",
        "onchain_features",
    ):
        r = conn.execute(
            f"SELECT COUNT(*) AS c, MIN(open_time) AS mn, MAX(open_time) AS mx"
            f" FROM {table} WHERE symbol = ?",
            (SYM,),
        ).fetchone()
        print(
            f"  {table:32} count={r['c']:5}"
            f"  range={ts(r['mn'])} .. {ts(r['mx'])}"
        )
    r = conn.execute("SELECT COUNT(*) AS c FROM market_sentiment_features").fetchone()
    print(f"  {'market_sentiment_features':32} count={r['c']:5}")

    print("\n--- Factor ranking cache (BTCUSDT) ---")
    for dur in DURATIONS:
        row = conn.execute(
            """
            SELECT total, updated_at, payload
            FROM factor_ranking_cache
            WHERE symbol = ? AND duration = ?
            """,
            (SYM, dur),
        ).fetchone()
        if not row:
            print(f"  {dur}: MISSING (not preheated)")
            continue
        payload = json.loads(row["payload"])
        ranking = payload.get("ranking") if isinstance(payload, dict) else payload
        meta = payload.get("cacheMeta") if isinstance(payload, dict) else None
        n = len(ranking) if isinstance(ranking, list) else 0
        md = (meta or {}).get("marketData") or {}
        print(
            f"  {dur}: total={row['total']} ranking_len={n}"
            f" updated={row['updated_at']}"
        )
        print(f"       cacheMeta.marketData={md}")

    print("\n--- App-level cache usability ---")
    print(
        f"  CANDIDATE_SCORE_LOOKBACK_BARS={CANDIDATE_SCORE_LOOKBACK_BARS}"
        f" MIN_DURATION_KLINE_ROWS={MIN_DURATION_KLINE_ROWS}"
        f" BACKTEST_MIN_PERIODS={BACKTEST_MIN_PERIODS}"
    )
    for dur in DURATIONS:
        cache = get_cached_ranking(SYM, dur)
        usable = cache_is_usable_for_live_signal(cache) if cache else False
        status = (cache or {}).get("cacheStatus") or {}
        reason = status.get("reason") or status.get("usable")
        print(
            f"  {dur}: cache_exists={cache is not None}"
            f" live_usable={usable} status={reason}"
        )

    print("\n--- Factor frame score readiness (sample factors) ---")
    for dur in DURATIONS:
        try:
            frame = load_factor_frame(SYM, dur, min_history=CANDIDATE_SCORE_LOOKBACK_BARS)
            entry = current_rule_entry_open_time_for_duration(dur)
            print(f"  {dur}: frame_rows={len(frame)} cols={len(frame.columns)} entry={entry}")
            for name in SAMPLE_FACTORS:
                if name not in frame.columns:
                    print(f"       {name}: NO_COLUMN")
                    continue
                s = frame[name]
                print(
                    f"       {name}: non_null={int(s.notna().sum())}/{len(s)}"
                    f" last5_valid={int(s.tail(5).notna().sum())}"
                )
        except Exception as exc:
            print(f"  {dur}: FRAME_ERROR {type(exc).__name__}: {exc}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
