"""Diagnostic: settled shadow predictions per model family and ensemble judge thresholds."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

DB = BACKEND / "data.db"
MODEL_FAMILIES = (
    "lstm",
    "gru",
    "cnn",
    "transformer",
    "random_forest",
    "xgboost",
    "svm",
    "bayesian",
    "knn",
    "rl_strategy",
)
MAJOR_TYPES = ("factor_combo", "high_winrate_combo", "model_family", "factor_candidate")


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    print_event_stats(conn)

    total = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    settled = conn.execute(
        "SELECT COUNT(*) c FROM predictions WHERE settled_at IS NOT NULL"
    ).fetchone()["c"]
    pending = conn.execute(
        "SELECT COUNT(*) c FROM predictions WHERE settled_at IS NULL"
    ).fetchone()["c"]

    print("=== 预测总览 ===")
    print(f"predictions 总数: {total}")
    print(f"已结算: {settled}")
    print(f"待结算: {pending}")
    print()

    rows = conn.execute(
        """
        SELECT
            CASE
                WHEN signal_key LIKE 'factor_lstm_shadow_%' THEN 'lstm'
                WHEN signal_key LIKE 'factor_gru_shadow_%' THEN 'gru'
                WHEN signal_key LIKE 'factor_cnn_shadow_%' THEN 'cnn'
                WHEN signal_key LIKE 'factor_transformer_shadow_%' THEN 'transformer'
                WHEN signal_key LIKE 'factor_random_forest_shadow_%' THEN 'random_forest'
                WHEN signal_key LIKE 'factor_xgboost_shadow_%' THEN 'xgboost'
                WHEN signal_key LIKE 'factor_svm_shadow_%' THEN 'svm'
                WHEN signal_key LIKE 'factor_bayesian_shadow_%' THEN 'bayesian'
                WHEN signal_key LIKE 'factor_knn_shadow_%' THEN 'knn'
                WHEN signal_key LIKE 'factor_rl_strategy_shadow_%' THEN 'rl_strategy'
                ELSE 'other'
            END AS family,
            symbol,
            duration,
            COUNT(*) AS settled,
            SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) AS wins
        FROM predictions
        WHERE settled_at IS NOT NULL
          AND signal_key LIKE 'factor_%_shadow_%'
        GROUP BY family, symbol, duration
        ORDER BY family, symbol, duration
        """
    ).fetchall()

    family_totals = {f: {"settled": 0, "wins": 0} for f in MODEL_FAMILIES}
    print("=== 各模型族 shadow 已结算 (按 symbol / duration) ===")
    if not rows:
        print("  (无 model family shadow 已结算记录)")
    for row in rows:
        fam = row["family"]
        if fam in family_totals:
            family_totals[fam]["settled"] += row["settled"]
            family_totals[fam]["wins"] += row["wins"] or 0
        wr = round(100.0 * (row["wins"] or 0) / row["settled"], 1) if row["settled"] else 0
        print(
            f"  {fam:16} {row['symbol']:8} {row['duration']:4}  "
            f"settled={row['settled']:4}  wins={row['wins'] or 0:4}  win_rate={wr}%"
        )

    print()
    print("=== 各模型族汇总 ===")
    for fam in MODEL_FAMILIES:
        total_settled = family_totals[fam]["settled"]
        wins = family_totals[fam]["wins"]
        wr = round(100.0 * wins / total_settled, 1) if total_settled else None
        gate = _gate_label(total_settled)
        wr_text = f"{wr}%" if wr is not None else "N/A"
        print(f"  {fam:16} settled={total_settled:5}  win_rate={wr_text:>6}  [{gate}]")

    print()
    print("=== signal_key 明细 TOP 30 (已结算 shadow) ===")
    detail = conn.execute(
        """
        SELECT signal_key, symbol, duration, COUNT(*) settled,
               SUM(CASE WHEN prediction_correct=1 THEN 1 ELSE 0 END) wins
        FROM predictions
        WHERE settled_at IS NOT NULL AND signal_key LIKE 'factor_%_shadow_%'
        GROUP BY signal_key, symbol, duration
        HAVING settled > 0
        ORDER BY settled DESC
        LIMIT 30
        """
    ).fetchall()
    if not detail:
        print("  (无记录)")
    for row in detail:
        print(
            f"  {row['signal_key']:35} {row['symbol']} {row['duration']}  "
            f"n={row['settled']} wins={row['wins']}"
        )

    print()
    print("=== ensemble_stage_status ===")
    stages = conn.execute(
        "SELECT * FROM ensemble_stage_status ORDER BY symbol, duration"
    ).fetchall()
    if not stages:
        print("  (无记录 — 可能尚未执行 ensemble refresh)")
    for row in stages:
        print(
            f"  {row['symbol']} {row['duration']}: stage={row['stage']} "
            f"recommended={row['recommended_stage']} reason={row['recommendation_reason']}"
        )

    print()
    print("=== ensemble 覆盖 (byMajorSignalType) ===")
    from app.services.ensemble_judge_constants import (
        ENSEMBLE_MIN_SETTLED_SAMPLES,
        ENSEMBLE_READY_SAMPLE_COUNT,
        LOSS_STREAK_THRESHOLD,
        WEIGHT_READY_MIN_DAYS,
        WEIGHT_READY_SAMPLE_COUNT,
    )
    from app.services.ensemble_judge_service import ensemble_status

    combos = conn.execute(
        """
        SELECT DISTINCT symbol, duration
        FROM predictions
        WHERE settled_at IS NOT NULL
        ORDER BY symbol, duration
        """
    ).fetchall()
    if not combos:
        print("  (无已结算预测，无法计算覆盖)")
    for combo in combos:
        sym, dur = combo["symbol"], combo["duration"]
        status = ensemble_status(sym, dur)
        coverage = status.get("sampleCoverage") or {}
        major = coverage.get("byMajorSignalType") or {}
        ready_count = coverage.get("readySignalTypeCount", 0)
        required = coverage.get("requiredSignalTypeCount", 4)
        print(f"\n  {sym} / {dur}")
        print(f"    stage={status.get('stage')}  recommended={status.get('recommendedStage')}")
        print(f"    reason={status.get('recommendationReason')}")
        print(f"    readySignalTypes={ready_count}/{required}")
        for kind in MAJOR_TYPES:
            item = major.get(kind) or {}
            sample_count = int(item.get("sampleCount") or 0)
            days = int(item.get("distinctTradingDays") or 0)
            losses = int(item.get("maxConsecutiveLosses") or 0)
            pf_bad = bool(item.get("recentProfitFactorBelowOne"))
            blockers = _weight_ready_blockers(
                sample_count,
                days,
                losses,
                pf_bad,
                WEIGHT_READY_SAMPLE_COUNT,
                WEIGHT_READY_MIN_DAYS,
                LOSS_STREAK_THRESHOLD,
            )
            print(
                f"    {kind:22} samples={sample_count:4} days={days:3} "
                f"lossStreak={losses} pfBelow1={pf_bad}  "
                f"[{_major_gate(sample_count, WEIGHT_READY_SAMPLE_COUNT, ENSEMBLE_READY_SAMPLE_COUNT)}]"
            )
            if blockers:
                print(f"      blockers: {', '.join(blockers)}")

    print()
    print("=== model_family 单信号评分 TOP (BTCUSDT 10m) ===")
    scores = conn.execute(
        """
        SELECT signal_key, sample_count, win_rate, profit_factor, score, weight_suggestion
        FROM ensemble_signal_scores
        WHERE symbol = 'BTCUSDT' AND duration = '10m' AND signal_type = 'model_family'
        ORDER BY sample_count DESC
        """
    ).fetchall()
    if not scores:
        print("  (无评分)")
    for row in scores:
        print(
            f"  {row['signal_key']:35} n={row['sample_count']:4} "
            f"wr={row['win_rate']:.1%} pf={row['profit_factor']:.2f} "
            f"score={row['score']:.1f} weight={row['weight_suggestion']:.1f}"
        )

    conn.close()


def _weight_ready_blockers(
    sample_count: int,
    days: int,
    losses: int,
    pf_bad: bool,
    weight_ready: int,
    min_days: int,
    loss_threshold: int,
) -> list[str]:
    blockers: list[str] = []
    if sample_count < weight_ready:
        blockers.append(f"samples need {weight_ready - sample_count} more (target {weight_ready})")
    if days < min_days:
        blockers.append(f"days need {min_days - days} more (target {min_days})")
    if pf_bad:
        blockers.append("profit_factor < 1")
    if losses >= loss_threshold:
        blockers.append(f"consecutive_losses>={loss_threshold}")
    return blockers


def _gate_label(count: int) -> str:
    if count >= 500:
        return ">=500 ensemble_ready"
    if count >= 200:
        return ">=200 weight_ready"
    if count >= 100:
        return ">=100 可评估"
    if count >= 50:
        return ">=50 低样本"
    return f"<50 还差 {50 - count} 到 low_sample"


def _major_gate(count: int, weight_ready: int, ensemble_ready: int) -> str:
    if count >= ensemble_ready:
        return f">={ensemble_ready} ensemble_ready OK"
    if count >= weight_ready:
        return f">={weight_ready} weight_ready OK"
    if count >= 100:
        return ">=100 min_eval OK"
    return f"need {max(100 - count, 0)} more for min_eval"


def print_event_stats(conn: sqlite3.Connection) -> None:
    print()
    print("=== 事件合约 (events) 实盘/模拟样本 ===")
    total = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
    settled = conn.execute("SELECT COUNT(*) c FROM events WHERE status='SETTLED'").fetchone()["c"]
    with_ai = conn.execute(
        "SELECT COUNT(*) c FROM events WHERE status='SETTLED' AND ai_predicted_direction IS NOT NULL"
    ).fetchone()["c"]
    with_order = conn.execute(
        """
        SELECT COUNT(DISTINCT e.id) c FROM events e
        JOIN orders o ON o.event_id = e.id WHERE e.status='SETTLED'
        """
    ).fetchone()["c"]
    live = conn.execute(
        """
        SELECT COUNT(*) c FROM orders o
        JOIN events e ON e.id = o.event_id
        WHERE o.external_order_id IS NOT NULL AND TRIM(o.external_order_id) != ''
        """
    ).fetchone()["c"]
    print(f"  events 总数={total}  已结算={settled}  带AI预测={with_ai}  有订单={with_order}  真仓external_order={live}")
    rows = conn.execute(
        """
        SELECT strategy_key, COUNT(*) total,
               SUM(CASE WHEN status='SETTLED' THEN 1 ELSE 0 END) settled,
               SUM(CASE WHEN ai_prediction_correct=1 THEN 1 ELSE 0 END) hits
        FROM events GROUP BY strategy_key ORDER BY settled DESC LIMIT 12
        """
    ).fetchall()
    for row in rows:
        settled_n = row["settled"] or 0
        hits = row["hits"] or 0
        rate = f"{100.0 * hits / settled_n:.1f}%" if settled_n else "N/A"
        print(f"  {row['strategy_key']:40} total={row['total']:4} settled={settled_n:4} ai_hits={hits:4} rate={rate}")


if __name__ == "__main__":
    main()
