"""
因子组合管理API
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.session import get_conn

router = APIRouter(prefix="/api/factors/library", tags=["factors"])


@router.post("/batch-import")
def batch_import_factors(factors: list[dict]) -> dict:
    """
    批量导入因子组合

    Request body:
    [
        {
            "name": "combo_xxx",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "formula": "...",
            "members": [...],
            "icir": 0.5,
            "win_rate": 0.58,
            "sharpe": 95.0,
            "max_drawdown": -0.02,
            "trades": 620
        },
        ...
    ]
    """
    import json

    conn = get_conn()

    inserted = 0
    skipped = 0
    failed = 0

    try:
        for factor in factors:
            try:
                name = factor.get('name')
                symbol = factor.get('symbol')
                duration = factor.get('duration')

                if not name or not symbol or not duration:
                    failed += 1
                    continue

                # 检查是否已存在
                existing = conn.execute(
                    'SELECT id FROM factor_combinations WHERE name = ? AND symbol = ? AND duration = ?',
                    (name, symbol, duration)
                ).fetchone()

                if existing:
                    skipped += 1
                    continue

                # 插入
                members_json = json.dumps(factor.get('members', []))

                conn.execute('''
                    INSERT INTO factor_combinations (
                        name, formula, members, symbol, duration,
                        backtest_completed, icir, win_rate, sharpe, max_drawdown, trades,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ''', (
                    name,
                    factor.get('formula', ''),
                    members_json,
                    symbol,
                    duration,
                    factor.get('ir') or factor.get('icir'),
                    factor.get('win_rate'),
                    factor.get('sharpe'),
                    factor.get('max_drawdown'),
                    factor.get('trades', 0)
                ))

                inserted += 1

            except Exception as e:
                failed += 1

        conn.commit()

        # 统计
        total = conn.execute('SELECT COUNT(*) FROM factor_combinations').fetchone()[0]

        return {
            "status": "success",
            "inserted": inserted,
            "skipped": skipped,
            "failed": failed,
            "total": total,
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()


@router.get("/stats")
def get_factor_stats() -> dict:
    """获取因子库统计"""
    conn = get_conn()

    try:
        result = conn.execute('''
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN icir IS NOT NULL THEN 1 END) as with_icir,
                COUNT(CASE WHEN win_rate IS NOT NULL THEN 1 END) as with_win_rate,
                COUNT(CASE WHEN trades > 0 THEN 1 END) as with_trades
            FROM factor_combinations
        ''').fetchone()

        by_symbol = conn.execute('''
            SELECT symbol, duration, COUNT(*) as count
            FROM factor_combinations
            GROUP BY symbol, duration
            ORDER BY symbol, duration
        ''').fetchall()

        return {
            "total": result['total'],
            "with_icir": result['with_icir'],
            "with_win_rate": result['with_win_rate'],
            "with_trades": result['with_trades'],
            "by_symbol": [
                {"symbol": row['symbol'], "duration": row['duration'], "count": row['count']}
                for row in by_symbol
            ]
        }

    finally:
        conn.close()
