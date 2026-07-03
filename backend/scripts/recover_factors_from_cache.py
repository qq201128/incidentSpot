"""
从缓存恢复组合因子到主表
"""
import sqlite3
import json
import time

def recover_factors_from_cache():
    """从缓存恢复组合因子"""

    # 重试机制
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('backend/data.db', timeout=30)
            conn.execute('PRAGMA journal_mode=WAL')  # 使用WAL模式避免锁

            print('Recovering factor combinations from cache...')

            caches = conn.execute('SELECT symbol, duration, payload FROM factor_combo_ranking_cache').fetchall()

            inserted = 0
            skipped = 0

            for cache in caches:
                symbol, duration, payload_json = cache

                try:
                    payload = json.loads(payload_json)
                    ranking = payload.get('ranking', [])

                    print(f'Processing {symbol}@{duration}: {len(ranking)} combos')

                    for combo in ranking:
                        # 提取字段
                        name = combo.get('factorName')
                        display_name = combo.get('factorDisplayName', '')
                        formula = combo.get('formula', '')
                        members_data = combo.get('members', [])
                        members_json = json.dumps(members_data)

                        # 指标
                        ir = combo.get('ir')
                        win_rate = combo.get('winRate')
                        sharpe = combo.get('sharpe')
                        max_drawdown = combo.get('maxDrawdown')
                        total_periods = combo.get('totalPeriods', 0)

                        if not name:
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
                        conn.execute('''
                            INSERT INTO factor_combinations (
                                name, formula, members, symbol, duration,
                                backtest_completed, icir, win_rate, sharpe, max_drawdown, trades,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                        ''', (name, formula, members_json, symbol, duration, ir, win_rate, sharpe, max_drawdown, total_periods))

                        inserted += 1

                except Exception as e:
                    print(f'  Failed to process {symbol}@{duration}: {e}')

            conn.commit()

            # 统计
            total = conn.execute('SELECT COUNT(*) FROM factor_combinations').fetchone()[0]
            with_icir = conn.execute('SELECT COUNT(*) FROM factor_combinations WHERE icir IS NOT NULL').fetchone()[0]
            with_win_rate = conn.execute('SELECT COUNT(*) FROM factor_combinations WHERE win_rate IS NOT NULL').fetchone()[0]

            print(f'\nCompleted:')
            print(f'  Inserted: {inserted}')
            print(f'  Skipped: {skipped}')
            print(f'  Total: {total}')
            print(f'  With ICIR: {with_icir}')
            print(f'  With Win Rate: {with_win_rate}')

            conn.close()
            return True

        except sqlite3.OperationalError as e:
            if 'locked' in str(e) and attempt < max_retries - 1:
                print(f'Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})')
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f'Failed after {attempt + 1} attempts: {e}')
                return False
        except Exception as e:
            print(f'Unexpected error: {e}')
            return False

if __name__ == '__main__':
    recover_factors_from_cache()
