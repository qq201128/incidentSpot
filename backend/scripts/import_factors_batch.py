"""
批量导入因子组合（带重试和WAL模式）
"""
import sqlite3
import json
import time

def import_factors_batch():
    """批量导入因子"""

    # 读取导出的数据
    print('Loading exported factors...')
    with open('factor_combinations_export.json', 'r', encoding='utf-8') as f:
        factors = json.load(f)

    print(f'Loaded {len(factors)} factors')

    # 重试导入
    max_retries = 10
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('backend/data.db', timeout=60)

            # 启用WAL模式（允许并发读写）
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=60000')  # 60秒超时

            print(f'\nAttempt {attempt + 1}/{max_retries}')

            inserted = 0
            skipped = 0
            failed = 0

            # 批量插入（每100条提交一次）
            batch_size = 100

            for i, factor in enumerate(factors):
                try:
                    name = factor['name']
                    symbol = factor['symbol']
                    duration = factor['duration']

                    # 检查是否已存在
                    existing = conn.execute(
                        'SELECT id FROM factor_combinations WHERE name = ? AND symbol = ? AND duration = ?',
                        (name, symbol, duration)
                    ).fetchone()

                    if existing:
                        skipped += 1
                        continue

                    # 插入
                    members_json = json.dumps(factor['members'])

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
                        factor.get('ir'),
                        factor.get('win_rate'),
                        factor.get('sharpe'),
                        factor.get('max_drawdown'),
                        factor.get('trades', 0)
                    ))

                    inserted += 1

                    # 定期提交
                    if (i + 1) % batch_size == 0:
                        conn.commit()
                        print(f'  Progress: {i + 1}/{len(factors)} ({inserted} inserted, {skipped} skipped)')

                except Exception as e:
                    failed += 1
                    if failed < 10:  # 只显示前10个错误
                        print(f'  ERROR inserting {factor.get("name", "unknown")}: {e}')

            # 最终提交
            conn.commit()

            # 统计
            total = conn.execute('SELECT COUNT(*) FROM factor_combinations').fetchone()[0]
            with_icir = conn.execute('SELECT COUNT(*) FROM factor_combinations WHERE icir IS NOT NULL').fetchone()[0]
            with_win_rate = conn.execute('SELECT COUNT(*) FROM factor_combinations WHERE win_rate IS NOT NULL').fetchone()[0]

            conn.close()

            print(f'\n=== Import Success ===')
            print(f'  Inserted: {inserted}')
            print(f'  Skipped: {skipped}')
            print(f'  Failed: {failed}')
            print(f'  Total in DB: {total}')
            print(f'  With ICIR: {with_icir}')
            print(f'  With Win Rate: {with_win_rate}')

            return True

        except sqlite3.OperationalError as e:
            if 'locked' in str(e).lower() and attempt < max_retries - 1:
                print(f'Database locked, waiting {retry_delay}s...')
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 30)  # 指数退避，最多30秒
            else:
                print(f'Failed after {attempt + 1} attempts: {e}')
                return False

        except Exception as e:
            print(f'Unexpected error: {e}')
            import traceback
            traceback.print_exc()
            return False

    return False

if __name__ == '__main__':
    success = import_factors_batch()
    if success:
        print('\n✓ Import completed successfully!')
    else:
        print('\n✗ Import failed. Please stop the backend service and try again.')
