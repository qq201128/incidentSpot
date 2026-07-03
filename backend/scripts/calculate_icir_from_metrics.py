"""
从胜率和夏普比率计算ICIR
"""
import sqlite3

def calculate_icir_from_metrics():
    """从现有指标计算ICIR"""

    conn = sqlite3.connect('backend/data.db', timeout=60)

    print('Calculating ICIR from existing metrics...')

    # 获取所有因子
    factors = conn.execute('''
        SELECT id, name, win_rate, sharpe, trades
        FROM factor_combinations
        WHERE icir IS NULL
    ''').fetchall()

    print(f'Found {len(factors)} factors without ICIR')

    updated = 0

    for factor in factors:
        factor_id = factor[0]
        win_rate = factor[2]
        sharpe = factor[3]
        trades = factor[4]

        # 从夏普比率估算ICIR
        # ICIR ≈ Sharpe / sqrt(252) * adjustment
        # 这是一个近似值
        if sharpe and trades and trades > 0:
            # 根据交易次数调整
            freq_adjustment = (trades / 252) ** 0.5  # 频率调整
            icir = sharpe * freq_adjustment / 15.87  # sqrt(252) ≈ 15.87

            # 根据胜率进一步调整
            if win_rate:
                wr_adjustment = (win_rate - 0.5) * 2  # [-1, 1]
                icir = icir * (1 + wr_adjustment * 0.3)

            # 更新数据库
            conn.execute('''
                UPDATE factor_combinations
                SET icir = ?
                WHERE id = ?
            ''', (icir, factor_id))

            updated += 1

            if updated % 100 == 0:
                print(f'  Progress: {updated}/{len(factors)}')

    conn.commit()

    # 统计
    total_with_icir = conn.execute(
        'SELECT COUNT(*) FROM factor_combinations WHERE icir IS NOT NULL'
    ).fetchone()[0]

    print(f'\nCompleted:')
    print(f'  Updated: {updated}')
    print(f'  Total with ICIR: {total_with_icir}')

    conn.close()

if __name__ == '__main__':
    calculate_icir_from_metrics()
