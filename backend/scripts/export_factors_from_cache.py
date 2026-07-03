"""
从缓存导出组合因子数据（不写入数据库）
"""
import sqlite3
import json

def export_factors_from_cache():
    """从缓存导出组合因子数据"""

    conn = sqlite3.connect('backend/data.db', timeout=1)
    conn.row_factory = sqlite3.Row

    print('Exporting factor combinations from cache...\n')

    caches = conn.execute('SELECT symbol, duration, payload FROM factor_combo_ranking_cache').fetchall()

    all_factors = []

    for cache in caches:
        symbol = cache['symbol']
        duration = cache['duration']
        payload_json = cache['payload']

        try:
            payload = json.loads(payload_json)
            ranking = payload.get('ranking', [])

            print(f'{symbol}@{duration}: {len(ranking)} combos')

            for combo in ranking:
                # 提取关键字段
                factor = {
                    'name': combo.get('factorName'),
                    'display_name': combo.get('factorDisplayName', ''),
                    'symbol': symbol,
                    'duration': duration,
                    'formula': combo.get('formula', ''),
                    'members': combo.get('members', []),
                    'ir': combo.get('ir'),
                    'win_rate': combo.get('winRate'),
                    'sharpe': combo.get('sharpe'),
                    'max_drawdown': combo.get('maxDrawdown'),
                    'trades': combo.get('totalPeriods', 0),
                    'profit_factor': combo.get('profitFactor'),
                    'walk_forward': combo.get('walkForward'),
                }

                if factor['name']:
                    all_factors.append(factor)

        except Exception as e:
            print(f'  ERROR processing {symbol}@{duration}: {e}')

    conn.close()

    # 保存到JSON文件
    with open('factor_combinations_export.json', 'w', encoding='utf-8') as f:
        json.dump(all_factors, f, indent=2, ensure_ascii=False)

    print(f'\n=== Export Summary ===')
    print(f'Total factors exported: {len(all_factors)}')
    print(f'Saved to: factor_combinations_export.json')

    # 统计
    by_symbol = {}
    with_icir = 0
    with_winrate = 0

    for f in all_factors:
        key = f'{f["symbol"]}@{f["duration"]}'
        by_symbol[key] = by_symbol.get(key, 0) + 1
        if f['ir'] is not None:
            with_icir += 1
        if f['win_rate'] is not None:
            with_winrate += 1

    print(f'\nBy Symbol/Duration:')
    for key, count in sorted(by_symbol.items()):
        print(f'  {key}: {count}')

    print(f'\nData Quality:')
    print(f'  With ICIR: {with_icir} ({with_icir/len(all_factors)*100:.1f}%)')
    print(f'  With Win Rate: {with_winrate} ({with_winrate/len(all_factors)*100:.1f}%)')

    return all_factors

if __name__ == '__main__':
    export_factors_from_cache()
