from __future__ import annotations

from typing import Any

TECHNICAL_INDICATOR_FACTORS: tuple[dict[str, Any], ...] = (
    {
        "name": "aroon_up_25",
        "category": "momentum",
        "description": "Aroon Up（25周期）",
        "formula": "100 * (25 - periods_since_high(25)) / 25",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "aroon_down_25",
        "category": "momentum",
        "description": "Aroon Down（25周期）",
        "formula": "100 * (25 - periods_since_low(25)) / 25",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "aroon_osc_25",
        "category": "momentum",
        "description": "Aroon振荡器（25周期）",
        "formula": "aroon_up_25 - aroon_down_25",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "dmi_spread_14",
        "category": "momentum",
        "description": "DMI方向差（14周期）",
        "formula": "+DI(14) - -DI(14)",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "keltner_width_20",
        "category": "volatility",
        "description": "肯特纳通道宽度（20周期）",
        "formula": "4 * ATR(10) / EMA(20)",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "keltner_pos_20",
        "category": "structure",
        "description": "肯特纳通道位置（20周期）",
        "formula": "(close - EMA(20)) / (2 * ATR(10))",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "trix_15",
        "category": "momentum",
        "description": "TRIX三重指数动量（15周期）",
        "formula": "pct_change(EMA(EMA(EMA(close,15),15),15))",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "tsi_25_13",
        "category": "momentum",
        "description": "真实强弱指数TSI（25/13）",
        "formula": "100 * double_ema(momentum,25,13) / double_ema(abs(momentum),25,13)",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
    {
        "name": "ultimate_osc_7_14_28",
        "category": "momentum",
        "description": "终极振荡器（7/14/28）",
        "formula": "100 * (4 * avg7 + 2 * avg14 + avg28) / 7",
        "source_file": "kline_technical_indicators.py",
        "direction": "neutral",
    },
)
