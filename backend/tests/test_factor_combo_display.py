from __future__ import annotations

from app.services.factor_combo_display import (
    collect_leaf_factor_names,
    combo_display_name,
    parse_combo_member_names,
    short_factor_label,
)


def test_short_factor_label_maps_common_factors() -> None:
    assert short_factor_label("adx_14") == "ADX(14)"
    assert short_factor_label("ma_ratio_120") == "均线偏离(120)"
    assert short_factor_label("profit_factor_60") == "盈亏比(60)"
    assert short_factor_label("macd_signal") == "MACD"


def test_parse_combo_member_names_handles_nested_combos() -> None:
    assert parse_combo_member_names("combo__adx_14__ma_ratio_120__profit_factor_60") == [
        "adx_14",
        "ma_ratio_120",
        "profit_factor_60",
    ]
    assert parse_combo_member_names(
        "combo__adx_14__combo__adx_14__ma_ratio_120__profit_factor_60"
    ) == [
        "adx_14",
        "combo__adx_14__ma_ratio_120__profit_factor_60",
    ]


def test_combo_display_name_dedupes_leaf_factors() -> None:
    members = [
        {"name": "adx_14", "displayName": "ADX趋势强度（14周期）"},
        {
            "name": "combo__adx_14__ma_ratio_120__profit_factor_60",
            "displayName": "组合：ADX趋势强度（14周期） + 120周期均线偏离 + 60周期盈亏比",
        },
        {
            "name": "combo__adx_14__profit_factor_60__macd_signal",
            "displayName": "组合：ADX趋势强度（14周期） + 60周期盈亏比 + MACD信号线",
        },
    ]

    assert combo_display_name(members) == "组合：ADX(14) + 均线偏离(120) + 盈亏比(60) + MACD"


def test_combo_display_name_flattens_combo_encoded_members() -> None:
    members = [
        {
            "name": (
                "combo__combo__adx_14__rolling_sharpe_60__ma_ratio_60"
                "__combo__adx_14__profit_factor_60__ma_ratio_60"
                "__combo__rolling_sharpe_60__combo__adx_14__ret_autocorr_20"
            )
        },
    ]

    assert combo_display_name(members) == (
        "组合：ADX(14) + 滚动夏普(60) + 均线偏离(60) + 盈亏比(60) + 20周期收益自相关"
    )


def test_collect_leaf_factor_names_preserves_first_seen_order() -> None:
    leaves = collect_leaf_factor_names([
        {"name": "factor_b"},
        {"name": "combo__factor_a__factor_b"},
    ])

    assert leaves == ["factor_b", "factor_a"]
