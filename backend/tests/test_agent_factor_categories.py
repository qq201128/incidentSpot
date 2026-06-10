from __future__ import annotations

from app.services.agent_factor_categories import (
    DERIVATIVES_CATEGORY,
    category_saturation,
    category_share,
    factor_category,
)


def test_factor_category_detects_derivatives_and_price_action() -> None:
    assert factor_category({"formula": "Delta(OpenInterestZ(60), 5)"}) == DERIVATIVES_CATEGORY
    assert factor_category({"formula": "Slope(close, 60)"}) == "price_action"


def test_category_share_counts_rows() -> None:
    rows = [
        {"formula": "FundingZ(20)"},
        {"formula": "OpenInterestZ(60)"},
        {"formula": "VWAPDev(close, volume, 20)"},
    ]

    share = category_share(rows)

    assert share[0]["category"] == DERIVATIVES_CATEGORY
    assert share[0]["count"] == 2
    assert round(share[0]["share"], 3) == 0.667


def test_derivatives_saturates_after_mature_overweight_library() -> None:
    rows = [{"formula": "FundingZ(20)"} for _ in range(5)]

    saturation = category_saturation(DERIVATIVES_CATEGORY, rows)

    assert saturation["saturated"] is True
    assert saturation["maxShare"] == 0.4
