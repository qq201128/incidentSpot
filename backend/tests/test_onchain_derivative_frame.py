from __future__ import annotations

import pandas as pd
import pytest

from app.services.external_factor_derivatives import onchain_derivative_frame


def test_onchain_derivative_frame_preserves_merged_daily_zscore() -> None:
    frame = pd.DataFrame(
        {
            "open_time": [1, 2, 3, 4],
            "exchange_netflow": [100.0, 100.0, 100.0, 200.0],
            "active_addresses": [1000.0, 1000.0, 1100.0, 1100.0],
            "transaction_count": [10.0, 10.0, 11.0, 11.0],
            "exchange_netflow_z_20": [-0.7, -0.7, -0.7, -0.7],
            "active_addresses_chg_1": [0.0, 0.0, 0.1, 0.0],
            "transaction_count_chg_1": [0.0, 0.0, 0.1, 0.0],
        }
    )

    result = onchain_derivative_frame(frame)

    assert result["exchange_netflow_z_20"].tolist() == [-0.7, -0.7, -0.7, -0.7]
    assert result["active_addresses_chg_1"].tolist() == [0.0, 0.0, 0.1, 0.0]


def test_onchain_derivative_frame_computes_missing_columns_from_daily_rows() -> None:
    frame = pd.DataFrame(
        {
            "open_time": [1, 2, 3],
            "exchange_netflow": [1.0, 3.0, 5.0],
            "active_addresses": [100.0, 110.0, 121.0],
            "transaction_count": [10.0, 11.0, 12.0],
        }
    )

    result = onchain_derivative_frame(frame)

    assert pd.notna(result.loc[2, "exchange_netflow_z_20"])
    assert result["active_addresses_chg_1"].dropna().tolist() == pytest.approx([0.1, 0.1])
    assert result["transaction_count_chg_1"].dropna().tolist() == pytest.approx([0.1, 1.0 / 11.0])
