from __future__ import annotations

POSITIONING_RAW_COLUMNS = (
    "open_interest",
    "open_interest_value",
    "long_short_ratio",
    "long_account",
    "short_account",
    "taker_buy_sell_ratio",
    "taker_buy_vol",
    "taker_sell_vol",
)
POSITIONING_DERIVED_COLUMNS = (
    "open_interest_chg_1",
    "open_interest_value_chg_1",
    "open_interest_z_20",
    "long_short_ratio_chg_1",
    "taker_buy_share",
)
POSITIONING_COLUMNS = (*POSITIONING_RAW_COLUMNS, *POSITIONING_DERIVED_COLUMNS)

SENTIMENT_RAW_COLUMNS = ("fear_greed_value",)
SENTIMENT_DERIVED_COLUMNS = ("fear_greed_chg_1", "fear_greed_z_30")
SENTIMENT_COLUMNS = (*SENTIMENT_RAW_COLUMNS, *SENTIMENT_DERIVED_COLUMNS)

ONCHAIN_RAW_COLUMNS = (
    "exchange_netflow",
    "stablecoin_supply_ratio",
    "active_addresses",
    "transaction_count",
)
ONCHAIN_DERIVED_COLUMNS = (
    "exchange_netflow_z_20",
    "active_addresses_chg_1",
    "transaction_count_chg_1",
)
ONCHAIN_COLUMNS = (*ONCHAIN_RAW_COLUMNS, *ONCHAIN_DERIVED_COLUMNS)
