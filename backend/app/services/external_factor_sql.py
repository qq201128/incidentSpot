from __future__ import annotations

UPSERT_POSITIONING_SQL = """
INSERT INTO futures_positioning_features(
  symbol, open_time, open_interest, open_interest_value, long_short_ratio,
  long_account, short_account, taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol,
  open_interest_chg_1, open_interest_value_chg_1, open_interest_z_20,
  long_short_ratio_chg_1, taker_buy_share
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, open_time) DO UPDATE SET
  open_interest=COALESCE(excluded.open_interest, futures_positioning_features.open_interest),
  open_interest_value=COALESCE(excluded.open_interest_value, futures_positioning_features.open_interest_value),
  long_short_ratio=COALESCE(excluded.long_short_ratio, futures_positioning_features.long_short_ratio),
  long_account=COALESCE(excluded.long_account, futures_positioning_features.long_account),
  short_account=COALESCE(excluded.short_account, futures_positioning_features.short_account),
  taker_buy_sell_ratio=COALESCE(excluded.taker_buy_sell_ratio, futures_positioning_features.taker_buy_sell_ratio),
  taker_buy_vol=COALESCE(excluded.taker_buy_vol, futures_positioning_features.taker_buy_vol),
  taker_sell_vol=COALESCE(excluded.taker_sell_vol, futures_positioning_features.taker_sell_vol),
  open_interest_chg_1=COALESCE(excluded.open_interest_chg_1, futures_positioning_features.open_interest_chg_1),
  open_interest_value_chg_1=COALESCE(excluded.open_interest_value_chg_1, futures_positioning_features.open_interest_value_chg_1),
  open_interest_z_20=COALESCE(excluded.open_interest_z_20, futures_positioning_features.open_interest_z_20),
  long_short_ratio_chg_1=COALESCE(excluded.long_short_ratio_chg_1, futures_positioning_features.long_short_ratio_chg_1),
  taker_buy_share=COALESCE(excluded.taker_buy_share, futures_positioning_features.taker_buy_share)
"""

UPSERT_SENTIMENT_SQL = """
INSERT INTO market_sentiment_features(
  source, open_time, fear_greed_value, fear_greed_classification
)
VALUES(?, ?, ?, ?)
ON CONFLICT(source, open_time) DO UPDATE SET
  fear_greed_value=excluded.fear_greed_value,
  fear_greed_classification=excluded.fear_greed_classification
"""

UPSERT_ONCHAIN_SQL = """
INSERT INTO onchain_features(
  symbol, open_time, exchange_netflow, stablecoin_supply_ratio, active_addresses,
  transaction_count, exchange_netflow_z_20, active_addresses_chg_1, transaction_count_chg_1
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(symbol, open_time) DO UPDATE SET
  exchange_netflow=COALESCE(excluded.exchange_netflow, onchain_features.exchange_netflow),
  stablecoin_supply_ratio=COALESCE(excluded.stablecoin_supply_ratio, onchain_features.stablecoin_supply_ratio),
  active_addresses=COALESCE(excluded.active_addresses, onchain_features.active_addresses),
  transaction_count=COALESCE(excluded.transaction_count, onchain_features.transaction_count),
  exchange_netflow_z_20=COALESCE(excluded.exchange_netflow_z_20, onchain_features.exchange_netflow_z_20),
  active_addresses_chg_1=COALESCE(excluded.active_addresses_chg_1, onchain_features.active_addresses_chg_1),
  transaction_count_chg_1=COALESCE(excluded.transaction_count_chg_1, onchain_features.transaction_count_chg_1)
"""

UPSERT_FUNDING_SQL = """
INSERT INTO funding_features(symbol, open_time, funding_rate)
VALUES(?, ?, ?)
ON CONFLICT(symbol, open_time) DO UPDATE SET
  funding_rate=excluded.funding_rate
"""

UPDATE_POSITIONING_DERIVED_SQL = """
UPDATE futures_positioning_features
SET open_interest_chg_1 = ?,
    open_interest_value_chg_1 = ?,
    open_interest_z_20 = ?,
    long_short_ratio_chg_1 = ?,
    taker_buy_share = ?
WHERE symbol = ? AND open_time = ?
"""

UPDATE_SENTIMENT_DERIVED_SQL = """
UPDATE market_sentiment_features
SET fear_greed_chg_1 = ?,
    fear_greed_z_30 = ?
WHERE source = ? AND open_time = ?
"""

UPDATE_ONCHAIN_DERIVED_SQL = """
UPDATE onchain_features
SET exchange_netflow_z_20 = ?,
    active_addresses_chg_1 = ?,
    transaction_count_chg_1 = ?
WHERE symbol = ? AND open_time = ?
"""
