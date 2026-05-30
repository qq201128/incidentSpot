from __future__ import annotations

SCHEMA_MIGRATIONS = (
  "ALTER TABLE events ADD COLUMN rule_type TEXT NOT NULL DEFAULT 'ABOVE'",
  "ALTER TABLE events ADD COLUMN upper_bound REAL",
  "ALTER TABLE events ADD COLUMN event_interval TEXT NOT NULL DEFAULT '1m'",
  "ALTER TABLE events ADD COLUMN settlement_price REAL",
  "ALTER TABLE events ADD COLUMN settlement_quote_time INTEGER",
  "ALTER TABLE events ADD COLUMN settlement_source TEXT",
  "ALTER TABLE events ADD COLUMN strategy_key TEXT NOT NULL DEFAULT 'manual'",
  "ALTER TABLE events ADD COLUMN ai_quality_score REAL",
  "ALTER TABLE events ADD COLUMN ai_quality_passed INTEGER",
  "ALTER TABLE predictions ADD COLUMN strategy_key TEXT NOT NULL DEFAULT 'factor_combo_ranker_v1'",
  "ALTER TABLE predictions ADD COLUMN signal_key TEXT",
  "ALTER TABLE predictions ADD COLUMN trade_quality_score REAL",
  "ALTER TABLE predictions ADD COLUMN trade_quality_passed INTEGER",
  "ALTER TABLE predictions ADD COLUMN trade_quality_gate TEXT",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate TEXT",
  "ALTER TABLE predictions ADD COLUMN high_winrate_rule TEXT",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate_passed INTEGER",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate_value REAL",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate_min REAL",
  "ALTER TABLE predictions ADD COLUMN entry_price REAL",
  "ALTER TABLE predictions ADD COLUMN expected_return REAL",
  "ALTER TABLE predictions ADD COLUMN model_version TEXT",
  "ALTER TABLE predictions ADD COLUMN model_family TEXT",
  "ALTER TABLE predictions ADD COLUMN validation_win_rate REAL",
  "ALTER TABLE predictions ADD COLUMN feature_window INTEGER",
  "ALTER TABLE predictions ADD COLUMN model_duration TEXT",
  "ALTER TABLE predictions ADD COLUMN model_trained_at TEXT",
  "ALTER TABLE predictions ADD COLUMN oos_win_rate REAL",
  "ALTER TABLE predictions ADD COLUMN walk_forward_result TEXT",
  "ALTER TABLE predictions ADD COLUMN recent_rolling_result TEXT",
  "ALTER TABLE predictions ADD COLUMN data_freshness_status TEXT",
  "ALTER TABLE predictions ADD COLUMN missing_feature_status TEXT",
  "ALTER TABLE predictions ADD COLUMN exit_price REAL",
  "ALTER TABLE predictions ADD COLUMN actual_return REAL",
  "ALTER TABLE predictions ADD COLUMN prediction_correct INTEGER",
  "ALTER TABLE predictions ADD COLUMN settled_at TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_gate TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_rule TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_passed INTEGER",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_value REAL",
  "ALTER TABLE orders ADD COLUMN external_order_id TEXT",
  "ALTER TABLE orders ADD COLUMN external_status TEXT",
  "ALTER TABLE orders ADD COLUMN external_response TEXT",
  "ALTER TABLE auto_trade_settings ADD COLUMN live_trading_enabled INTEGER NOT NULL DEFAULT 0",
  "ALTER TABLE orderbook_features ADD COLUMN best_bid REAL",
  "ALTER TABLE orderbook_features ADD COLUMN best_ask REAL",
  "ALTER TABLE orderbook_features ADD COLUMN best_bid_qty REAL",
  "ALTER TABLE orderbook_features ADD COLUMN best_ask_qty REAL",
  "ALTER TABLE orderbook_features ADD COLUMN microprice REAL",
  "ALTER TABLE orderbook_features ADD COLUMN microprice_bps REAL",
  "ALTER TABLE orderbook_features ADD COLUMN ofi REAL",
  "ALTER TABLE orderbook_features ADD COLUMN ofi_ratio REAL",
  "ALTER TABLE orderbook_features ADD COLUMN quote_time INTEGER",
  "ALTER TABLE futures_positioning_features ADD COLUMN open_interest_chg_1 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN open_interest_value_chg_1 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN open_interest_z_20 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN long_short_ratio_chg_1 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN taker_buy_share REAL",
  "ALTER TABLE market_sentiment_features ADD COLUMN fear_greed_chg_1 REAL",
  "ALTER TABLE market_sentiment_features ADD COLUMN fear_greed_z_30 REAL",
  "ALTER TABLE onchain_features ADD COLUMN exchange_netflow_z_20 REAL",
  "ALTER TABLE onchain_features ADD COLUMN active_addresses_chg_1 REAL",
  "ALTER TABLE onchain_features ADD COLUMN transaction_count_chg_1 REAL",
  """
  CREATE TABLE IF NOT EXISTS futures_positioning_features (
    symbol TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open_interest REAL,
    open_interest_value REAL,
    long_short_ratio REAL,
    long_account REAL,
    short_account REAL,
    taker_buy_sell_ratio REAL,
    taker_buy_vol REAL,
    taker_sell_vol REAL,
    open_interest_chg_1 REAL,
    open_interest_value_chg_1 REAL,
    open_interest_z_20 REAL,
    long_short_ratio_chg_1 REAL,
    taker_buy_share REAL,
    PRIMARY KEY (symbol, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS market_sentiment_features (
    source TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    fear_greed_value REAL,
    fear_greed_classification TEXT,
    fear_greed_chg_1 REAL,
    fear_greed_z_30 REAL,
    PRIMARY KEY (source, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS onchain_features (
    symbol TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    exchange_netflow REAL,
    stablecoin_supply_ratio REAL,
    active_addresses REAL,
    transaction_count REAL,
    exchange_netflow_z_20 REAL,
    active_addresses_chg_1 REAL,
    transaction_count_chg_1 REAL,
    PRIMARY KEY (symbol, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS factor_combo_ranking_cache (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total INTEGER NOT NULL,
    search_config TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS factor_combo_signal_cache (
    symbol TEXT NOT NULL PRIMARY KEY,
    updated_at TEXT NOT NULL,
    top_per_duration INTEGER NOT NULL,
    limit_count INTEGER NOT NULL,
    payload TEXT NOT NULL
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS factor_combo_backtest_cache (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, duration, factor_name)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS high_winrate_combo_ranking_cache (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total INTEGER NOT NULL,
    search_config TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS factor_combo_feature_snapshots (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    entry_open_time INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(symbol, duration, entry_open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS high_winrate_strategy_status (
    strategy_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    win_rate REAL,
    profit_factor REAL,
    consecutive_losses INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (strategy_key, symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS paper_live_candidate_status (
    candidate_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(candidate_key, symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS paper_live_prediction_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_key TEXT NOT NULL,
    strategy_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS ensemble_stage_status (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    stage TEXT NOT NULL,
    recommended_stage TEXT NOT NULL,
    recommendation_reason TEXT NOT NULL,
    confirmed_stage TEXT,
    confirmed_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS ensemble_signal_scores (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    signal_key TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    avg_return REAL NOT NULL,
    profit_factor REAL NOT NULL,
    consecutive_losses INTEGER NOT NULL,
    stability_score REAL NOT NULL,
    weight_suggestion REAL NOT NULL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, duration, signal_key)
  )
  """,
  "ALTER TABLE events ADD COLUMN prediction_open_time INTEGER",
  "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_duration_open ON predictions(symbol, duration, open_time)",
  "CREATE INDEX IF NOT EXISTS idx_predictions_settled ON predictions(symbol, duration, settled_at)",
  "CREATE INDEX IF NOT EXISTS idx_predictions_candidate_settled ON predictions(signal_key, COALESCE(high_winrate_rule, model_version, signal_key), settled_at, open_time DESC)",
  "CREATE INDEX IF NOT EXISTS idx_events_shadow_pairing ON events(symbol, event_interval, status, prediction_open_time)",
  "CREATE INDEX IF NOT EXISTS idx_events_settled_strategy ON events(symbol, event_interval, strategy_key, status)",
  "CREATE INDEX IF NOT EXISTS idx_orders_event_id ON orders(event_id)",
  "CREATE INDEX IF NOT EXISTS idx_settlements_event_id ON settlements(event_id)",
  "CREATE INDEX IF NOT EXISTS idx_events_settled_ai_history ON events(symbol, event_interval, strategy_key) WHERE status = 'SETTLED' AND ai_predicted_direction IS NOT NULL AND ai_prediction_correct IS NOT NULL",
)
