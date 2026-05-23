CREATE TABLE IF NOT EXISTS klines (
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL,
  close_time INTEGER NOT NULL,
  PRIMARY KEY (symbol, interval, open_time)
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  strategy_key TEXT NOT NULL DEFAULT 'manual',
  symbol TEXT NOT NULL,
  title TEXT NOT NULL,
  event_interval TEXT NOT NULL DEFAULT '1m',
  rule_type TEXT NOT NULL DEFAULT 'ABOVE',
  strike_value REAL NOT NULL,
  upper_bound REAL,
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'OPEN',
  result TEXT,
  settlement_price REAL,
  settlement_quote_time INTEGER,
  settlement_source TEXT,
  ai_probability_up REAL,
  ai_predicted_direction TEXT,
  ai_prediction_correct INTEGER,
  ai_quality_score REAL,
  ai_quality_passed INTEGER,
  ai_high_winrate_gate TEXT,
  ai_high_winrate_rule TEXT,
  ai_high_winrate_passed INTEGER,
  ai_high_winrate_value REAL,
  prediction_open_time INTEGER
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  side TEXT NOT NULL,
  price REAL NOT NULL,
  qty REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'OPEN',
  created_at TEXT NOT NULL,
  external_order_id TEXT,
  external_status TEXT,
  external_response TEXT,
  FOREIGN KEY (event_id) REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS settlements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  order_id INTEGER NOT NULL,
  pnl REAL NOT NULL,
  settled_at TEXT NOT NULL,
  FOREIGN KEY (event_id) REFERENCES events(id),
  FOREIGN KEY (order_id) REFERENCES orders(id)
);

-- Enhanced prediction: orderbook snapshots aligned to open_time
CREATE TABLE IF NOT EXISTS orderbook_features (
  symbol TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  imbalance REAL,
  spread_bps REAL,
  bid_qty_sum REAL,
  ask_qty_sum REAL,
  best_bid REAL,
  best_ask REAL,
  best_bid_qty REAL,
  best_ask_qty REAL,
  microprice REAL,
  microprice_bps REAL,
  ofi REAL,
  ofi_ratio REAL,
  quote_time INTEGER,
  PRIMARY KEY (symbol, open_time)
);

CREATE TABLE IF NOT EXISTS orderbook_ticks (
  symbol TEXT NOT NULL,
  quote_time INTEGER NOT NULL,
  best_bid REAL NOT NULL,
  best_ask REAL NOT NULL,
  best_bid_qty REAL NOT NULL,
  best_ask_qty REAL NOT NULL,
  bid_qty_sum REAL NOT NULL,
  ask_qty_sum REAL NOT NULL,
  imbalance REAL NOT NULL,
  microprice REAL NOT NULL,
  microprice_bps REAL NOT NULL,
  ofi REAL,
  ofi_ratio REAL,
  PRIMARY KEY (symbol, quote_time)
);

-- Enhanced prediction: funding-rate snapshots
CREATE TABLE IF NOT EXISTS funding_features (
  symbol TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  funding_rate REAL,
  PRIMARY KEY (symbol, open_time)
);

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
);

CREATE TABLE IF NOT EXISTS market_sentiment_features (
  source TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  fear_greed_value REAL,
  fear_greed_classification TEXT,
  fear_greed_chg_1 REAL,
  fear_greed_z_30 REAL,
  PRIMARY KEY (source, open_time)
);

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
);

CREATE TABLE IF NOT EXISTS index_price_ticks (
  symbol TEXT NOT NULL,
  quote_time INTEGER NOT NULL,
  index_price REAL NOT NULL,
  mark_price REAL NOT NULL,
  PRIMARY KEY (symbol, quote_time)
);

-- Enhanced prediction: multi-timeframe klines (5m, 15m, 1h) for richer feature cross-sections
CREATE TABLE IF NOT EXISTS klines_multi (
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL,
  close_time INTEGER NOT NULL,
  PRIMARY KEY (symbol, interval, open_time)
);

-- Real-time prediction stream table
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_key TEXT NOT NULL DEFAULT 'factor_combo_ranker_v1',
  strategy_key TEXT NOT NULL DEFAULT 'factor_combo_ranker_v1',
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  direction TEXT NOT NULL,
  probability_up REAL NOT NULL,
  confidence REAL NOT NULL,
  certainty_label TEXT NOT NULL,
  trade_quality_score REAL,
  trade_quality_passed INTEGER,
  trade_quality_gate TEXT,
  high_winrate_gate TEXT,
  high_winrate_rule TEXT,
  high_winrate_gate_passed INTEGER,
  high_winrate_gate_value REAL,
  high_winrate_gate_min REAL,
  entry_price REAL,
  expected_return REAL,
  model_version TEXT,
  feature_window INTEGER,
  model_duration TEXT,
  model_trained_at TEXT,
  exit_price REAL,
  actual_return REAL,
  prediction_correct INTEGER,
  settled_at TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auto_trade_settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  enabled INTEGER NOT NULL DEFAULT 0,
  live_trading_enabled INTEGER NOT NULL DEFAULT 0,
  symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
  duration TEXT NOT NULL DEFAULT '10m',
  duration_minutes INTEGER NOT NULL DEFAULT 10,
  qty REAL NOT NULL DEFAULT 5,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auto_trade_strategies (
  strategy_key TEXT NOT NULL,
  duration TEXT NOT NULL DEFAULT '10m',
  enabled INTEGER NOT NULL DEFAULT 0,
  live_trading_enabled INTEGER NOT NULL DEFAULT 0,
  symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
  duration_minutes INTEGER NOT NULL DEFAULT 10,
  qty REAL NOT NULL DEFAULT 5,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (strategy_key, duration)
);

-- Background-precomputed factor IR rankings (see factor_ranking_background)
CREATE TABLE IF NOT EXISTS factor_ranking_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  total INTEGER NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration)
);

-- Background-precomputed multi-factor combination rankings.
CREATE TABLE IF NOT EXISTS factor_combo_ranking_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  total INTEGER NOT NULL,
  search_config TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration)
);

CREATE TABLE IF NOT EXISTS factor_combo_signal_cache (
  symbol TEXT NOT NULL PRIMARY KEY,
  updated_at TEXT NOT NULL,
  top_per_duration INTEGER NOT NULL,
  limit_count INTEGER NOT NULL,
  payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factor_combo_backtest_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  factor_name TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration, factor_name)
);

CREATE TABLE IF NOT EXISTS high_winrate_combo_ranking_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  total INTEGER NOT NULL,
  search_config TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration)
);

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
);

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
);

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
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_duration_open ON predictions(symbol, duration, open_time);
CREATE INDEX IF NOT EXISTS idx_predictions_settled ON predictions(symbol, duration, settled_at);
CREATE INDEX IF NOT EXISTS idx_events_shadow_pairing ON events(symbol, event_interval, status, prediction_open_time);
CREATE INDEX IF NOT EXISTS idx_events_settled_strategy ON events(symbol, event_interval, strategy_key, status);
CREATE INDEX IF NOT EXISTS idx_orders_event_id ON orders(event_id);
