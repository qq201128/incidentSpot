from __future__ import annotations

from dataclasses import dataclass

from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS

MS_PER_MINUTE = 60_000
LSTM_SHADOW_STRATEGY_PREFIX = "factor_lstm_shadow"
LSTM_RULE_NAME = "lstm_shadow_signal_v1"
DEFAULT_FEATURE_WINDOW = 64
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MIN_MOVE_BPS = 8.0
DEFAULT_SEED = 20260513

@dataclass(frozen=True)
class LstmTrainingConfig:
    symbol: str
    duration: str = "10m"
    feature_window: int = DEFAULT_FEATURE_WINDOW
    horizon_minutes: int | None = None
    min_samples: int = 120
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    min_move_bps: float = DEFAULT_MIN_MOVE_BPS
    seed: int = DEFAULT_SEED


def validated_lstm_config(config: LstmTrainingConfig) -> LstmTrainingConfig:
    symbol = config.symbol.strip().upper()
    if len(symbol) < 6:
        raise ValueError("symbol must contain at least 6 characters")
    _validate_duration(config.duration)
    if config.feature_window < 4:
        raise ValueError("feature_window must be >= 4")
    if config.min_samples <= config.feature_window:
        raise ValueError("min_samples must be greater than feature_window")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.hidden_size <= 0 or config.num_layers <= 0:
        raise ValueError("hidden_size and num_layers must be positive")
    if config.train_ratio <= 0 or config.val_ratio <= 0 or config.train_ratio + config.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave a test split")
    if config.min_move_bps < 0:
        raise ValueError("min_move_bps must be >= 0")
    horizon = config.horizon_minutes or duration_minutes(config.duration)
    if horizon <= 0:
        raise ValueError("horizon_minutes must be positive")
    return LstmTrainingConfig(**{**config.__dict__, "symbol": symbol, "horizon_minutes": horizon})


def duration_minutes(duration: str) -> int:
    _validate_duration(duration)
    return int(DURATION_TO_MINUTES[duration])


def duration_ms(duration: str) -> int:
    return duration_minutes(duration) * MS_PER_MINUTE


def lstm_shadow_strategy_key(duration: str) -> str:
    _validate_duration(duration)
    return f"{LSTM_SHADOW_STRATEGY_PREFIX}_{duration}"


def is_lstm_shadow_strategy(strategy_key: str | None) -> bool:
    return bool(strategy_key and strategy_key.startswith(f"{LSTM_SHADOW_STRATEGY_PREFIX}_"))


def lstm_strategy_duration(strategy_key: str) -> str:
    prefix = f"{LSTM_SHADOW_STRATEGY_PREFIX}_"
    if not strategy_key.startswith(prefix):
        raise ValueError(f"not an LSTM shadow strategy: {strategy_key}")
    duration = strategy_key.removeprefix(prefix)
    _validate_duration(duration)
    return duration


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported LSTM duration: {duration}")
