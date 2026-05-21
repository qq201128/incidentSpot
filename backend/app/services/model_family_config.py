from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.lstm_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FEATURE_WINDOW,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_MOVE_BPS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    duration_minutes,
)

MODEL_FAMILIES = (
    "lstm",
    "gru",
    "cnn",
    "transformer",
    "random_forest",
    "xgboost",
    "svm",
    "bayesian",
    "knn",
    "rl_strategy",
)
TORCH_MODEL_FAMILIES = frozenset({"lstm", "gru", "cnn", "transformer"})
JOBLIB_MODEL_FAMILIES = frozenset(set(MODEL_FAMILIES) - set(TORCH_MODEL_FAMILIES))
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1


@dataclass(frozen=True)
class ModelFamilyTrainingConfig:
    family: str
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
    params: dict[str, Any] = field(default_factory=dict)


def normalize_model_family(family: str) -> str:
    selected = family.strip().lower()
    if selected not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family: {family}")
    return selected


def model_family_rule_name(family: str) -> str:
    return f"{normalize_model_family(family)}_shadow_signal_v1"


def model_family_strategy_key(family: str, duration: str) -> str:
    selected = normalize_model_family(family)
    duration_minutes(duration)
    return f"factor_{selected}_shadow_{duration}"


def parse_model_family_strategy(strategy_key: str | None) -> tuple[str, str] | None:
    if not strategy_key or not strategy_key.startswith("factor_"):
        return None
    suffix = strategy_key.removeprefix("factor_")
    for family in sorted(MODEL_FAMILIES, key=len, reverse=True):
        prefix = f"{family}_shadow_"
        if suffix.startswith(prefix):
            duration = suffix.removeprefix(prefix)
            duration_minutes(duration)
            return family, duration
    return None


def is_model_family_shadow_strategy(strategy_key: str | None) -> bool:
    return parse_model_family_strategy(strategy_key) is not None


def validated_model_family_config(config: ModelFamilyTrainingConfig) -> ModelFamilyTrainingConfig:
    family = normalize_model_family(config.family)
    symbol = config.symbol.strip().upper()
    if len(symbol) < 6:
        raise ValueError("symbol must contain at least 6 characters")
    horizon = config.horizon_minutes or duration_minutes(config.duration)
    _validate_positive(config.feature_window, "feature_window", minimum=4)
    _validate_positive(config.min_samples, "min_samples", minimum=config.feature_window + 1)
    _validate_positive(config.epochs, "epochs")
    _validate_positive(config.batch_size, "batch_size")
    _validate_positive(config.hidden_size, "hidden_size")
    _validate_positive(config.num_layers, "num_layers")
    if config.train_ratio <= 0 or config.val_ratio <= 0 or config.train_ratio + config.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and leave a test split")
    if config.min_move_bps < 0:
        raise ValueError("min_move_bps must be >= 0")
    if horizon <= 0:
        raise ValueError("horizon_minutes must be positive")
    return ModelFamilyTrainingConfig(
        **{**config.__dict__, "family": family, "symbol": symbol, "horizon_minutes": horizon}
    )


def _validate_positive(value: int, name: str, *, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
