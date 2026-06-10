from __future__ import annotations

from typing import Any

from app.services.experiment_profiles import lstm_training_config_for_profile, normalize_experiment_profile
from app.services.model_family_config import ModelFamilyTrainingConfig, normalize_model_family
from app.services.model_family_default_params import family_default_params


def model_training_config_for_profile(
    family: str,
    symbol: str,
    duration: str,
    *,
    profile: str,
    **overrides: Any,
) -> ModelFamilyTrainingConfig:
    selected = normalize_model_family(family)
    base = lstm_training_config_for_profile(symbol, duration, normalize_experiment_profile(profile), **overrides)
    params = family_default_params(selected)
    return ModelFamilyTrainingConfig(
        family=selected,
        symbol=base.symbol,
        duration=base.duration,
        feature_window=base.feature_window,
        horizon_minutes=base.horizon_minutes,
        min_samples=base.min_samples,
        epochs=base.epochs,
        batch_size=base.batch_size,
        hidden_size=base.hidden_size,
        num_layers=base.num_layers,
        learning_rate=base.learning_rate,
        train_ratio=base.train_ratio,
        val_ratio=base.val_ratio,
        min_move_bps=base.min_move_bps,
        seed=base.seed,
        params=params,
    )
