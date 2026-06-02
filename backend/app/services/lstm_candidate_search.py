from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from app.services.experiment_profiles import lstm_training_config_for_profile
from app.services.lstm_candidate_keys import search_key_for_config
from app.services.lstm_config import LstmTrainingConfig

DEFAULT_FEATURE_WINDOWS = (24, 32, 48, 64, 96)
DEFAULT_MIN_MOVE_BPS_VALUES = (8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_EPOCH_VALUES = (8, 12, 16)
DEFAULT_SEEDS = (20260513, 20260519, 20260601)
DEFAULT_CANDIDATES_PER_DURATION = (
    len(DEFAULT_FEATURE_WINDOWS)
    * len(DEFAULT_MIN_MOVE_BPS_VALUES)
    * len(DEFAULT_EPOCH_VALUES)
    * len(DEFAULT_SEEDS)
)
DEFAULT_PARALLEL_WORKERS = 1


@dataclass(frozen=True)
class LstmCandidateSearchConfig:
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS
    min_move_bps_values: tuple[float, ...] = DEFAULT_MIN_MOVE_BPS_VALUES
    epoch_values: tuple[int, ...] = DEFAULT_EPOCH_VALUES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    candidates_per_duration: int = DEFAULT_CANDIDATES_PER_DURATION
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS


@dataclass(frozen=True)
class LstmCandidateSearchRequest:
    symbol: str
    duration: str
    profile: str
    attempted_keys: frozenset[str]
    search_config: LstmCandidateSearchConfig


def validated_search_config(config: LstmCandidateSearchConfig) -> LstmCandidateSearchConfig:
    feature_windows = _positive_ints(config.feature_windows, "feature_windows")
    min_move_bps_values = _positive_floats(config.min_move_bps_values, "min_move_bps_values")
    epoch_values = _positive_ints(config.epoch_values, "epoch_values")
    seeds = _positive_ints(config.seeds, "seeds")
    if config.candidates_per_duration <= 0:
        raise ValueError("candidates_per_duration must be positive")
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    return LstmCandidateSearchConfig(
        feature_windows=feature_windows,
        min_move_bps_values=min_move_bps_values,
        epoch_values=epoch_values,
        seeds=seeds,
        candidates_per_duration=int(config.candidates_per_duration),
        parallel_workers=int(config.parallel_workers),
    )


def next_candidate_configs(request: LstmCandidateSearchRequest) -> list[LstmTrainingConfig]:
    cfg = validated_search_config(request.search_config)
    selected = []
    for candidate in _candidate_grid(request, cfg):
        if search_key_for_config(candidate, request.profile) in request.attempted_keys:
            continue
        selected.append(candidate)
        if len(selected) >= cfg.candidates_per_duration:
            break
    return selected


def search_space_size(config: LstmCandidateSearchConfig) -> int:
    cfg = validated_search_config(config)
    return (
        len(cfg.feature_windows)
        * len(cfg.min_move_bps_values)
        * len(cfg.epoch_values)
        * len(cfg.seeds)
    )


def _candidate_grid(
    request: LstmCandidateSearchRequest,
    config: LstmCandidateSearchConfig,
) -> list[LstmTrainingConfig]:
    return [
        lstm_training_config_for_profile(
            request.symbol,
            request.duration,
            request.profile,
            feature_window=window,
            epochs=epochs,
            min_move_bps=min_move_bps,
            seed=seed,
        )
        for window, min_move_bps, epochs, seed in product(
            config.feature_windows,
            config.min_move_bps_values,
            config.epoch_values,
            config.seeds,
        )
    ]


def _positive_ints(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    selected = tuple(int(value) for value in values)
    if not selected or any(value <= 0 for value in selected):
        raise ValueError(f"{name} must contain positive integers")
    return selected


def _positive_floats(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    selected = tuple(float(value) for value in values)
    if not selected or any(value <= 0 for value in selected):
        raise ValueError(f"{name} must contain positive numbers")
    return selected
