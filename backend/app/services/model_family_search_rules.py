from __future__ import annotations

from itertools import product
from typing import Any

from app.services.model_family_config import normalize_model_family
from app.services.model_search_resource_defaults import DEFAULT_INTERNAL_THREADS, DEFAULT_XGBOOST_PROCESS_WORKERS

DEFAULT_PARALLEL_WORKERS = 1
TARGET_WIN_RATE_EXCLUSIVE = 0.62
SUCCESSIVE_HALVING_SURVIVAL_RATE = 0.50
COMMON_FEATURE_WINDOWS = (24, 32, 48, 64, 96)
COMMON_MIN_MOVE_BPS = (8.0, 10.0, 12.0, 15.0, 20.0)
COMMON_SEEDS = (20260513, 20260519, 20260601)
NEURAL_EPOCHS = (8, 12, 16)
NEURAL_REGULARIZATION_PROFILES = (
    {},
    {"dropout": 0.15, "weight_decay": 1e-4, "early_stopping_patience": 3},
    {"class_weight_mode": "balanced"},
    {"return_weight_mode": "abs_return"},
)
TRANSFORMER_ATTENTION_PROFILES = (
    {},
    {"transformer_nhead": 8, "use_positional_encoding": True},
)


def model_family_search_grid(family: str) -> list[dict[str, Any]]:
    selected = normalize_model_family(family)
    if selected in {"lstm", "gru", "cnn", "transformer"}:
        return _neural_grid(selected)
    if selected == "random_forest":
        return _with_common(_random_forest_params())
    if selected == "extra_trees":
        return _with_common(_extra_trees_params())
    if selected == "xgboost":
        return _with_common(_xgboost_params())
    if selected == "lightgbm":
        return _with_common(_lightgbm_params())
    if selected == "catboost":
        return _with_common(_catboost_params())
    if selected == "logistic_elasticnet":
        return _with_common(_logistic_elasticnet_params(), windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513,))
    if selected == "svm":
        return _with_common(_svm_params(), windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513,))
    if selected == "bayesian":
        return _with_common(_bayesian_params(), windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513,))
    if selected == "knn":
        return _with_common(_knn_params(), windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513,))
    return _with_common(_rl_params(), windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513, 20260519))


def model_family_training_rules(family: str) -> dict[str, Any]:
    selected = normalize_model_family(family)
    grid = model_family_search_grid(selected)
    return {
        "modelFamily": selected,
        "searchMode": "successive_halving",
        "searchSpaceTotal": len(grid),
        "internalThreads": DEFAULT_INTERNAL_THREADS,
        "parallelWorkers": DEFAULT_PARALLEL_WORKERS,
        "xgboostProcessWorkers": DEFAULT_XGBOOST_PROCESS_WORKERS,
        "targetWinRateExclusive": TARGET_WIN_RATE_EXCLUSIVE,
        "requiresValidationAndTestAboveTarget": False,
        "selectionMetricSource": "validation_only",
        "testPolicy": "reported_once_for_final_candidate_not_used_for_threshold_selection",
        "successiveHalving": _successive_halving_rules(),
        "candidateSearchAxes": _axes_for_family(selected),
    }


def _successive_halving_rules() -> list[dict[str, Any]]:
    return [
        {
            "stage": "coarse",
            "purpose": "short_epoch_or_reduced_sample_screen",
            "survivalRate": SUCCESSIVE_HALVING_SURVIVAL_RATE,
            "publishesArtifacts": False,
        },
        {
            "stage": "full",
            "purpose": "complete_training_for_survivors",
            "survivalRate": SUCCESSIVE_HALVING_SURVIVAL_RATE,
            "publishesArtifacts": False,
        },
        {
            "stage": "walk_forward",
            "purpose": "final_oos_validation_and_serial_publish",
            "survivalRate": 1.0,
            "publishesArtifacts": True,
        },
    ]


def _neural_grid(family: str) -> list[dict[str, Any]]:
    return [
        _neural_candidate(window, move, epochs, seed, regularization, attention)
        for window, move, epochs, seed, regularization, attention in product(
            COMMON_FEATURE_WINDOWS,
            COMMON_MIN_MOVE_BPS,
            NEURAL_EPOCHS,
            COMMON_SEEDS,
            NEURAL_REGULARIZATION_PROFILES,
            _attention_profiles(family),
        )
    ]


def _attention_profiles(family: str) -> tuple[dict[str, Any], ...]:
    return TRANSFORMER_ATTENTION_PROFILES if family == "transformer" else ({},)


def _neural_candidate(window, move, epochs, seed, regularization, attention) -> dict[str, Any]:
    params = {**regularization, **attention}
    candidate = {"feature_window": window, "min_move_bps": move, "epochs": epochs, "seed": seed}
    return {**candidate, "params": params} if params else candidate


def _with_common(
    params: list[dict[str, Any]],
    *,
    windows: tuple[int, ...] = (24, 48, 64, 96),
    min_moves: tuple[float, ...] = (8.0, 12.0, 20.0),
    seeds: tuple[int, ...] = (20260513, 20260519),
) -> list[dict[str, Any]]:
    return [
        {"feature_window": window, "min_move_bps": move, "seed": seed, "params": item}
        for window, move, seed, item in product(windows, min_moves, seeds, params)
    ]


def _random_forest_params() -> list[dict[str, Any]]:
    return [
        {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf}
        for trees, depth, leaf in product((80, 140, 220), (None, 6, 10), (1, 2))
    ]


def _extra_trees_params() -> list[dict[str, Any]]:
    return [
        {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf}
        for trees, depth, leaf in product((80, 140, 220), (None, 6, 10), (1, 2))
    ]


def _xgboost_params() -> list[dict[str, Any]]:
    return [
        {"n_estimators": trees, "max_depth": depth, "learning_rate": rate, "subsample": subsample}
        for trees, depth, rate, subsample in product((60, 100, 160), (3, 4, 5), (0.04, 0.08), (0.8, 1.0))
    ]


def _lightgbm_params() -> list[dict[str, Any]]:
    return [
        {"n_estimators": trees, "num_leaves": leaves, "learning_rate": rate, "subsample": subsample}
        for trees, leaves, rate, subsample in product((60, 100, 160), (15, 31, 63), (0.04, 0.08), (0.8, 1.0))
    ]


def _catboost_params() -> list[dict[str, Any]]:
    return [
        {"iterations": iterations, "depth": depth, "learning_rate": rate, "l2_leaf_reg": reg}
        for iterations, depth, rate, reg in product((80, 140), (3, 5), (0.04, 0.08), (3.0, 8.0))
    ]


def _logistic_elasticnet_params() -> list[dict[str, Any]]:
    return [{"C": c, "l1_ratio": ratio} for c, ratio in product((0.5, 1.0, 2.0), (0.15, 0.5, 0.85))]


def _svm_params() -> list[dict[str, Any]]:
    return [{"C": c, "kernel": kernel, "gamma": gamma} for c, kernel, gamma in product((0.5, 1.0, 2.0), ("rbf", "linear"), ("scale", "auto"))]


def _bayesian_params() -> list[dict[str, Any]]:
    return [{"var_smoothing": value} for value in (1e-9, 1e-8, 1e-7)]


def _knn_params() -> list[dict[str, Any]]:
    return [{"n_neighbors": k, "weights": w} for k, w in product((3, 5, 7, 9), ("uniform", "distance"))]


def _rl_params() -> list[dict[str, Any]]:
    return [
        {"state_bins": bins, "alpha": alpha, "gamma": gamma, "epsilon": eps, "episodes": episodes}
        for bins, alpha, gamma, eps, episodes in product((4, 6, 8), (0.1, 0.2), (0.6, 0.8), (0.05, 0.1), (10, 20))
    ]


def _axes_for_family(family: str) -> dict[str, list[Any]]:
    common = _common_axes()
    compact = _common_axes(windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513,))
    compact_rl = _common_axes(windows=(24, 48, 64), min_moves=(8.0, 12.0, 20.0), seeds=(20260513, 20260519))
    axes = {
        "lstm": {**common, "epochs": list(NEURAL_EPOCHS), **_neural_regularization_axes()},
        "gru": {**common, "epochs": list(NEURAL_EPOCHS), **_neural_regularization_axes()},
        "cnn": {**common, "epochs": list(NEURAL_EPOCHS), **_neural_regularization_axes()},
        "transformer": {**common, "epochs": list(NEURAL_EPOCHS), **_neural_regularization_axes(), **_transformer_axes()},
        "random_forest": {**common, "nEstimators": [80, 140, 220], "maxDepth": [None, 6, 10], "minSamplesLeaf": [1, 2]},
        "extra_trees": {**common, "nEstimators": [80, 140, 220], "maxDepth": [None, 6, 10], "minSamplesLeaf": [1, 2]},
        "xgboost": {**common, "nEstimators": [60, 100, 160], "maxDepth": [3, 4, 5], "learningRate": [0.04, 0.08], "subsample": [0.8, 1.0]},
        "lightgbm": {**common, "nEstimators": [60, 100, 160], "numLeaves": [15, 31, 63], "learningRate": [0.04, 0.08], "subsample": [0.8, 1.0]},
        "catboost": {**common, "iterations": [80, 140], "depth": [3, 5], "learningRate": [0.04, 0.08], "l2LeafReg": [3.0, 8.0]},
        "logistic_elasticnet": {**compact, "C": [0.5, 1.0, 2.0], "l1Ratio": [0.15, 0.5, 0.85]},
        "svm": {**compact, "C": [0.5, 1.0, 2.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        "bayesian": {**compact, "varSmoothing": [1e-9, 1e-8, 1e-7]},
        "knn": {**compact, "nNeighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        "rl_strategy": {**compact_rl, "stateBins": [4, 6, 8], "alpha": [0.1, 0.2], "gamma": [0.6, 0.8], "epsilon": [0.05, 0.1], "episodes": [10, 20]},
    }
    return axes[family]


def _common_axes(
    *,
    windows: tuple[int, ...] = COMMON_FEATURE_WINDOWS,
    min_moves: tuple[float, ...] = COMMON_MIN_MOVE_BPS,
    seeds: tuple[int, ...] = COMMON_SEEDS,
) -> dict[str, list[Any]]:
    return {"featureWindow": list(windows), "minMoveBps": list(min_moves), "seed": list(seeds)}


def _neural_regularization_axes() -> dict[str, list[Any]]:
    return {
        "dropout": [0.0, 0.15],
        "weightDecay": [0.0, 1e-4],
        "earlyStoppingPatience": [0, 3],
        "classWeightMode": ["none", "balanced"],
        "returnWeightMode": ["none", "abs_return"],
    }


def _transformer_axes() -> dict[str, list[Any]]:
    return {"transformerNhead": [4, 8], "usePositionalEncoding": [False, True]}
