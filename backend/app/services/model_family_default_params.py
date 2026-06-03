from __future__ import annotations

from typing import Any


def family_default_params(family: str) -> dict[str, Any]:
    return {
        "random_forest": {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
        "extra_trees": {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
        "xgboost": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08},
        "lightgbm": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.08},
        "catboost": {"iterations": 100, "depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 3.0},
        "logistic_elasticnet": {"C": 1.0, "l1_ratio": 0.5},
        "svm": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
        "bayesian": {"var_smoothing": 1e-9},
        "knn": {"n_neighbors": 7, "weights": "distance"},
        "rl_strategy": {"state_bins": 5, "alpha": 0.2, "gamma": 0.8, "epsilon": 0.1, "episodes": 20},
    }.get(family, {})
