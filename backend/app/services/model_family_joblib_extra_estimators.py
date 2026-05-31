from __future__ import annotations

from typing import Any

LIGHTGBM_FORCE_COL_WISE = True
LOGISTIC_MAX_ITERATIONS = 1_000
CATBOOST_LOSS_FUNCTION = "Logloss"


def extra_trees_estimator(params: dict[str, Any], seed: int):
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=int(params.get("n_estimators", 120)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        random_state=seed,
        n_jobs=1,
    )


def lightgbm_estimator(params: dict[str, Any], seed: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("missing dependency: lightgbm is required for lightgbm model family") from exc
    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        num_leaves=int(params.get("num_leaves", 31)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        random_state=seed,
        n_jobs=1,
        verbose=-1,
        force_col_wise=LIGHTGBM_FORCE_COL_WISE,
    )


def catboost_estimator(params: dict[str, Any], seed: int):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("missing dependency: catboost is required for catboost model family") from exc
    return CatBoostClassifier(
        iterations=int(params.get("iterations", 100)),
        depth=int(params.get("depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        loss_function=CATBOOST_LOSS_FUNCTION,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )


def logistic_elasticnet_estimator(params: dict[str, Any], seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=float(params.get("C", 1.0)),
        l1_ratio=float(params.get("l1_ratio", 0.5)),
        max_iter=int(params.get("max_iter", LOGISTIC_MAX_ITERATIONS)),
        random_state=seed,
        n_jobs=1,
    )
