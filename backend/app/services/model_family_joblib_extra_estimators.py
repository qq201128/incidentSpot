from __future__ import annotations

import inspect
from typing import Any

LIGHTGBM_FORCE_COL_WISE = True
LIGHTGBM_VALIDATION_HOOKS = ("_LGBMCheckXY", "_LGBMCheckArray")
LOGISTIC_ALPHA_MIN = 1e-6
LOGISTIC_REGULARIZATION_SCALE = 10_000
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1e-3
LOGISTIC_EARLY_STOPPING_FRACTION = 0.1
LOGISTIC_NO_CHANGE_ITERATIONS = 5
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
    _patch_lightgbm_sklearn_validation_compat()
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


def _patch_lightgbm_sklearn_validation_compat() -> None:
    import lightgbm.sklearn as lgb_sklearn

    for hook_name in LIGHTGBM_VALIDATION_HOOKS:
        hook = getattr(lgb_sklearn, hook_name, None)
        if hook is None:
            continue
        setattr(lgb_sklearn, hook_name, _sklearn_finite_arg_adapter(hook))


def _sklearn_finite_arg_adapter(func):
    signature = inspect.signature(func)
    if "force_all_finite" in signature.parameters or "ensure_all_finite" not in signature.parameters:
        return func
    if getattr(func, "_incident_spot_finite_arg_adapter", False):
        return func

    def wrapped(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return func(*args, **kwargs)

    wrapped._incident_spot_finite_arg_adapter = True
    return wrapped


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
    from sklearn.linear_model import SGDClassifier

    c_value = float(params.get("C", 1.0))
    alpha = max(LOGISTIC_ALPHA_MIN, 1.0 / (c_value * LOGISTIC_REGULARIZATION_SCALE))
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=float(params.get("l1_ratio", 0.5)),
        max_iter=int(params.get("max_iter", LOGISTIC_MAX_ITERATIONS)),
        tol=LOGISTIC_TOLERANCE,
        random_state=seed,
        average=True,
        early_stopping=True,
        validation_fraction=LOGISTIC_EARLY_STOPPING_FRACTION,
        n_iter_no_change=LOGISTIC_NO_CHANGE_ITERATIONS,
    )
