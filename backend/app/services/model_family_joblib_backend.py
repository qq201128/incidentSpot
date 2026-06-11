from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from app.services.prediction_timing import record_timing
from app.services.model_family_joblib_extra_estimators import (
    catboost_estimator,
    extra_trees_estimator,
    lightgbm_estimator,
    logistic_elasticnet_estimator,
)

XGBOOST_EARLY_STOPPING_ROUNDS = 20
XGBOOST_TREE_METHOD = "hist"
SVM_ALPHA_MIN = 1e-6
SVM_REGULARIZATION_SCALE = 10_000
SVM_RBF_COMPONENTS = 256
SVM_SGD_MAX_ITERATIONS = 1_000
SVM_SGD_TOLERANCE = 1e-3


@dataclass(frozen=True)
class JoblibModelOptions:
    family: str
    seed: int
    params: dict[str, Any]


class JoblibModelBackend:
    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        options: JoblibModelOptions,
        model_path: Path,
        persist_model: bool = True,
    ) -> dict[str, Any]:
        model = _estimator(options)
        _fit_estimator(model, train_x, train_y, val_x, val_y, options)
        self._trained_model = model
        if persist_model:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
        return {"trainLoss": None, "valLoss": None}

    def predict_trained(self, x: np.ndarray) -> np.ndarray:
        model = getattr(self, "_trained_model", None)
        if model is None:
            raise RuntimeError("joblib model has not been trained")
        return _predict_model(model, x)

    def predict(self, model_path: Path, x: np.ndarray, *, timings: dict[str, Any] | None = None) -> np.ndarray:
        started = time.perf_counter()
        model = joblib.load(model_path)
        record_timing(timings, "modelLoadSeconds", started)
        started = time.perf_counter()
        result = _predict_model(model, x)
        record_timing(timings, "modelPredictSeconds", started)
        return result


def _estimator(options: JoblibModelOptions):
    for family, factory in _estimator_factories():
        if options.family == family:
            return factory(options)
    raise ValueError(f"unsupported joblib model family: {options.family}")


def _estimator_factories():
    return (
        ("random_forest", _random_forest),
        ("extra_trees", lambda options: extra_trees_estimator(options.params, options.seed)),
        ("xgboost", _xgboost),
        ("lightgbm", lambda options: lightgbm_estimator(options.params, options.seed)),
        ("catboost", lambda options: catboost_estimator(options.params, options.seed)),
        ("logistic_elasticnet", lambda options: logistic_elasticnet_estimator(options.params, options.seed)),
        ("svm", _svm),
        ("bayesian", _bayesian),
        ("knn", _knn),
        ("rl_strategy", _rl_strategy),
    )


def _fit_estimator(model, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, options: JoblibModelOptions) -> None:
    train_labels = train_y.astype(int)
    if options.family == "rl_strategy":
        model.fit(train_x, train_labels)
        return
    train_flat = _flat(train_x)
    if options.family != "xgboost":
        model.fit(train_flat, train_labels)
        return
    model.fit(
        train_flat,
        train_labels,
        eval_set=[(_flat(val_x), val_y.astype(int))],
        verbose=False,
    )


def _random_forest(options: JoblibModelOptions):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=int(options.params.get("n_estimators", 120)),
        max_depth=options.params.get("max_depth"),
        min_samples_leaf=int(options.params.get("min_samples_leaf", 2)),
        random_state=options.seed,
        n_jobs=1,
    )


def _xgboost(options: JoblibModelOptions):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("missing dependency: xgboost is required for xgboost model family") from exc
    return XGBClassifier(
        n_estimators=int(options.params.get("n_estimators", 80)),
        max_depth=int(options.params.get("max_depth", 3)),
        learning_rate=float(options.params.get("learning_rate", 0.08)),
        subsample=float(options.params.get("subsample", 0.9)),
        colsample_bytree=float(options.params.get("colsample_bytree", 0.9)),
        random_state=options.seed,
        eval_metric="logloss",
        tree_method=XGBOOST_TREE_METHOD,
        n_jobs=1,
        early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
    )

def _svm(options: JoblibModelOptions):
    from sklearn.pipeline import make_pipeline

    kernel = str(options.params.get("kernel", "rbf"))
    classifier = _sgd_hinge_svm(options)
    if kernel == "linear":
        return classifier
    gamma = options.params.get("gamma", "scale")
    sampler = AutoGammaRBFSampler(gamma=gamma, n_components=SVM_RBF_COMPONENTS, random_state=options.seed)
    return make_pipeline(sampler, classifier)


def _sgd_hinge_svm(options: JoblibModelOptions):
    from sklearn.linear_model import SGDClassifier

    c_value = float(options.params.get("C", 1.0))
    alpha = max(SVM_ALPHA_MIN, 1.0 / (c_value * SVM_REGULARIZATION_SCALE))
    return SGDClassifier(
        loss="hinge",
        alpha=alpha,
        max_iter=SVM_SGD_MAX_ITERATIONS,
        tol=SVM_SGD_TOLERANCE,
        random_state=options.seed,
        average=True,
    )


def _bayesian(options: JoblibModelOptions):
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB(var_smoothing=float(options.params.get("var_smoothing", 1e-9)))


def _knn(options: JoblibModelOptions):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(
        n_neighbors=int(options.params.get("n_neighbors", 7)),
        weights=str(options.params.get("weights", "distance")),
        n_jobs=1,
    )


def _rl_strategy(options: JoblibModelOptions):
    return QTableDirectionClassifier(
        state_bins=int(options.params.get("state_bins", 5)),
        alpha=float(options.params.get("alpha", 0.2)),
        gamma=float(options.params.get("gamma", 0.8)),
        epsilon=float(options.params.get("epsilon", 0.1)),
        episodes=int(options.params.get("episodes", 20)),
        seed=options.seed,
    )


class AutoGammaRBFSampler(BaseEstimator, TransformerMixin):
    def __init__(self, gamma, n_components: int, random_state: int) -> None:
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, x: np.ndarray, y=None):
        from sklearn.kernel_approximation import RBFSampler

        gamma = self._resolved_gamma(x)
        self.sampler_ = RBFSampler(gamma=gamma, n_components=self.n_components, random_state=self.random_state)
        self.sampler_.fit(x, y)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.sampler_.transform(x)

    def _resolved_gamma(self, x: np.ndarray):
        if self.gamma == "auto":
            return 1.0 / x.shape[1]
        return self.gamma


class QTableDirectionClassifier:
    model_kind = "q_table_direction_classifier"

    def __init__(self, state_bins: int, alpha: float, gamma: float, epsilon: float, episodes: int, seed: int) -> None:
        self.state_bins = state_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        features = _rl_features(x)
        self.edges_ = _quantile_edges(features, self.state_bins)
        states = self._states(features)
        self.q_ = np.zeros((int(states.max()) + 1, 2), dtype=np.float32)
        for _episode in range(self.episodes):
            for state, label in zip(states, y.astype(int)):
                action = int(rng.integers(0, 2)) if rng.random() < self.epsilon else int(np.argmax(self.q_[state]))
                reward = 1.0 if action == int(label) else -1.0
                target = reward + self.gamma * float(np.max(self.q_[state]))
                self.q_[state, action] += self.alpha * (target - self.q_[state, action])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        states = self._states(_rl_features(x))
        logits = self.q_[np.minimum(states, self.q_.shape[0] - 1)]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _states(self, x: np.ndarray) -> np.ndarray:
        codes = [np.searchsorted(edge, x[:, idx], side="right") for idx, edge in enumerate(self.edges_)]
        state = np.zeros(x.shape[0], dtype=np.int64)
        for code in codes:
            state = state * self.state_bins + np.minimum(code, self.state_bins - 1)
        return state


# Backward compatibility: older artifacts pickled this name before the rename.
QLearningDirectionClassifier = QTableDirectionClassifier


def _flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)


def _predict_model(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, QTableDirectionClassifier):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    flat = _flat(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(flat)[:, 1].astype(np.float32)
    decision = model.decision_function(flat)
    return _sigmoid(decision).astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    negative_exp = np.exp(value[~positive])
    result = np.empty_like(value, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result

def _rl_features(x: np.ndarray) -> np.ndarray:
    values = x.reshape(x.shape[0], -1).astype(np.float32)
    last = x[:, -1, :].astype(np.float32)
    return np.column_stack(
        [
            values.mean(axis=1),
            values.std(axis=1),
            last.mean(axis=1),
            last.std(axis=1),
        ]
    ).astype(np.float32)


def _quantile_edges(x: np.ndarray, bins: int) -> list[np.ndarray]:
    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    return [np.quantile(x[:, idx], quantiles).astype(np.float32) for idx in range(x.shape[1])]
