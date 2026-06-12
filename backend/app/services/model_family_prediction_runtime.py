from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.services.lstm_artifacts import require_json
from app.services.model_family_config import JOBLIB_MODEL_FAMILIES
from app.services.model_family_joblib_extra_estimators import ensure_lightgbm_sklearn_validation_compat

FeatureWindowLoader = Callable[..., tuple[np.ndarray, dict[str, Any]]]


@dataclass
class PredictionCycleContext:
    artifact_json: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = field(default_factory=dict)
    feature_windows: dict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]] = field(default_factory=dict)

    def artifacts(self, paths) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        key = str(paths.root)
        if key not in self.artifact_json:
            self.artifact_json[key] = prediction_artifacts(paths)
        return self.artifact_json[key]

    def live_feature_window(
        self,
        loader: FeatureWindowLoader,
        symbol: str,
        duration: str,
        columns: list[str],
        feature_window: int,
        entry_open_time: int | None,
        model_family: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        key = (symbol, duration, int(entry_open_time or 0), tuple(columns), int(feature_window))
        if key not in self.feature_windows:
            self.feature_windows[key] = loader(symbol, duration, columns, feature_window, entry_open_time, model_family=model_family)
        window, meta = self.feature_windows[key]
        return window.copy(), dict(meta)


def prediction_artifacts(paths) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        require_json(paths.features, "features"),
        require_json(paths.scaler, "scaler"),
        require_json(paths.version, "version"),
        require_json(paths.report, "training report"),
    )


def prediction_artifacts_for_context(paths, cycle_context: PredictionCycleContext | None):
    if cycle_context is None:
        return prediction_artifacts(paths)
    return cycle_context.artifacts(paths)


def live_feature_window_for_context(
    loader: FeatureWindowLoader,
    symbol: str,
    duration: str,
    columns: list[str],
    feature_window: int,
    entry: int | None,
    family: str,
    cycle_context: PredictionCycleContext | None,
):
    if cycle_context is None:
        return loader(symbol, duration, columns, feature_window, entry, model_family=family)
    return cycle_context.live_feature_window(loader, symbol, duration, columns, feature_window, entry, family)


def live_feature_windows_for_context(
    loader: FeatureWindowLoader,
    symbol: str,
    duration: str,
    columns: list[str],
    feature_window: int,
    entries: list[int],
    family: str,
    cycle_context: PredictionCycleContext | None,
):
    windows, metas = [], []
    for entry in entries:
        window, meta = live_feature_window_for_context(
            loader, symbol, duration, columns, feature_window, entry, family, cycle_context
        )
        windows.append(window.reshape(feature_window, len(columns)))
        metas.append(meta)
    return np.asarray(windows, dtype=np.float32), metas


def predict_backend(
    family: str,
    model_path: Path,
    window: np.ndarray,
    backend: Any | None,
    default_backend: Callable[[str], Any],
    timings: dict[str, Any] | None,
) -> np.ndarray:
    selected = backend or default_backend(family)
    if backend is not None:
        return selected.predict(model_path, window)
    _prepare_default_prediction_runtime(family)
    if family in JOBLIB_MODEL_FAMILIES:
        return selected.predict(model_path, window, timings=timings)
    return selected.predict(model_path, window)


def _prepare_default_prediction_runtime(family: str) -> None:
    if family == "lightgbm":
        ensure_lightgbm_sklearn_validation_compat()
