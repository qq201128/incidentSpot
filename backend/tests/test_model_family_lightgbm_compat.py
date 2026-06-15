from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from app.services.model_family_joblib_backend import _predict_model
from app.services.model_family_joblib_backend import JoblibModelBackend
from app.services.model_family_joblib_extra_estimators import lightgbm_estimator
from app.services.model_family_prediction_runtime import predict_backend


def test_lightgbm_estimator_ignores_missing_private_validation_hooks(monkeypatch) -> None:
    lightgbm = types.ModuleType("lightgbm")
    lightgbm.__path__ = []
    sklearn = types.ModuleType("lightgbm.sklearn")
    sklearn._LGBMCheckArray = _check_array
    lightgbm.LGBMClassifier = _FakeLGBMClassifier
    lightgbm.sklearn = sklearn
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
    monkeypatch.setitem(sys.modules, "lightgbm.sklearn", sklearn)

    model = lightgbm_estimator({}, 20260513)

    assert model.__class__.__name__ == "_FakeLGBMClassifier"
    assert not hasattr(sklearn, "_LGBMCheckXY")
    assert sklearn._LGBMCheckArray(None, force_all_finite=False) is False


def test_lightgbm_predict_path_patches_loaded_model_validation_hooks(monkeypatch) -> None:
    lightgbm = types.ModuleType("lightgbm")
    lightgbm.__path__ = []
    sklearn = types.ModuleType("lightgbm.sklearn")
    sklearn._LGBMCheckArray = _check_array
    lightgbm.sklearn = sklearn
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
    monkeypatch.setitem(sys.modules, "lightgbm.sklearn", sklearn)

    result = _predict_model(_FakeLoadedLGBMClassifier(), np.ones((1, 2, 3), dtype=np.float32))

    assert result.tolist() == [0.75]


def test_lightgbm_default_backend_entry_patches_validation_hooks(monkeypatch) -> None:
    lightgbm = types.ModuleType("lightgbm")
    lightgbm.__path__ = []
    sklearn = types.ModuleType("lightgbm.sklearn")
    sklearn._LGBMCheckArray = _check_array
    lightgbm.sklearn = sklearn
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
    monkeypatch.setitem(sys.modules, "lightgbm.sklearn", sklearn)

    result = predict_backend(
        "lightgbm",
        Path("model.joblib"),
        np.ones((1, 2, 3), dtype=np.float32),
        None,
        lambda _family: _FakeRuntimeBackend(),
        {},
    )

    assert result.tolist() == [0.75]


def test_joblib_backend_reuses_loaded_model_for_same_path(monkeypatch) -> None:
    loads = []

    def fake_load(path):
        loads.append(path)
        return _FakeProbabilityModel()

    monkeypatch.setattr("app.services.model_family_joblib_backend.joblib.load", fake_load)
    backend = JoblibModelBackend()
    x = np.ones((1, 2, 3), dtype=np.float32)

    first = backend.predict(Path("model.joblib"), x)
    second = backend.predict(Path("model.joblib"), x)

    assert first.tolist() == [0.75]
    assert second.tolist() == [0.75]
    assert loads == [Path("model.joblib")]


def _check_array(_value, *, ensure_all_finite=True):
    return ensure_all_finite


class _FakeLGBMClassifier:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeLoadedLGBMClassifier:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import lightgbm.sklearn as lgb_sklearn

        assert lgb_sklearn._LGBMCheckArray(x, force_all_finite=False) is False
        return np.array([[0.25, 0.75]], dtype=np.float32)


_FakeLoadedLGBMClassifier.__module__ = "lightgbm.sklearn"


class _FakeProbabilityModel:
    def predict_proba(self, _x: np.ndarray) -> np.ndarray:
        return np.array([[0.25, 0.75]], dtype=np.float32)


class _FakeRuntimeBackend:
    def predict(self, _model_path, x: np.ndarray, *, timings=None) -> np.ndarray:
        import lightgbm.sklearn as lgb_sklearn

        assert timings == {}
        assert lgb_sklearn._LGBMCheckArray(x, force_all_finite=False) is False
        return np.array([0.75], dtype=np.float32)
