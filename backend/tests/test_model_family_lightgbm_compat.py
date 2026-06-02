from __future__ import annotations

import sys
import types

from app.services.model_family_joblib_extra_estimators import lightgbm_estimator


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


def _check_array(_value, *, ensure_all_finite=True):
    return ensure_all_finite


class _FakeLGBMClassifier:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
