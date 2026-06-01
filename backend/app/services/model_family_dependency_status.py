from __future__ import annotations

import importlib
from typing import Any

from app.services.lstm_torch_backend import torch_availability
from app.services.model_family_config import JOBLIB_MODEL_FAMILIES, TORCH_MODEL_FAMILIES

_DEPENDENCY_STATUS_CACHE: dict[str, dict[str, Any]] = {}
_OPTIONAL_DEPENDENCY_MODULES = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def dependency_status(family: str) -> dict[str, Any]:
    cached = _DEPENDENCY_STATUS_CACHE.get(family)
    if cached is not None:
        return cached
    payload = _dependency_status_uncached(family)
    _DEPENDENCY_STATUS_CACHE[family] = payload
    return payload


def _dependency_status_uncached(family: str) -> dict[str, Any]:
    if family in TORCH_MODEL_FAMILIES:
        return torch_availability()
    if family in _OPTIONAL_DEPENDENCY_MODULES:
        return _optional_dependency_status(_OPTIONAL_DEPENDENCY_MODULES[family])
    return {"available": family in JOBLIB_MODEL_FAMILIES}


def _optional_dependency_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, "version": getattr(module, "__version__", None)}
