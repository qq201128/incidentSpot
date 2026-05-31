from __future__ import annotations

from pathlib import Path

import joblib

from app.services.model_family_candidate_search_service import model_training_config_for_profile
from app.services.model_family_config import model_family_label
from app.services.model_family_joblib_backend import (
    JoblibModelOptions,
    QLearningDirectionClassifier,
    QTableDirectionClassifier,
    _estimator,
)
from app.services.strategy_registry import strategy_definition


def test_rl_strategy_is_labeled_as_qtable_direction_classifier() -> None:
    strategy = strategy_definition("factor_rl_strategy_shadow_10m")

    assert model_family_label("rl_strategy") == "QTableDirection"
    assert "QTable方向分类器" in strategy.name
    assert "QTable方向分类器" in strategy.description


def test_rl_strategy_backend_uses_qtable_classifier_name() -> None:
    config = model_training_config_for_profile("rl_strategy", "BTCUSDT", "10m", profile="fast")
    model = _estimator(JoblibModelOptions(config.family, config.seed, config.params))
    assert isinstance(model, QTableDirectionClassifier)
    assert model.model_kind == "q_table_direction_classifier"


def test_rl_strategy_joblib_artifact_loads_legacy_classifier_name() -> None:
    model_path = Path(__file__).resolve().parents[1] / "models/ml/rl_strategy/BTCUSDT/60m/model.joblib"
    if not model_path.is_file():
        return
    model = joblib.load(model_path)
    assert QLearningDirectionClassifier is QTableDirectionClassifier
    assert isinstance(model, QTableDirectionClassifier)
