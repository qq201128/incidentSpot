from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from app.services import lstm_combo_ranking
from app.services import lstm_combo_sync_service as combo_sync
from app.services.lstm_combo_sync_service import (
    LSTM_SYNC_TRAINED,
    sync_lstm_model_to_combo_ranking,
)
from app.services.lstm_config import LstmTrainingConfig


def test_sync_lstm_model_uses_high_winrate_when_primary_is_stale(
    monkeypatch,
) -> None:
    artifact_root = _runtime_path("stale-high-winrate")
    primary = _stale_ranking("combo_primary")
    high = _ranking("combo_high")
    captured = {}

    def fake_build_dataset(config: LstmTrainingConfig, *, ranking_loader):
        captured["ranking"] = ranking_loader(config.symbol, config.duration)
        return object()

    def fake_train(config: LstmTrainingConfig, *, artifact_root=None, dataset_builder):
        captured["artifactRoot"] = artifact_root
        dataset_builder(config)
        return {"status": LSTM_SYNC_TRAINED, "modelVersion": "lstm_new"}

    monkeypatch.setattr(combo_sync, "build_lstm_training_dataset", fake_build_dataset)
    monkeypatch.setattr(combo_sync, "train_lstm_model", fake_train)
    monkeypatch.setattr(lstm_combo_ranking, "get_cached_high_winrate_combo_ranking", lambda *_args: high)

    result = sync_lstm_model_to_combo_ranking(
        "BTCUSDT",
        "10m",
        ranking_report=primary,
        artifact_root=artifact_root,
    )

    assert result["status"] == LSTM_SYNC_TRAINED
    assert captured["artifactRoot"] == artifact_root
    assert captured["ranking"]["ranking"][0]["factorName"] == "combo_high"
    assert (
        captured["ranking"]["lstmComboRankingSource"]
        == lstm_combo_ranking.LSTM_COMBO_SOURCE_HIGH_WINRATE
    )


def test_sync_lstm_model_rebuilds_stale_primary_with_default_dataset_builder(
    monkeypatch,
) -> None:
    artifact_root = _runtime_path("stale-primary-rebuild")
    primary = _stale_ranking("combo_primary")
    captured = {}

    def fake_train(
        config: LstmTrainingConfig,
        *,
        artifact_root=None,
        **kwargs,
    ) -> dict[str, Any]:
        captured["symbol"] = config.symbol
        captured["artifactRoot"] = artifact_root
        captured["datasetBuilderInjected"] = "dataset_builder" in kwargs
        return {"status": LSTM_SYNC_TRAINED, "modelVersion": "lstm_new"}

    monkeypatch.setattr(combo_sync, "train_lstm_model", fake_train)
    monkeypatch.setattr(lstm_combo_ranking, "get_cached_high_winrate_combo_ranking", lambda *_args: None)

    result = sync_lstm_model_to_combo_ranking(
        "btcusdt",
        "10m",
        ranking_report=primary,
        artifact_root=artifact_root,
    )

    assert result["status"] == LSTM_SYNC_TRAINED
    assert captured["symbol"] == "BTCUSDT"
    assert captured["artifactRoot"] == artifact_root
    assert captured["datasetBuilderInjected"] is False


def _ranking(factor_name: str) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [
            {"factorName": factor_name, "members": [{"name": "factor_a"}]},
            {"factorName": "combo_top2", "members": [{"name": "factor_b"}]},
            {"factorName": "combo_top3", "members": [{"name": "factor_c"}]},
        ],
    }


def _stale_ranking(factor_name: str) -> dict[str, Any]:
    return {
        **_ranking(factor_name),
        "cacheStatus": {"usable": False, "reason": "market_data_changed"},
    }


def _runtime_path(name: str) -> Path:
    base = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp"
    path = base / f"lstm-combo-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
