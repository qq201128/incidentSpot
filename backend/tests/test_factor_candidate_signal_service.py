from __future__ import annotations

import pandas as pd
import pytest

from app.services import factor_candidate_signal_service as service
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key


def test_predict_factor_candidate_signals_from_ranking_cache(monkeypatch) -> None:
    step_ms = 600_000
    frame = pd.DataFrame(
        {
            "open_time": [index * step_ms for index in range(520)],
            "close": [100 + index for index in range(520)],
            "rsi_14": [40 + index * 0.01 for index in range(520)],
        }
    )
    ranking = [
        {
            "factorName": "rsi_14",
            "factorDisplayName": "RSI",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "category": "momentum",
            "sourceFile": "kline_features.py",
            "direction": "neutral",
            "winRate": 0.61,
            "profitFactor": 1.2,
            "factorScore": 80,
            "backtestValid": True,
            "totalPeriods": 500,
        }
    ]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args: frame)

    predictions = service.predict_factor_candidate_signals(
        "btcusdt",
        "10m",
        entry_open_time=520 * step_ms,
    )

    assert len(predictions) == 1
    assert predictions[0]["signal_key"] == factor_candidate_signal_key("rsi_14")
    assert predictions[0]["strategy_key"] == factor_candidate_signal_key("rsi_14")
    assert predictions[0]["symbol"] == "BTCUSDT"
    assert predictions[0]["high_winrate_rule"] == "rsi_14"
    assert predictions[0]["signal_source"] == "factor_candidate_signal"


def test_predict_factor_candidate_signals_refreshes_required_source_kline(monkeypatch) -> None:
    calls = []
    step_ms = 600_000
    frame = pd.DataFrame(
        {
            "open_time": [index * step_ms for index in range(520)],
            "close": [100 + index for index in range(520)],
            "rsi_14": [40 + index * 0.01 for index in range(520)],
        }
    )
    ranking = [
        {
            "factorName": "rsi_14",
            "factorDisplayName": "RSI",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "category": "momentum",
            "sourceFile": "kline_features.py",
            "direction": "neutral",
            "winRate": 0.61,
            "profitFactor": 1.2,
            "factorScore": 80,
            "backtestValid": True,
            "totalPeriods": 500,
        }
    ]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *args: calls.append(args))
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args: frame)

    service.predict_factor_candidate_signals("btcusdt", "10m", entry_open_time=520 * step_ms)

    assert calls == [
        ("btcusdt", "1m", 520 * step_ms - 60_000),
        ("btcusdt", "10m", 519 * step_ms),
    ]


def test_predict_factor_candidate_signals_keeps_valid_rows_when_one_candidate_fails(monkeypatch) -> None:
    step_ms = 600_000
    frame = pd.DataFrame(
        {
            "open_time": [index * step_ms for index in range(520)],
            "close": [100 + index for index in range(520)],
            "rsi_14": [40 + index * 0.01 for index in range(520)],
            "bad_factor": [None for _index in range(520)],
        }
    )
    ranking = [_ranking_row("bad_factor"), _ranking_row("rsi_14")]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args: frame)

    predictions = service.predict_factor_candidate_signals(
        "btcusdt",
        "10m",
        entry_open_time=520 * step_ms,
    )

    assert [item["signal_key"] for item in predictions] == [factor_candidate_signal_key("rsi_14")]


def test_predict_factor_candidate_signals_raises_when_all_candidates_fail(monkeypatch) -> None:
    step_ms = 600_000
    frame = pd.DataFrame(
        {
            "open_time": [index * step_ms for index in range(520)],
            "close": [100 + index for index in range(520)],
            "bad_factor": [None for _index in range(520)],
        }
    )

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": [_ranking_row("bad_factor")]})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args: frame)

    with pytest.raises(ValueError, match="all factor candidate signals failed"):
        service.predict_factor_candidate_signals("btcusdt", "10m", entry_open_time=520 * step_ms)


def _ranking_row(factor_name: str) -> dict:
    return {
        "factorName": factor_name,
        "factorDisplayName": factor_name,
        "symbol": "BTCUSDT",
        "duration": "10m",
        "category": "momentum",
        "sourceFile": "kline_features.py",
        "direction": "neutral",
        "winRate": 0.61,
        "profitFactor": 1.2,
        "factorScore": 80,
        "backtestValid": True,
        "totalPeriods": 500,
    }
