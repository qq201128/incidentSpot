from __future__ import annotations

import pandas as pd
import pytest

from app.services import factor_candidate_signal_service as service
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key


@pytest.fixture(autouse=True)
def no_agent_factor_rows(monkeypatch) -> None:
    monkeypatch.setattr(service, "agent_factor_rows_for_duration", lambda *_args: [])


def test_predict_factor_candidate_signals_from_ranking_cache(monkeypatch) -> None:
    step_ms = 600_000
    frame = _candidate_frame(step_ms=step_ms)
    ranking = [
        {
            "factorName": "rsi_14",
            "factorDisplayName": "RSI",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "category": "momentum",
            "sourceFile": "kline_features.py",
            "direction": "neutral",
            "winRate": 0.65,
            "profitFactor": 1.2,
            "factorScore": 80,
            "backtestValid": True,
            "totalPeriods": 500,
        }
    ]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "refresh_positioning_features_for_lookback", lambda *_args: 0)
    monkeypatch.setattr(service, "_backfill_duration_klines_if_needed", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: frame)

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
    assert predictions[0]["entry_price"] == pytest.approx(619.0)


def test_predict_factor_candidate_signals_observes_backtest_failed_rows(monkeypatch) -> None:
    step_ms = 600_000
    frame = _candidate_frame(step_ms=step_ms)
    ranking = [{**_ranking_row("rsi_14"), "winRate": 0.55, "profitFactor": 0.9}]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "refresh_positioning_features_for_lookback", lambda *_args: 0)
    monkeypatch.setattr(service, "_backfill_duration_klines_if_needed", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: frame)

    predictions = service.predict_factor_candidate_signals(
        "btcusdt",
        "10m",
        entry_open_time=520 * step_ms,
    )

    assert len(predictions) == 1
    assert predictions[0]["signal_key"] == factor_candidate_signal_key("rsi_14")
    assert predictions[0]["trade_quality_passed"] is False


def test_predict_factor_candidate_signals_refreshes_required_source_kline(monkeypatch) -> None:
    calls = []
    step_ms = 600_000
    frame = _candidate_frame(step_ms=step_ms)
    ranking = [
        {
            "factorName": "rsi_14",
            "factorDisplayName": "RSI",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "category": "momentum",
            "sourceFile": "kline_features.py",
            "direction": "neutral",
            "winRate": 0.65,
            "profitFactor": 1.2,
            "factorScore": 80,
            "backtestValid": True,
            "totalPeriods": 500,
        }
    ]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *args: calls.append(args))
    monkeypatch.setattr(service, "refresh_positioning_features_for_lookback", lambda *_args: 0)
    monkeypatch.setattr(service, "_backfill_duration_klines_if_needed", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: frame)

    service.predict_factor_candidate_signals("btcusdt", "10m", entry_open_time=520 * step_ms)

    lookback_start = 519 * step_ms - (service.CANDIDATE_SCORE_LOOKBACK_BARS - 1) * step_ms
    one_m_lookback_start = 520 * step_ms - service.CANDIDATE_SCORE_LOOKBACK_BARS * 60_000
    assert calls == [
        ("btcusdt", "10m", lookback_start),
        ("btcusdt", "10m", 519 * step_ms),
        ("btcusdt", "1m", one_m_lookback_start),
        ("btcusdt", "1m", 520 * step_ms - 60_000),
    ]


def test_predict_factor_candidate_signals_keeps_valid_rows_when_one_candidate_fails(monkeypatch) -> None:
    step_ms = 600_000
    frame = _candidate_frame(step_ms=step_ms)
    frame["bad_factor"] = [None for _index in range(520)]
    ranking = [_ranking_row("bad_factor"), _ranking_row("rsi_14")]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "refresh_positioning_features_for_lookback", lambda *_args: 0)
    monkeypatch.setattr(service, "_backfill_duration_klines_if_needed", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: frame)

    predictions = service.predict_factor_candidate_signals(
        "btcusdt",
        "10m",
        entry_open_time=520 * step_ms,
    )

    assert [item["signal_key"] for item in predictions] == [factor_candidate_signal_key("rsi_14")]


def test_predict_factor_candidate_signals_raises_when_all_candidates_fail(monkeypatch) -> None:
    step_ms = 600_000
    frame = _candidate_frame(step_ms=step_ms).drop(columns=["rsi_14"])
    frame["bad_factor"] = [None for _index in range(520)]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": [_ranking_row("bad_factor")]})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "refresh_positioning_features_for_lookback", lambda *_args: 0)
    monkeypatch.setattr(service, "_backfill_duration_klines_if_needed", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: frame)

    with pytest.raises(ValueError, match=r"all factor candidate signals failed.*bad_factor"):
        service.predict_factor_candidate_signals("btcusdt", "10m", entry_open_time=520 * step_ms)


def test_predict_factor_candidate_signal_blocks_counter_trend_long(monkeypatch) -> None:
    step_ms = 600_000
    frame = _candidate_frame(step_ms=step_ms, trend="down")
    frame["rsi_14"] = [40.0 + index * 0.01 for index in range(519)] + [70.0]
    ranking = [{**_ranking_row("rsi_14"), "winRate": 0.65}]

    monkeypatch.setattr(service, "get_cached_ranking", lambda *_args: {"ranking": ranking})
    monkeypatch.setattr(service, "refresh_prediction_klines", lambda *_args: None)
    monkeypatch.setattr(service, "refresh_positioning_features_for_lookback", lambda *_args: 0)
    monkeypatch.setattr(service, "_backfill_duration_klines_if_needed", lambda *_args: None)
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: frame)

    prediction = service.predict_factor_candidate_signals(
        "btcusdt",
        "10m",
        entry_open_time=520 * step_ms,
    )[0]

    assert prediction["direction"] == "up"
    assert prediction["trade_quality_passed"] is False
    assert "regime_reason=regime_counter_trend_long" in prediction["rule_reasons"]


def _ranking_row(factor_name: str) -> dict:
    return {
        "factorName": factor_name,
        "factorDisplayName": factor_name,
        "symbol": "BTCUSDT",
        "duration": "10m",
        "category": "momentum",
        "sourceFile": "kline_features.py",
        "direction": "neutral",
        "winRate": 0.65,
        "profitFactor": 1.2,
        "factorScore": 80,
        "backtestValid": True,
        "totalPeriods": 500,
    }


def _candidate_frame(*, step_ms: int, trend: str = "up") -> pd.DataFrame:
    close = _close_values(trend)
    return pd.DataFrame(
        {
            "open_time": [index * step_ms for index in range(520)],
            "open": close,
            "high": [value * 1.001 for value in close],
            "low": [value * 0.999 for value in close],
            "close": close,
            "volume": [100 for _index in range(520)],
            "rsi_14": [40 + index * 0.01 for index in range(520)],
        }
    )


def _close_values(trend: str) -> list[float]:
    step = -1 if trend == "down" else 1
    start = 1000 if trend == "down" else 100
    return [start + index * step for index in range(520)]
