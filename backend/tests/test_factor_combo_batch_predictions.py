from __future__ import annotations

from app.services import factor_combo_batch_predictions as service


def test_offline_screening_reports_focus_pool_and_rejected_reasons(monkeypatch) -> None:
    rows = [_ranking_row(index) for index in range(service.OBSERVATION_POOL_LIMIT + 1)]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "materialize_factor_combo_frame_for_row", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)

    report = service.offline_candidate_screening_report("btcusdt", "10m")

    assert report["focusedCount"] == service.OBSERVATION_POOL_LIMIT
    assert report["policy"] == "offline_cross_period_stability_sample_size_profit_factor_prefilter_only"
    assert report["rankingPolicy"] == ["cross_period_stability", "sample_count", "profit_factor"]
    assert report["candidateCount"] == service.OBSERVATION_POOL_LIMIT + 1
    assert report["rejectedCount"] == 1
    assert report["rejectedReasons"][0]["reason"] == "outside_observation_pool_limit"
    assert report["focusedCandidates"][0]["factorName"] == f"combo_{service.OBSERVATION_POOL_LIMIT}"


def test_offline_screening_ranks_stability_before_sample_size_before_profit_factor(monkeypatch) -> None:
    rows = [
        _ranking_row(1, stability=0.80, total_periods=100, profit_factor=1.1),
        _ranking_row(2, stability=0.79, total_periods=1000, profit_factor=5.0),
        _ranking_row(3, stability=0.80, total_periods=200, profit_factor=1.1),
        _ranking_row(4, stability=0.80, total_periods=100, profit_factor=5.0),
    ]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "materialize_factor_combo_frame_for_row", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)

    report = service.offline_candidate_screening_report("BTCUSDT", "10m")

    assert [row["factorName"] for row in report["focusedCandidates"]] == [
        "combo_3",
        "combo_4",
        "combo_1",
        "combo_2",
    ]


def test_backtest_qualified_combo_rows_use_shared_backtest_gate(monkeypatch) -> None:
    rows = [
        {**_ranking_row(1), "winRate": 0.63, "profitFactor": 1.1, "totalPeriods": 120},
        {**_ranking_row(2), "winRate": 0.50, "profitFactor": 1.1, "totalPeriods": 120},
    ]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])

    qualified = service.backtest_qualified_factor_combo_rows("BTCUSDT", "10m")

    assert [row["factorName"] for row in qualified] == ["combo_1"]


def test_prediction_rows_use_offline_focused_pool(monkeypatch) -> None:
    focused = [{**_ranking_row(9), "totalPeriods": 120}]
    predicted = []
    monkeypatch.setattr(service, "eligible_factor_combo_rows", lambda *_args: focused)
    monkeypatch.setattr(
        service,
        "predict_factor_combo_row_direction",
        lambda _symbol, _duration, row, **_kwargs: predicted.append(row["factorName"]) or _prediction(row),
    )

    rows = service.predict_eligible_factor_combo_rows(
        "BTCUSDT",
        "10m",
        entry_open_time=1_700_000_000_000,
        entry_grace_ms=60_000,
    )

    assert predicted == ["combo_9"]
    assert rows[0]["strategy_key"] == "paper_combo_9"
    assert rows[0]["trade_quality_passed"] is True


def test_prediction_rows_only_use_ranked_offline_focused_pool(monkeypatch) -> None:
    rows = [_ranking_row(index) for index in range(service.OBSERVATION_POOL_LIMIT + 1)]
    predicted = []
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "materialize_factor_combo_frame_for_row", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)
    monkeypatch.setattr(
        service,
        "predict_factor_combo_row_direction",
        lambda _symbol, _duration, row, **_kwargs: predicted.append(row["factorName"]) or _prediction(row),
    )

    result = service.predict_eligible_factor_combo_rows(
        "BTCUSDT",
        "10m",
        entry_open_time=1_700_000_000_000,
        entry_grace_ms=60_000,
    )

    assert len(result) == service.OBSERVATION_POOL_LIMIT
    assert predicted == [f"combo_{index}" for index in range(service.OBSERVATION_POOL_LIMIT, 0, -1)]
    assert "combo_0" not in predicted


def test_prediction_rows_log_and_skip_row_level_failures(monkeypatch) -> None:
    focused = [_ranking_row(1), _ranking_row(2)]
    failures = []
    monkeypatch.setattr(service, "eligible_factor_combo_rows", lambda *_args: focused)
    monkeypatch.setattr(service, "log_prediction_failure", lambda **kwargs: failures.append(kwargs))

    def _predict(_symbol, _duration, row, **_kwargs) -> dict:
        if row["factorName"] == "combo_1":
            raise ValueError("combination signal missing factors: missing_factor")
        return _prediction(row)

    monkeypatch.setattr(service, "predict_factor_combo_row_direction", _predict)

    rows = service.predict_eligible_factor_combo_rows(
        "BTCUSDT",
        "10m",
        entry_open_time=1_700_000_000_000,
        entry_grace_ms=60_000,
    )

    assert [row["strategy_key"] for row in rows] == ["paper_combo_2"]
    assert failures == [
        {
            "candidate_key": "combo_1",
            "strategy_key": service.FACTOR_COMBO_STRATEGY_KEY,
            "symbol": "BTCUSDT",
            "duration": "10m",
            "stage": "factor_combo_shadow_prediction_row",
            "reason": "combination signal missing factors: missing_factor",
            "details": {
                "entryOpenTime": 1_700_000_000_000,
                "comboRank": None,
                "exceptionType": "ValueError",
            },
        }
    ]


def test_prediction_rows_still_raise_system_level_failures(monkeypatch) -> None:
    focused = [_ranking_row(1)]
    monkeypatch.setattr(service, "eligible_factor_combo_rows", lambda *_args: focused)

    def _predict(*_args, **_kwargs) -> dict:
        raise ValueError("factor combination ranking BTCUSDT 10m cache is stale: expired")

    monkeypatch.setattr(service, "predict_factor_combo_row_direction", _predict)

    try:
        service.predict_eligible_factor_combo_rows(
            "BTCUSDT",
            "10m",
            entry_open_time=1_700_000_000_000,
            entry_grace_ms=60_000,
        )
    except ValueError as exc:
        assert "cache is stale" in str(exc)
    else:
        raise AssertionError("expected stale cache failure to be raised")


def test_offline_screening_reports_no_usable_cache_reason(monkeypatch) -> None:
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [])

    report = service.offline_candidate_screening_report("BTCUSDT", "10m")

    assert report["focusedCandidates"] == []
    assert report["rejectedReasons"][0]["reason"] == "no_usable_offline_candidate_cache"


def test_offline_screening_rejects_rows_with_non_finite_live_score(monkeypatch) -> None:
    rows = [_ranking_row(1), _ranking_row(2)]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "materialize_factor_combo_frame_for_row", lambda *_args, **_kwargs: object())

    def _signal(_frame, row, **_kwargs) -> dict:
        if row["factorName"] == "combo_1":
            raise ValueError("combination signal has no finite score at 10m entry: combo_1")
        return {"qualityPassed": True}

    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)

    report = service.offline_candidate_screening_report("BTCUSDT", "10m")

    assert [row["factorName"] for row in report["focusedCandidates"]] == ["combo_2"]
    assert report["rejectedReasons"][0]["factorName"] == "combo_1"
    assert "no finite score" in report["rejectedReasons"][0]["reason"]


def test_offline_screening_materializes_each_candidate_row(monkeypatch) -> None:
    rows = [_ranking_row(1), _ranking_row(2)]
    source_rows = [{"factorName": "source_combo"}]
    materialized = []
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(service, "mined_factor_rows_for_duration", lambda *_args: source_rows)

    def _materialize(_frame, **kwargs) -> object:
        materialized.append((kwargs["row"]["factorName"], kwargs["source_rows"]))
        return object()

    monkeypatch.setattr(service, "materialize_factor_combo_frame_for_row", _materialize)
    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)

    report = service.offline_candidate_screening_report("BTCUSDT", "10m")

    assert report["focusedCount"] == 2
    assert materialized == [("combo_1", source_rows), ("combo_2", source_rows)]


def test_offline_screening_rejects_materialization_failure(monkeypatch) -> None:
    rows = [_ranking_row(1)]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args, **_kwargs: object())

    def _materialize(_frame, **_kwargs) -> object:
        raise ValueError("factor combo materialization failed: combo_1")

    monkeypatch.setattr(service, "materialize_factor_combo_frame_for_row", _materialize)
    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)

    report = service.offline_candidate_screening_report("BTCUSDT", "10m")

    assert report["focusedCandidates"] == []
    assert report["rejectedReasons"][0]["factorName"] == "combo_1"
    assert "materialization failed" in report["rejectedReasons"][0]["reason"]


def _ranking_row(
    index: int,
    *,
    stability: float | None = None,
    total_periods: int | None = None,
    profit_factor: float | None = None,
) -> dict:
    return {
        "factorName": f"combo_{index}",
        "winRate": 0.63 + index / 1000,
        "profitFactor": profit_factor if profit_factor is not None else 1.2 + index / 100,
        "totalPeriods": total_periods if total_periods is not None else 120,
        "walkForward": {"stabilityScore": stability if stability is not None else index, "oosWinRate": 0.6},
        "recentRollingResult": {"winRate": 0.61},
        "walkForwardPassed": True,
    }


def _signal(_frame, _row, **_kwargs) -> dict:
    return {"qualityPassed": True}


def _prediction(row: dict) -> dict:
    return {
        "strategy_key": f"paper_{row['factorName']}",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_700_000_000_000,
        "direction": "up",
        "probability_up": 0.7,
        "confidence": 0.7,
        "certainty_label": "high",
        "trade_quality_passed": False,
    }
