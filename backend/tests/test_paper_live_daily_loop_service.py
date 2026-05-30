from __future__ import annotations

from app.services.paper_live_daily_loop_service import (
    PaperLiveDailyLoopDeps,
    run_paper_live_daily_closed_loop,
)


def test_daily_closed_loop_reports_all_required_steps() -> None:
    calls = []

    def refresh(symbol: str, duration: str, *, run_learning_agent: bool) -> None:
        calls.append(("refresh", symbol, duration, run_learning_agent))

    deps = PaperLiveDailyLoopDeps(
        symbols=lambda: ["BTCUSDT"],
        refresh_candidates=refresh,
        settle_predictions=lambda symbol, duration: {"checked": 2, "settled": 1, "pendingData": 1},
        refresh_states=lambda symbol, duration: _candidate_report(symbol, duration),
        candidate_report=lambda symbol, duration: _candidate_report(symbol, duration),
        offline_screening=lambda symbol, duration: _offline_screening(symbol, duration),
        model_candidates=lambda symbol, duration: _model_candidates(symbol, duration),
    )

    report = run_paper_live_daily_closed_loop(symbols=["btcusdt"], durations=["10m"], deps=deps)
    result = report["results"][0]

    assert report["status"] == "passed"
    assert report["dailyTaskCount"] == 11
    assert calls == [("refresh", "BTCUSDT", "10m", True)]
    assert len(result["dailyChecklist"]) == 11
    assert result["dailyChecklist"][4]["task"] == "generate_real_time_predictions"
    assert result["dailyChecklist"][4]["status"] == "delegated"
    assert result["candidatePool"]["avoidNextSearch"][0]["reason"] == "paper_live_win_rate_below_target"
    assert result["realTimePredictionExecutor"]["realTradingEnabled"] is False
    offline = result["stages"][0]["payload"]["offlineScreening"]
    assert offline["rejectedReasons"][0]["reason"] == "outside_observation_pool_limit"
    models = result["stages"][0]["payload"]["modelCandidates"]
    assert models["paperLiveReadyCount"] == 1
    assert models["models"][0]["paperLiveStatus"] == "paper_collecting"


def test_daily_closed_loop_runs_default_btc_and_eth_symbols() -> None:
    calls = []

    def refresh(symbol: str, duration: str, *, run_learning_agent: bool) -> None:
        calls.append((symbol, duration, run_learning_agent))

    deps = PaperLiveDailyLoopDeps(
        symbols=lambda: ["BTCUSDT", "ETHUSDT"],
        refresh_candidates=refresh,
        settle_predictions=lambda symbol, duration: {"checked": 0, "settled": 0, "pendingData": 0},
        refresh_states=lambda symbol, duration: _candidate_report(symbol, duration),
        candidate_report=lambda symbol, duration: _candidate_report(symbol, duration),
        offline_screening=lambda symbol, duration: _offline_screening(symbol, duration),
        model_candidates=lambda symbol, duration: _model_candidates(symbol, duration),
    )

    report = run_paper_live_daily_closed_loop(durations=["10m"], deps=deps)

    assert report["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert [(row["symbol"], row["duration"]) for row in report["results"]] == [
        ("BTCUSDT", "10m"),
        ("ETHUSDT", "10m"),
    ]
    assert calls == [("BTCUSDT", "10m", True), ("ETHUSDT", "10m", True)]


def test_daily_closed_loop_surfaces_stage_failure() -> None:
    def refresh(_symbol: str, _duration: str, *, run_learning_agent: bool) -> None:
        raise RuntimeError("market refresh failed")

    deps = PaperLiveDailyLoopDeps(
        symbols=lambda: ["BTCUSDT"],
        refresh_candidates=refresh,
        settle_predictions=lambda _symbol, _duration: {"checked": 0, "settled": 0, "pendingData": 0},
        refresh_states=lambda symbol, duration: _candidate_report(symbol, duration),
        candidate_report=lambda symbol, duration: _candidate_report(symbol, duration),
        offline_screening=lambda symbol, duration: _offline_screening(symbol, duration),
        model_candidates=lambda symbol, duration: _model_candidates(symbol, duration),
    )

    report = run_paper_live_daily_closed_loop(symbols=["BTCUSDT"], durations=["10m"], deps=deps)
    first_stage = report["results"][0]["stages"][0]

    assert report["status"] == "failed"
    assert first_stage["status"] == "failed"
    assert first_stage["reason"] == "market refresh failed"
    assert first_stage["exceptionType"] == "RuntimeError"
    assert "RuntimeError: market refresh failed" in first_stage["traceback"]


def _candidate_report(symbol: str, duration: str) -> dict:
    return {
        "symbol": symbol,
        "duration": duration,
        "observationPoolLimit": 10,
        "stable": [{"candidateKey": "stable_a", "status": "paper_stable"}],
        "collecting": [{"candidateKey": "collecting_a", "status": "paper_collecting"}],
        "failed": [{"candidateKey": "failed_a", "status": "paper_failed"}],
        "predictionFailures": [{"candidateKey": "failed_b", "reason": "missing_feature"}],
        "avoidNextSearch": [{"candidateKey": "failed_a", "reason": "paper_live_win_rate_below_target"}],
    }


def _offline_screening(symbol: str, duration: str) -> dict:
    return {
        "policy": "offline_oos_walk_forward_recent_rolling_prefilter_only",
        "observationPoolLimit": 10,
        "focusedCount": 1,
        "candidateCount": 2,
        "rejectedCount": 1,
        "rejectedReasons": [{"factorName": "slow_beta", "reason": "outside_observation_pool_limit"}],
        "reasonCounts": {"outside_observation_pool_limit": 1},
    }


def _model_candidates(symbol: str, duration: str) -> dict:
    return {
        "policy": "model_validation_gate_is_paper_live_prefilter_only",
        "familyCount": 1,
        "paperLiveReadyCount": 1,
        "failures": [],
        "models": [
            {
                "modelFamily": "xgboost",
                "modelVersion": "xgboost_v1",
                "paperLiveStatus": "paper_collecting",
                "paperLiveAdmissionAllowed": True,
            }
        ],
        "realTradingEnabled": False,
    }
