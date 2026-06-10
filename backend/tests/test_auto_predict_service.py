from __future__ import annotations

import asyncio

from app.services import auto_predict_input_deps as input_deps
from app.services import auto_predict_service as service
from app.services.auto_predict_loop_status import auto_predict_loop_status
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import (
    factor_combo_shadow_strategy_key,
    simulation_strategy_key_for_factor_name,
)
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_prediction_readiness import PredictionReadiness
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
)
from app.services.event_final_decision_service import EVENT_FINAL_DECISION_STRATEGY_KEY

ASYNC_TEST_TIMEOUT_SECONDS = 1.0
DEFAULT_DURATION = "10m"
DEFAULT_QTY = 5.0
ENTRY_OPEN_TIME = 1778121600000


def test_auto_predict_loop_failure_is_exposed(monkeypatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()

    def fail_targets():
        stop_event.set()
        raise RuntimeError("target load failed")

    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "_prediction_targets", fail_targets)

    asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))

    status = auto_predict_loop_status()
    assert status["status"] == "failed"
    assert status["error"] == "target load failed"
    assert status["exceptionType"] == "RuntimeError"
    assert status["nextWaitSeconds"] == 1.0
    background_status = background_loop_statuses()["auto_predict"]
    assert background_status["status"] == "failed"
    assert background_status["lastError"] == "target load failed"
    assert background_status["lastExceptionType"] == "RuntimeError"
    assert background_status["lastFailureDetails"] == {
        "nextWaitSeconds": 1.0,
        "failureDetails": None,
    }


def test_auto_predict_invalid_initial_delay_is_exposed(monkeypatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()

    monkeypatch.setenv("PREDICT_INITIAL_DELAY_SECONDS", "bad")

    try:
        asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))
    except ValueError as exc:
        assert "PREDICT_INITIAL_DELAY_SECONDS must be numeric" in str(exc)
    else:
        raise AssertionError("invalid predict initial delay was not exposed")

    status = auto_predict_loop_status()
    assert status["status"] == "failed"
    assert status["exceptionType"] == "ValueError"
    assert "PREDICT_INITIAL_DELAY_SECONDS must be numeric" in status["error"]
    background_status = background_loop_statuses()["auto_predict"]
    assert background_status["status"] == "failed"
    assert background_status["lastExceptionType"] == "ValueError"
    assert "PREDICT_INITIAL_DELAY_SECONDS must be numeric" in background_status["lastError"]


def test_prepare_prediction_inputs_deduplicates_shared_work(monkeypatch) -> None:
    refresh_1m_calls = []
    refresh_duration_calls = []
    settlement_calls = []
    paper_live_calls = []
    strategy_settings = [
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        _settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY),
        _settings(FACTOR_COMBO_STRATEGY_KEY, symbol="ETHUSDT"),
    ]

    monkeypatch.setattr(input_deps, "refresh_1m_prediction_input", lambda *args: refresh_1m_calls.append(args))
    monkeypatch.setattr(
        input_deps,
        "refresh_duration_prediction_input",
        lambda *args: refresh_duration_calls.append(args),
    )
    monkeypatch.setattr(input_deps, "settle_due_predictions", lambda *args: settlement_calls.append(args))
    monkeypatch.setattr(input_deps, "refresh_paper_live_candidate_states", lambda *args: paper_live_calls.append(args))
    monkeypatch.setattr(
        input_deps,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )

    asyncio.run(service._prepare_prediction_inputs(strategy_settings))

    assert refresh_1m_calls == [
        ("BTCUSDT", ENTRY_OPEN_TIME),
        ("ETHUSDT", ENTRY_OPEN_TIME),
    ]
    assert refresh_duration_calls == [
        ("BTCUSDT", DEFAULT_DURATION, ENTRY_OPEN_TIME),
        ("ETHUSDT", DEFAULT_DURATION, ENTRY_OPEN_TIME),
    ]
    assert sorted(settlement_calls) == [("BTCUSDT", DEFAULT_DURATION), ("ETHUSDT", DEFAULT_DURATION)]
    assert sorted(paper_live_calls) == [("BTCUSDT", DEFAULT_DURATION), ("ETHUSDT", DEFAULT_DURATION)]


def test_refresh_prediction_inputs_request_required_completed_klines(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        input_deps,
        "refresh_prediction_klines",
        lambda *args: calls.append(args),
    )

    input_deps.refresh_1m_prediction_input("btcusdt", ENTRY_OPEN_TIME)
    input_deps.refresh_duration_prediction_input("btcusdt", DEFAULT_DURATION, ENTRY_OPEN_TIME)

    assert calls == [
        ("btcusdt", "1m", ENTRY_OPEN_TIME - input_deps.MS_PER_MINUTE),
        ("btcusdt", DEFAULT_DURATION, ENTRY_OPEN_TIME - input_deps._duration_ms(DEFAULT_DURATION)),
    ]


def test_should_predict_entry_backfills_missing_current_bucket_prediction(monkeypatch) -> None:
    existing_calls = []

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "prediction_exists", lambda **kwargs: existing_calls.append(kwargs) or False)

    assert service._should_predict_entry(_settings(FACTOR_COMBO_STRATEGY_KEY))

    assert existing_calls[0]["strategy_key"] == FACTOR_COMBO_STRATEGY_KEY


def test_should_predict_entry_backfills_ready_lstm_shadow(monkeypatch) -> None:
    calls = []
    missing_calls = []

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})
    monkeypatch.setattr(
        service,
        "eligible_factor_combo_rows",
        lambda *_args: [{"factorName": "combo__a"}, {"factorName": "combo__b"}],
    )
    monkeypatch.setattr(
        service,
        "prediction_exists",
        lambda **kwargs: calls.append(kwargs["strategy_key"]) or True,
    )
    monkeypatch.setattr(service, "factor_candidate_signal_keys", lambda *_args: ())
    monkeypatch.setattr(
        service,
        "missing_lstm_shadow_entry_times",
        lambda symbol, duration, bucket: missing_calls.append((symbol, duration, bucket)) or (ENTRY_OPEN_TIME,),
    )

    assert service._should_predict_entry(_settings(FACTOR_COMBO_STRATEGY_KEY))
    assert calls == [FACTOR_COMBO_STRATEGY_KEY]
    assert missing_calls == [("BTCUSDT", DEFAULT_DURATION, ENTRY_OPEN_TIME)]


def test_factor_combo_shadow_due_uses_offline_focused_candidates(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        service,
        "eligible_factor_combo_rows",
        lambda *_args: [{"factorName": "focused_combo"}],
    )
    monkeypatch.setattr(
        service,
        "prediction_exists",
        lambda **kwargs: calls.append(kwargs["strategy_key"]) or False,
    )

    due = service._factor_combo_shadow_due(
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        ENTRY_OPEN_TIME,
    )

    assert due is True
    assert calls == [simulation_strategy_key_for_factor_name("focused_combo")]


def test_factor_candidate_signal_slot_is_collection_only(monkeypatch) -> None:
    candidate_key = factor_candidate_signal_key("rsi_14")
    calls = []

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "prediction_exists", lambda **kwargs: calls.append(kwargs) or False)

    assert service._should_predict_entry(_settings(candidate_key)) is False
    assert calls == []


def test_candidate_collection_saves_top_two_and_three_shadow_rows(monkeypatch) -> None:
    saved = []
    trades = []

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", _batch_predictions)
    monkeypatch.setattr(service, "predict_factor_candidate_signals", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "create_batch_combo_simulation_trade", lambda settings, result: trades.append(result["strategy_key"]))
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {"shadowPredictionReady": False, "shadowPredictionBlockedReason": "torch_unavailable"},
    )
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})

    asyncio.run(
        service._save_candidate_collection_predictions(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock())
    )

    assert saved == [
        (simulation_strategy_key_for_factor_name("combo__a"), False),
        (simulation_strategy_key_for_factor_name("combo__b"), False),
    ]
    assert trades == [simulation_strategy_key_for_factor_name("combo__a"), simulation_strategy_key_for_factor_name("combo__b")]


def test_factor_combo_existing_primary_does_not_save_candidates_inline(monkeypatch) -> None:
    saved = []

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return result["strategy_key"] != FACTOR_COMBO_STRATEGY_KEY

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", _batch_predictions)
    monkeypatch.setattr(service, "predict_factor_candidate_signals", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "predict_lstm_shadow_prediction",
        lambda *_args, **_kwargs: _prediction(
            lstm_shadow_strategy_key(DEFAULT_DURATION),
            symbol="BTCUSDT",
            duration=DEFAULT_DURATION,
        ),
    )
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    asyncio.run(service._run_prediction(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock()))

    assert saved == [(FACTOR_COMBO_STRATEGY_KEY, False)]


def test_candidate_collection_saves_factor_candidate_signals(monkeypatch) -> None:
    saved = []
    trades = []
    candidate_key = factor_candidate_signal_key("rsi_14")

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "predict_factor_candidate_signals",
        lambda symbol, duration, **_kwargs: [_prediction(candidate_key, symbol=symbol, duration=duration)],
    )
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {"shadowPredictionReady": False, "shadowPredictionBlockedReason": "torch_unavailable"},
    )
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(
        service,
        "create_batch_combo_simulation_trade",
        lambda _settings, result: trades.append(result["strategy_key"]),
    )
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    asyncio.run(
        service._save_candidate_collection_predictions(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock())
    )

    assert saved == [(candidate_key, False)]
    assert trades == [candidate_key]


def test_factor_candidate_collection_failure_is_exposed(monkeypatch) -> None:
    saved = []

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "predict_factor_candidate_signals",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("candidate failed")),
    )
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {"shadowPredictionReady": False, "shadowPredictionBlockedReason": "torch_unavailable"},
    )
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    try:
        asyncio.run(
            service._save_candidate_collection_predictions(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock())
        )
    except RuntimeError as exc:
        assert str(exc) == "candidate failed"
    else:
        raise AssertionError("candidate collection failure was not exposed")

    assert saved == []


def test_model_family_shadow_prediction_failure_is_recorded(monkeypatch) -> None:
    failures = []

    monkeypatch.setattr(
        service.collection_helpers,
        "log_prediction_failure",
        lambda **kwargs: failures.append(kwargs),
    )

    try:
        asyncio.run(
            service.collection_helpers.save_one_model_family_shadow_prediction(
                _model_context("xgboost", {
                    "model_family_status": lambda *_args: {
                        "shadowPredictionReady": True,
                        "modelVersion": "xgboost_v2",
                        "status": "trade_active",
                    },
                    "predict_model_family_shadow_prediction": lambda *_args, **_kwargs: (
                        _ for _ in ()
                    ).throw(RuntimeError("xgboost failed")),
                    "save_prediction": lambda *_args, **_kwargs: None,
                    "create_batch_combo_simulation_trade": lambda *_args: None,
                }),
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "xgboost failed"
    else:
        raise AssertionError("model family shadow failure was not exposed")

    assert failures == [
        {
            "candidate_key": "xgboost:BTCUSDT:10m",
            "strategy_key": FACTOR_COMBO_STRATEGY_KEY,
            "symbol": "BTCUSDT",
            "duration": DEFAULT_DURATION,
            "stage": "model_shadow_prediction",
            "reason": "xgboost failed",
            "details": {
                "family": "xgboost",
                "entryOpenTime": ENTRY_OPEN_TIME,
                "exceptionType": "RuntimeError",
                "modelVersion": "xgboost_v2",
                "status": "trade_active",
            },
        }
    ]


def test_predict_due_entries_collects_candidates_when_primary_not_due(monkeypatch) -> None:
    calls = []
    targets = [_settings(FACTOR_COMBO_STRATEGY_KEY)]

    async def prepare(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("prepare", [(item.strategy_key, item.duration) for item in settings_list]))

    async def run_batch(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("run_batch", len(settings_list)))

    async def collect(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("collect", [(item.strategy_key, item.duration) for item in settings_list]))

    async def backfill(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("backfill", [(item.strategy_key, item.duration) for item in settings_list]))

    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda _targets: _ready_result([]))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda _targets: targets)
    monkeypatch.setattr(service, "_prepare_prediction_inputs", prepare)
    monkeypatch.setattr(service, "_run_prediction_batch", run_batch)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", collect)
    monkeypatch.setattr(service, "_backfill_lstm_shadow_predictions", backfill)

    asyncio.run(service._predict_due_entries(targets))

    assert calls == [
        ("prepare", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
        ("collect", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
        ("backfill", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
    ]


def test_predict_due_entries_runs_final_decision_after_candidates(monkeypatch) -> None:
    calls = []
    primary = _settings(FACTOR_COMBO_STRATEGY_KEY)
    final = _settings(EVENT_FINAL_DECISION_STRATEGY_KEY)
    targets = [primary, final]

    async def prepare(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("prepare", [item.strategy_key for item in settings_list]))

    async def run_batch(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("run_batch", [item.strategy_key for item in settings_list]))

    async def collect(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("collect", [item.strategy_key for item in settings_list]))

    async def backfill(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("backfill", [item.strategy_key for item in settings_list]))

    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda _targets: _ready_result(targets))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda _targets: [primary])
    monkeypatch.setattr(service, "_prepare_prediction_inputs", prepare)
    monkeypatch.setattr(service, "_run_prediction_batch", run_batch)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", collect)
    monkeypatch.setattr(service, "_backfill_lstm_shadow_predictions", backfill)

    asyncio.run(service._predict_due_entries(targets))

    assert calls == [
        ("prepare", [FACTOR_COMBO_STRATEGY_KEY, EVENT_FINAL_DECISION_STRATEGY_KEY]),
        ("run_batch", [FACTOR_COMBO_STRATEGY_KEY]),
        ("collect", [FACTOR_COMBO_STRATEGY_KEY]),
        ("backfill", [FACTOR_COMBO_STRATEGY_KEY, EVENT_FINAL_DECISION_STRATEGY_KEY]),
        ("run_batch", [EVENT_FINAL_DECISION_STRATEGY_KEY]),
    ]


def test_predict_due_entries_collects_candidates_when_primary_batch_fails(monkeypatch) -> None:
    calls = []
    targets = [_settings(FACTOR_COMBO_STRATEGY_KEY)]

    async def prepare(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("prepare", [(item.strategy_key, item.duration) for item in settings_list]))

    async def run_batch(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("run_batch", [(item.strategy_key, item.duration) for item in settings_list]))
        raise RuntimeError("primary failed")

    async def collect(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("collect", [(item.strategy_key, item.duration) for item in settings_list]))

    async def backfill(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("backfill", [(item.strategy_key, item.duration) for item in settings_list]))

    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda _targets: _ready_result(targets))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda _targets: targets)
    monkeypatch.setattr(service, "_prepare_prediction_inputs", prepare)
    monkeypatch.setattr(service, "_run_prediction_batch", run_batch)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", collect)
    monkeypatch.setattr(service, "_backfill_lstm_shadow_predictions", backfill)

    try:
        asyncio.run(service._predict_due_entries(targets))
    except RuntimeError as exc:
        assert str(exc) == "primary failed"
    else:
        raise AssertionError("primary prediction failure was not exposed")

    assert calls == [
        ("prepare", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
        ("run_batch", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
        ("collect", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
        ("backfill", [(FACTOR_COMBO_STRATEGY_KEY, DEFAULT_DURATION)]),
    ]


def test_predict_due_entries_continues_ready_targets_when_one_is_skipped(monkeypatch) -> None:
    calls = []
    ready = _settings(FACTOR_COMBO_STRATEGY_KEY, duration="30m")
    skipped = {
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
        "symbol": "BTCUSDT",
        "duration": DEFAULT_DURATION,
        "reason": "ranking_cache_empty",
    }
    targets = [_settings(FACTOR_COMBO_STRATEGY_KEY), ready]

    async def prepare(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("prepare", [item.duration for item in settings_list]))

    async def run_batch(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("run_batch", [item.duration for item in settings_list]))

    async def backfill(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("backfill", [item.duration for item in settings_list]))

    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda _targets: _ready_result([ready], [skipped]))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda _targets: [])
    monkeypatch.setattr(service, "_prepare_prediction_inputs", prepare)
    monkeypatch.setattr(service, "_run_prediction_batch", run_batch)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", _noop_settings)
    monkeypatch.setattr(service, "_backfill_lstm_shadow_predictions", backfill)

    summary = asyncio.run(service._predict_due_entries(targets))

    assert calls == [
        ("prepare", ["30m"]),
        ("run_batch", ["30m"]),
        ("backfill", ["30m"]),
    ]
    assert summary["readyDueCount"] == 1
    assert summary["skippedTargets"] == [skipped]


def test_candidate_collection_uses_factor_combo_source_for_non_combo_strategy(monkeypatch) -> None:
    saved = []
    trades = []
    candidate_key = factor_candidate_signal_key("rsi_14")

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(
        service,
        "predict_factor_candidate_signals",
        lambda symbol, duration, **_kwargs: [_prediction(candidate_key, symbol=symbol, duration=duration)],
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {"shadowPredictionReady": False, "shadowPredictionBlockedReason": "torch_unavailable"},
    )
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(
        service,
        "create_batch_combo_simulation_trade",
        lambda _settings, result: trades.append(result["strategy_key"]),
    )
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    target = service._collection_settings(_settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY))
    asyncio.run(service._save_candidate_collection_predictions(target, write_lock=asyncio.Lock()))

    assert target.strategy_key == FACTOR_COMBO_STRATEGY_KEY
    assert saved == [(candidate_key, False)]
    assert trades == [candidate_key]


def test_lstm_strategy_prediction_saves_own_simulation_row(monkeypatch) -> None:
    saved = []
    lstm_key = lstm_shadow_strategy_key("10m")

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(
        service,
        "predict_rule_direction",
        lambda symbol, duration, **kwargs: _prediction(kwargs["strategy_key"], symbol=symbol, duration=duration),
    )
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(service, "prediction_response", lambda result: result)
    monkeypatch.setattr(service, "_broadcast", _noop_broadcast)

    asyncio.run(service._run_prediction(_settings(lstm_key), write_lock=asyncio.Lock()))

    assert saved == [(lstm_key, False)]


def test_ready_model_family_shadow_creates_simulation_trade(monkeypatch) -> None:
    saved = []
    trades = []
    lstm_key = lstm_shadow_strategy_key("10m")

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append((result["strategy_key"], allow_existing))
        return True

    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(
        service,
        "predict_lstm_shadow_prediction",
        lambda *_args, **_kwargs: _prediction(lstm_key, symbol="BTCUSDT", duration=DEFAULT_DURATION),
    )
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(
        service,
        "create_batch_combo_simulation_trade",
        lambda settings, result: trades.append((settings.strategy_key, result["strategy_key"])),
    )

    asyncio.run(
        service._save_one_model_family_shadow_prediction(
            "lstm",
            {
                "settings": _settings(FACTOR_COMBO_STRATEGY_KEY),
                "entry_open_time": ENTRY_OPEN_TIME,
                "write_lock": asyncio.Lock(),
            },
        )
    )

    assert saved == [(lstm_key, False)]
    assert trades == [(FACTOR_COMBO_STRATEGY_KEY, lstm_key)]


def test_candidate_collection_saves_ready_model_family_shadow_sim(monkeypatch) -> None:
    saved = []
    trades = []
    lstm_key = lstm_shadow_strategy_key(DEFAULT_DURATION)

    async def save_prediction(result: dict, _write_lock: asyncio.Lock, *, allow_existing: bool = False) -> bool:
        saved.append(result["strategy_key"])
        return True

    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "predict_eligible_factor_combo_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "predict_factor_candidate_signals", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "lstm_model_status", lambda *_args: {"shadowPredictionReady": True})
    monkeypatch.setattr(service, "model_family_status", lambda *_args: {"shadowPredictionReady": False})
    monkeypatch.setattr(
        service,
        "predict_lstm_shadow_prediction",
        lambda *_args, **_kwargs: _prediction(lstm_key, symbol="BTCUSDT", duration=DEFAULT_DURATION),
    )
    monkeypatch.setattr(service, "_save_prediction", save_prediction)
    monkeypatch.setattr(
        service,
        "create_batch_combo_simulation_trade",
        lambda _settings, result: trades.append(result["strategy_key"]),
    )

    asyncio.run(
        service._save_candidate_collection_predictions(_settings(FACTOR_COMBO_STRATEGY_KEY), write_lock=asyncio.Lock())
    )

    assert lstm_key in saved
    assert lstm_key in trades


def test_prediction_targets_include_all_enabled_slots(monkeypatch) -> None:
    mixed = [
        _settings(FACTOR_COMBO_STRATEGY_KEY, duration="30m"),
        _settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration="10m"),
    ]
    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: mixed)
    targets = service._prediction_targets()
    keys = {(target.strategy_key, target.duration) for target in targets}
    assert keys == {
        (FACTOR_COMBO_STRATEGY_KEY, "30m"),
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "10m"),
    }


def test_ready_due_prediction_targets_exposes_skipped_and_keeps_ready(monkeypatch) -> None:
    mixed = [
        _settings(FACTOR_COMBO_STRATEGY_KEY, duration="10m"),
        _settings(FACTOR_COMBO_STRATEGY_KEY, duration="30m"),
    ]
    readiness = {
        (FACTOR_COMBO_STRATEGY_KEY, "10m"): _readiness(False, "ranking_cache_empty"),
        (FACTOR_COMBO_STRATEGY_KEY, "30m"): _readiness(True),
    }
    recovery_flags = []

    monkeypatch.setattr(
        service,
        "strategy_prediction_readiness",
        lambda strategy_key, _symbol, duration, **kwargs: recovery_flags.append(kwargs["attempt_recovery"])
        or readiness[(strategy_key, duration)],
    )
    monkeypatch.setattr(service, "_due_prediction_targets", lambda targets: targets)

    result = service._ready_due_prediction_targets(mixed)

    assert list(result.ready) == [mixed[1]]
    assert result.skipped[0]["reason"] == "ranking_cache_empty"
    assert result.skipped[0]["strategyKey"] == FACTOR_COMBO_STRATEGY_KEY
    assert recovery_flags == [False, False]


def test_auto_predict_loop_records_due_readiness_skips(monkeypatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()
    skipped = {
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
        "symbol": "BTCUSDT",
        "duration": DEFAULT_DURATION,
        "reason": "ranking_cache_empty",
    }

    def skip_ready(_targets):
        stop_event.set()
        return _ready_result([], [skipped])

    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "_prediction_targets", lambda: [_settings(FACTOR_COMBO_STRATEGY_KEY)])
    monkeypatch.setattr(service, "_ready_due_prediction_targets", skip_ready)
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda _targets: [])

    asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))

    status = auto_predict_loop_status()
    assert status["status"] == "stopped"
    assert status["error"] is None
    assert status["exceptionType"] is None
    assert status["skippedTargetCount"] == 1
    assert status["skippedTargets"] == [skipped]
    background_status = background_loop_statuses()["auto_predict"]
    assert background_status["lastSuccessDetails"]["skippedTargets"] == [skipped]


def test_prediction_targets_do_not_fallback_to_default_when_enabled_targets_invalid(monkeypatch) -> None:
    mixed = [_settings(FACTOR_COMBO_STRATEGY_KEY, duration="10m")]

    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: mixed)

    assert service._prediction_targets() == mixed


def test_prediction_targets_expose_enabled_unsupported_duration(monkeypatch) -> None:
    target = _settings(FACTOR_COMBO_STRATEGY_KEY, duration="30m")

    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: [target])
    monkeypatch.setattr(service, "strategy_supports_duration", lambda *_args: False)

    try:
        service._prediction_targets()
    except service.target_helpers.PredictionTargetConfigError as exc:
        assert exc.details == {
            "invalidTargets": [
                {
                    "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
                    "symbol": "BTCUSDT",
                    "duration": "30m",
                    "reason": "unsupported_duration",
                }
            ]
        }
    else:
        raise AssertionError("unsupported enabled predict target was silently dropped")


def test_auto_predict_loop_records_invalid_enabled_target_details(monkeypatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()
    invalid = {
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
        "symbol": "BTCUSDT",
        "duration": "30m",
        "reason": "unsupported_duration",
    }

    def fail_targets():
        stop_event.set()
        raise service.target_helpers.PredictionTargetConfigError([invalid])

    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "_prediction_targets", fail_targets)

    asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))

    status = auto_predict_loop_status()
    assert status["status"] == "failed"
    assert status["exceptionType"] == "PredictionTargetConfigError"
    assert status["failureDetails"] == {"invalidTargets": [invalid]}
    background_status = background_loop_statuses()["auto_predict"]
    assert background_status["status"] == "failed"
    assert background_status["lastFailureDetails"]["failureDetails"] == {"invalidTargets": [invalid]}


def test_next_predict_wait_polls_when_candidate_collection_is_due(monkeypatch) -> None:
    settings = _settings(FACTOR_COMBO_STRATEGY_KEY)
    current_bucket = ENTRY_OPEN_TIME
    next_wait_seconds = 120.0

    monkeypatch.setattr(service, "utc_now_ms", lambda: current_bucket)
    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: current_bucket,
    )
    monkeypatch.setattr(service, "_should_predict_entry", lambda _settings: False)
    monkeypatch.setattr(
        service,
        "_candidate_collection_due",
        lambda collection, bucket: collection.strategy_key == FACTOR_COMBO_STRATEGY_KEY
        and bucket == current_bucket,
    )
    monkeypatch.setattr(
        service,
        "seconds_until_next_rule_entry_for_duration",
        lambda _duration, _now_ms=None: next_wait_seconds,
    )

    assert service._next_predict_wait([settings], 1) == 1.0


def test_prediction_targets_expose_default_readiness_failure(monkeypatch) -> None:
    disabled = [_settings(FACTOR_COMBO_STRATEGY_KEY, duration="10m", enabled=False)]
    recovery_flags = []

    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: disabled)
    monkeypatch.setattr(service, "get_auto_trade_settings", lambda _key: _settings(FACTOR_COMBO_STRATEGY_KEY))
    monkeypatch.setattr(
        service,
        "strategy_prediction_readiness",
        lambda *_args, **kwargs: recovery_flags.append(kwargs["attempt_recovery"])
        or _readiness(False, "ranking_cache_empty", recovery_attempted=True),
    )

    try:
        service._prediction_targets()
    except service.target_helpers.PredictionTargetReadinessError as exc:
        assert exc.details["skippedTargets"][0]["reason"] == "ranking_cache_empty"
        assert exc.details["skippedTargets"][0]["recoveryAttempted"] is True
    else:
        raise AssertionError("default predict target readiness failure was silently converted to no targets")
    assert recovery_flags == [True]


def test_auto_predict_loop_records_default_readiness_failure(monkeypatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()

    def fail_readiness(*_args, **_kwargs):
        stop_event.set()
        return _readiness(False, "ranking_cache_empty")

    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "list_auto_trade_settings", lambda: [])
    monkeypatch.setattr(service, "get_auto_trade_settings", lambda _key: _settings(FACTOR_COMBO_STRATEGY_KEY))
    monkeypatch.setattr(service, "strategy_prediction_readiness", fail_readiness)

    asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))

    status = auto_predict_loop_status()
    assert status["status"] == "failed"
    assert status["exceptionType"] == "PredictionTargetReadinessError"
    assert status["failureDetails"]["skippedTargets"][0]["reason"] == "ranking_cache_empty"


def test_ready_lstm_shadow_due_does_not_sync_snapshot_mismatch(monkeypatch) -> None:
    settings = _settings(FACTOR_COMBO_STRATEGY_KEY)

    monkeypatch.setattr(service, "prediction_exists", lambda **_kwargs: False)
    monkeypatch.setattr(
        service,
        "lstm_model_status",
        lambda *_args: {
            "shadowPredictionReady": False,
            "shadowPredictionBlockedReason": "combo_snapshot_mismatch",
        },
    )

    assert service._ready_lstm_shadow_due(settings, ENTRY_OPEN_TIME) is False


def test_run_prediction_batch_starts_strategies_concurrently(monkeypatch) -> None:
    strategy_settings = [
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        _settings(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY),
        _settings(lstm_shadow_strategy_key(DEFAULT_DURATION)),
    ]
    started = []
    completed = []
    expected_count = len(strategy_settings)
    start_gate = {}

    async def run_prediction(setting: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
        del write_lock
        started.append(setting.strategy_key)
        if len(started) == expected_count:
            start_gate["event"].set()
        await start_gate["event"].wait()
        completed.append(setting.strategy_key)

    async def run_batch() -> None:
        start_gate["event"] = asyncio.Event()
        monkeypatch.setattr(service, "_run_prediction", run_prediction)
        await asyncio.wait_for(
            service._run_prediction_batch(strategy_settings),
            timeout=ASYNC_TEST_TIMEOUT_SECONDS,
        )

    asyncio.run(run_batch())

    assert set(started) == {settings.strategy_key for settings in strategy_settings}
    assert set(completed) == {settings.strategy_key for settings in strategy_settings}


def test_candidate_collection_batch_error_has_details() -> None:
    setting = _settings(FACTOR_COMBO_STRATEGY_KEY)
    failure = RuntimeError("collection failed")
    logged = []

    class Logger:
        def error(self, message: str, *args, exc_info=None) -> None:
            logged.append((message, args, exc_info))

    async def save_collection(_setting: AutoTradeSettings, *, write_lock: asyncio.Lock) -> None:
        del write_lock
        raise failure

    try:
        asyncio.run(service.runtime_helpers.run_candidate_collection_batch([setting], save_collection, Logger()))
    except service.runtime_helpers.CandidateCollectionBatchError as exc:
        assert str(exc) == "candidate collection failed for: BTCUSDT:10m"
        assert exc.details == [
            {
                "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
                "symbol": "BTCUSDT",
                "duration": DEFAULT_DURATION,
                "error": "collection failed",
                "exceptionType": "RuntimeError",
            }
        ]
    else:
        raise AssertionError("candidate collection batch failure details were not exposed")

    assert logged[0][0] == "candidate collection failed strategy=%s symbol=%s duration=%s"
    assert logged[0][1] == (FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", DEFAULT_DURATION)
    assert logged[0][2][0] is RuntimeError


def test_prediction_failures_include_base_exceptions() -> None:
    class BaseFailure(BaseException):
        pass

    failure = BaseFailure("base prediction failed")
    failures = service.runtime_helpers.prediction_failures(
        [_settings(FACTOR_COMBO_STRATEGY_KEY)],
        [failure],
    )

    assert len(failures) == 1
    error = service.runtime_helpers.PredictionBatchError(failures)
    assert error.details == [
        {
            "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
            "symbol": "BTCUSDT",
            "duration": DEFAULT_DURATION,
            "error": "base prediction failed",
            "exceptionType": "BaseFailure",
        }
    ]


def test_predict_due_entries_runs_lstm_shadow_backfill_after_current_predictions(monkeypatch) -> None:
    calls = []
    targets = [_settings(lstm_shadow_strategy_key(DEFAULT_DURATION))]

    async def run_batch(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("run_batch", [item.strategy_key for item in settings_list]))

    async def backfill(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("backfill", [item.strategy_key for item in settings_list]))

    async def prepare(settings_list: list[AutoTradeSettings]) -> None:
        calls.append(("prepare", len(settings_list)))

    async def collect(_settings_list: list[AutoTradeSettings]) -> None:
        return None

    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda items: _ready_result(items))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda _items: [])
    monkeypatch.setattr(service, "_prepare_prediction_inputs", prepare)
    monkeypatch.setattr(service, "_run_prediction_batch", run_batch)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", collect)
    monkeypatch.setattr(service, "_backfill_lstm_shadow_predictions", backfill)

    asyncio.run(service._predict_due_entries(targets))

    assert calls == [
        ("prepare", 1),
        ("run_batch", [lstm_shadow_strategy_key(DEFAULT_DURATION)]),
        ("backfill", [lstm_shadow_strategy_key(DEFAULT_DURATION)]),
    ]


def test_backfill_shadow_predictions_exposes_returned_exception(monkeypatch) -> None:
    logged = []
    failure = RuntimeError("backfill failed")

    class Logger:
        def error(self, message: str, *args, exc_info=None) -> None:
            logged.append((message, args, exc_info))

    def backfill(*_args) -> None:
        raise failure

    monkeypatch.setattr(
        service,
        "_ready_model_family_shadow_backfill_targets",
        lambda _settings_list: [("gru", "BTCUSDT", DEFAULT_DURATION)],
    )
    monkeypatch.setattr(
        service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration: ENTRY_OPEN_TIME,
    )
    monkeypatch.setattr(service, "backfill_model_family_shadow_predictions", backfill)
    monkeypatch.setattr(service, "logger", Logger())

    try:
        asyncio.run(service._backfill_lstm_shadow_predictions([_settings(FACTOR_COMBO_STRATEGY_KEY)]))
    except service.shadow_helpers.ShadowBackfillBatchError as exc:
        assert str(exc) == "model family shadow backfill failed for: gru:BTCUSDT:10m"
        assert exc.details == [
            {
                "family": "gru",
                "symbol": "BTCUSDT",
                "duration": DEFAULT_DURATION,
                "error": "backfill failed",
                "exceptionType": "RuntimeError",
            }
        ]
    else:
        raise AssertionError("shadow backfill failure was not exposed")

    assert logged == [
        (
            "model family shadow backfill failed family=%s symbol=%s duration=%s",
            ("gru", "BTCUSDT", DEFAULT_DURATION),
            (RuntimeError, failure, failure.__traceback__),
        )
    ]


def test_backfill_shadow_summary_exposes_base_exception() -> None:
    logged = []

    class BaseFailure(BaseException):
        pass

    class Logger:
        def error(self, message: str, *args, exc_info=None) -> None:
            logged.append((message, args, exc_info))

    failure = BaseFailure("base backfill failed")
    target = service.shadow_helpers.ShadowBackfillTarget("gru", "BTCUSDT", DEFAULT_DURATION)
    result = service.shadow_helpers.handle_backfill_summary(target, failure, Logger())

    assert result is not None
    error = service.shadow_helpers.ShadowBackfillBatchError([result])
    assert error.details == [
        {
            "family": "gru",
            "symbol": "BTCUSDT",
            "duration": DEFAULT_DURATION,
            "error": "base backfill failed",
            "exceptionType": "BaseFailure",
        }
    ]
    assert logged[0][2][0] is BaseFailure


def test_auto_predict_loop_records_shadow_backfill_failure_details(monkeypatch) -> None:
    stop_event = asyncio.Event()
    failure = RuntimeError("backfill failed")

    def backfill(*_args) -> None:
        stop_event.set()
        raise failure

    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "_prediction_targets", lambda: [_settings(FACTOR_COMBO_STRATEGY_KEY)])
    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda _targets: _ready_result([]))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda targets: targets)
    monkeypatch.setattr(service, "_prepare_prediction_inputs", _noop_settings)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", _noop_settings)
    monkeypatch.setattr(
        service,
        "_ready_model_family_shadow_backfill_targets",
        lambda _settings_list: [("gru", "BTCUSDT", DEFAULT_DURATION)],
    )
    monkeypatch.setattr(service, "current_rule_entry_open_time_for_duration", lambda _duration: ENTRY_OPEN_TIME)
    monkeypatch.setattr(service, "backfill_model_family_shadow_predictions", backfill)

    asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))

    status = auto_predict_loop_status()
    assert status["status"] == "failed"
    assert status["exceptionType"] == "ShadowBackfillBatchError"
    assert status["failureDetails"] == [
        {
            "family": "gru",
            "symbol": "BTCUSDT",
            "duration": DEFAULT_DURATION,
            "error": "backfill failed",
            "exceptionType": "RuntimeError",
        }
    ]


def test_auto_predict_loop_records_candidate_collection_failure_details(monkeypatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()
    setting = _settings(FACTOR_COMBO_STRATEGY_KEY)
    failure = RuntimeError("collection failed")

    async def collect(settings_list: list[AutoTradeSettings]) -> None:
        stop_event.set()
        raise service.runtime_helpers.CandidateCollectionBatchError(
            [service.runtime_helpers.PredictionFailure(settings_list[0], failure)]
        )

    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "_prediction_targets", lambda: [setting])
    monkeypatch.setattr(service, "_ready_due_prediction_targets", lambda _targets: _ready_result([]))
    monkeypatch.setattr(service, "_candidate_collection_targets", lambda targets: targets)
    monkeypatch.setattr(service, "_prepare_prediction_inputs", _noop_settings)
    monkeypatch.setattr(service, "_run_candidate_collection_batch", collect)
    monkeypatch.setattr(service, "_backfill_lstm_shadow_predictions", _noop_settings)

    asyncio.run(service.auto_predict_loop(stop_event, poll_seconds=1))

    status = auto_predict_loop_status()
    assert status["status"] == "failed"
    assert status["exceptionType"] == "CandidateCollectionBatchError"
    assert status["failureDetails"] == [
        {
            "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
            "symbol": "BTCUSDT",
            "duration": DEFAULT_DURATION,
            "error": "collection failed",
            "exceptionType": "RuntimeError",
        }
    ]
    background_status = background_loop_statuses()["auto_predict"]
    assert background_status["lastFailureDetails"]["failureDetails"] == status["failureDetails"]


def test_prediction_broadcast_failure_is_exposed() -> None:
    class BadWs:
        async def send_json(self, _payload: dict) -> None:
            raise RuntimeError("send failed")

    subscribers = {("BTCUSDT", DEFAULT_DURATION, FACTOR_COMBO_STRATEGY_KEY): {BadWs()}}
    result = {
        "symbol": "BTCUSDT",
        "duration": DEFAULT_DURATION,
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
    }

    try:
        asyncio.run(service.runtime_helpers.broadcast(result, subscribers, service.DEFAULT_STRATEGY_KEY))
    except service.runtime_helpers.BroadcastDeliveryError as exc:
        assert str(exc) == "prediction broadcast failed for 1 subscriber(s)"
        assert exc.details == [{"error": "send failed", "exceptionType": "RuntimeError"}]
    else:
        raise AssertionError("broadcast failure was not exposed")

    assert subscribers[("BTCUSDT", DEFAULT_DURATION, FACTOR_COMBO_STRATEGY_KEY)] == set()


def _settings(
    strategy_key: str,
    *,
    symbol: str = "BTCUSDT",
    duration: str = DEFAULT_DURATION,
    enabled: bool = True,
) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=enabled,
        symbol=symbol,
        duration=duration,
        duration_minutes=int(DURATION_TO_MINUTES[duration]),
        qty=DEFAULT_QTY,
        live_trading_enabled=False,
    )


def _prediction(strategy_key: str, *, symbol: str, duration: str) -> dict:
    return {
        "strategy_key": strategy_key,
        "symbol": symbol,
        "duration": duration,
        "open_time": ENTRY_OPEN_TIME,
        "direction": "up",
        "probability_up": 0.55,
        "confidence": 0.55,
        "certainty_label": "FACTOR_COMBO_WAIT",
        "trade_quality_score": 0.5,
        "trade_quality_passed": True,
    }


def _batch_predictions(symbol: str, duration: str, **_kwargs) -> list[dict]:
    return [
        _prediction(simulation_strategy_key_for_factor_name("combo__a"), symbol=symbol, duration=duration),
        _prediction(simulation_strategy_key_for_factor_name("combo__b"), symbol=symbol, duration=duration),
    ]


async def _noop_broadcast(_result: dict) -> None:
    return None


async def _noop_settings(_settings_list: list[AutoTradeSettings]) -> None:
    return None


def _ready_result(
    ready: list[AutoTradeSettings],
    skipped: list[dict] | None = None,
) -> service.target_helpers.ReadyPredictionTargetResult:
    return service.target_helpers.ReadyPredictionTargetResult(tuple(ready), tuple(skipped or []))


def _readiness(ready: bool, reason: str = "ready", **kwargs):
    return PredictionReadiness(ready, reason, **kwargs)


def _model_context(family: str, deps: dict):
    collection = service.collection_helpers.CandidateCollectionContext(
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        ENTRY_OPEN_TIME,
        asyncio.Lock(),
        deps,
    )
    return service.collection_helpers.ModelFamilyShadowContext(family, collection)
