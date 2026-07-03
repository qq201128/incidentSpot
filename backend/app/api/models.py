from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.params import Param

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.model_family_config import MODEL_FAMILIES, normalize_model_family
from app.services.model_family_search_rules import DEFAULT_PARALLEL_WORKERS
from app.services.model_family_prediction_service import predict_model_family_signal
from app.services.model_family_research_bundle import model_family_research_bundle
from app.services.model_family_status_service import model_family_status
from app.services.model_search_api_worker import ensure_api_model_search_worker
from app.services.model_search_resource import ModelSearchResourceConfig, resource_payload, validated_resource_config
from app.services.model_search_resource_defaults import DEFAULT_INTERNAL_THREADS, DEFAULT_XGBOOST_PROCESS_WORKERS
from app.services.model_search_status_service import model_search_queue_status, model_search_status_with_lifecycle
from app.services.model_search_untrained_enqueue import enqueue_untrained_model_search_jobs
from app.services.runtime_symbols import configured_runtime_symbols, parse_symbol_csv

router = APIRouter(prefix="/api/models", tags=["models"])
DEFAULT_MODEL_SEARCH_DURATIONS = ("10m", "30m", "60m", "1d")
QUICK_MODEL_SEARCH_PRIORITY = 50


def _query_str(value: object, default: str | None = None) -> str | None:
    return value if isinstance(value, str) else default


def _query_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _query_int(value: object, default: int | None = None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


@router.get("/research-bundle")
def model_research_bundle(symbol: str = Query(..., min_length=6), duration: str = Query("10m")) -> dict:
    try:
        return model_family_research_bundle(symbol.upper(), _query_str(duration, "10m") or "10m")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{family}/status")
def model_status(family: str, symbol: str = Query(..., min_length=6), duration: str = Query("10m")) -> dict:
    try:
        return model_family_status(family, symbol.upper(), _query_str(duration, "10m") or "10m")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{family}/train")
def model_train(
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("full"),
    feature_window: int | None = Query(None, alias="featureWindow"),
    epochs: int | None = Query(None),
    batch_size: int | None = Query(None, alias="batchSize"),
    min_samples: int | None = Query(None, alias="minSamples"),
    learning_rate: float | None = Query(None, alias="learningRate"),
    hidden_size: int | None = Query(None, alias="hiddenSize"),
    num_layers: int | None = Query(None, alias="numLayers"),
    min_move_bps: float | None = Query(None, alias="minMoveBps"),
) -> dict:
    try:
        _reject_direct_train_overrides(
            feature_window,
            epochs,
            batch_size,
            min_samples,
            learning_rate,
            hidden_size,
            num_layers,
            min_move_bps,
        )
        return model_candidate_search(
            family,
            symbol=symbol,
            duration=_query_str(duration, "10m") or "10m",
            profile=_query_str(profile, "full") or "full",
            reset_history=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _reject_direct_train_overrides(*values: object | None) -> None:
    if any(_explicit_query_value(value) for value in values):
        raise ValueError("direct in-process training is disabled; enqueue candidate-search jobs instead")


def _explicit_query_value(value: object | None) -> bool:
    return value is not None and not isinstance(value, Param)


@router.post("/{family}/candidate-search")
def model_candidate_search(
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    profile: str = Query("fast"),
    reset_history: bool = Query(False, alias="resetHistory"),
    parallel_workers: int = Query(DEFAULT_PARALLEL_WORKERS, alias="parallelWorkers", ge=1),
    internal_threads: int = Query(DEFAULT_INTERNAL_THREADS, alias="internalThreads", ge=1),
    xgboost_process_workers: int = Query(DEFAULT_XGBOOST_PROCESS_WORKERS, alias="xgboostProcessWorkers", ge=1),
    search_mode: str = Query("balanced", alias="searchMode"),  # 新增：智能搜索模式
) -> dict:
    try:
        selected_duration = _query_str(duration, "10m") or "10m"
        selected_profile = normalize_experiment_profile(_query_str(profile, "fast") or "fast")
        selected = normalize_model_family(family)
        sym = symbol.upper()
        reset_requested = _query_bool(reset_history)
        mode = _query_str(search_mode, "balanced") or "balanced"

        # 验证搜索模式
        if mode not in ["fast", "balanced", "exhaustive", "legacy"]:
            mode = "balanced"

        resource = _resource_from_query(
            internal_threads=internal_threads,
            parallel_workers=parallel_workers,
            xgboost_process_workers=xgboost_process_workers,
        )

        # 如果使用智能搜索模式，走新逻辑
        if mode != "legacy":
            from app.services.model_search_smart_enqueue import enqueue_smart_model_search

            queued = enqueue_smart_model_search(
                family=selected,
                symbol=sym,
                duration=selected_duration,
                mode=mode,
                priority=QUICK_MODEL_SEARCH_PRIORITY,
                resource=resource,
            )
        else:
            # 兼容旧逻辑
            queued = enqueue_untrained_model_search_jobs(
                symbols=(sym,),
                durations=(selected_duration,),
                families=(selected,),
                profile=selected_profile,
                priority=QUICK_MODEL_SEARCH_PRIORITY,
                reset_existing=reset_requested,
                reset_history=reset_requested,
                resource=resource,
            )

        _ensure_worker_for_jobs(queued, resource)
        status = model_family_status(selected, sym, selected_duration)
        worker = _candidate_search_worker_status(sym, selected_duration, selected)
        return {
            **status,
            "modelSearchJob": queued["jobs"][0] if queued["jobs"] else None,
            "modelSearchQueue": queued,
            "workerStatus": worker,
            "searchMode": mode,
            "message": _candidate_search_message(selected, worker, queued),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search/retrain-all")
def model_search_retrain_all(
    *,
    symbols: str | None = Query(None),
    durations: str | None = Query(None),
    families: str | None = Query(None),
    profile: str = Query("full"),
    reset_history: bool = Query(True, alias="resetHistory"),
    internal_threads: int = Query(DEFAULT_INTERNAL_THREADS, alias="internalThreads", ge=1),
    parallel_workers: int = Query(DEFAULT_PARALLEL_WORKERS, alias="parallelWorkers", ge=1),
    xgboost_process_workers: int = Query(DEFAULT_XGBOOST_PROCESS_WORKERS, alias="xgboostProcessWorkers", ge=1),
    search_mode: str = Query("balanced", alias="searchMode"),  # 新增：智能搜索模式
) -> dict:
    try:
        selected = _batch_search_targets(symbols=symbols, durations=durations, families=families)
        selected_profile = normalize_experiment_profile(_query_str(profile, "full") or "full")
        mode = _query_str(search_mode, "balanced") or "balanced"

        # 验证搜索模式
        if mode not in ["fast", "balanced", "exhaustive", "legacy"]:
            mode = "balanced"

        resource = _resource_from_query(
            internal_threads=internal_threads,
            parallel_workers=parallel_workers,
            xgboost_process_workers=xgboost_process_workers,
        )

        # 如果使用智能搜索模式，走新逻辑
        if mode != "legacy":
            from app.services.model_search_smart_batch import enqueue_smart_batch_search

            queued = enqueue_smart_batch_search(
                symbols=selected["symbols"],
                durations=selected["durations"],
                families=selected["families"],
                mode=mode,
                resource=resource,
            )
        else:
            # 兼容旧逻辑
            queued = enqueue_untrained_model_search_jobs(
                **selected,
                profile=selected_profile,
                reset_existing=True,
                reset_history=_query_bool(reset_history, True),
                resource=resource,
            )

        _ensure_worker_for_jobs(queued, resource)
        worker = model_search_queue_status(selected, include_symbol_details=False)["workerStatus"]
        return {
            "version": "model_search_retrain_all_v2",
            "targets": {key: list(value) for key, value in selected.items()},
            "modelSearchQueue": queued,
            "workerStatus": worker,
            "searchMode": mode,
            "message": _batch_search_message(queued, worker),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{family}/predict")
def model_predict(
    family: str,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    entry_open_time: int | None = Query(None, alias="entryOpenTime"),
) -> dict:
    try:
        selected_duration = _query_str(duration, "10m") or "10m"
        return predict_model_family_signal(
            family,
            symbol.upper(),
            selected_duration,
            entry_open_time=_query_int(entry_open_time),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/search/jobs/status")
def model_search_jobs_status(
    symbols: str | None = Query(None),
    duration: str | None = Query(None),
    family: str | None = Query(None),
    status: str | None = Query(None),
) -> dict:
    selected_symbols = _query_str(symbols)
    selected_duration = _query_str(duration)
    selected_family = _query_str(family)
    selected_status = _query_str(status)
    filters = {
        "symbols": parse_symbol_csv(selected_symbols) if selected_symbols else (),
        "durations": (selected_duration,) if selected_duration else (),
        "families": (normalize_model_family(selected_family),) if selected_family else (),
        "statuses": (selected_status,) if selected_status else (),
    }
    return model_search_status_with_lifecycle(filters)


def _candidate_search_message(family: str, worker: dict, queued: dict) -> str:
    if not queued.get("jobs"):
        return f"{family}已有可用模型，本次未重复入队训练。"
    if worker["state"] in {"queued", "running"}:
        return f"{family}候选搜索已入队，model search worker 正在执行队列。"
    command = worker["workerRequiredCommand"]
    return f"{family}候选搜索已入队，但当前未检测到运行中的 worker。请启动：{command}"


def _candidate_search_worker_status(symbol: str, duration: str, family: str) -> dict:
    status = model_search_queue_status({"symbols": (symbol,), "durations": (duration,), "families": (family,)})
    return status["workerStatus"]


def _ensure_worker_for_jobs(queued: dict, resource: dict) -> None:
    if not queued.get("jobs"):
        return
    ensure_api_model_search_worker(resource)


def _batch_search_targets(
    *,
    symbols: str | None,
    durations: str | None,
    families: str | None,
) -> dict[str, tuple[str, ...]]:
    selected_symbols = _query_str(symbols)
    selected_durations = _query_str(durations)
    selected_families = _query_str(families)
    return {
        "symbols": parse_symbol_csv(selected_symbols) if selected_symbols else configured_runtime_symbols(),
        "durations": _csv_values(selected_durations) if selected_durations else DEFAULT_MODEL_SEARCH_DURATIONS,
        "families": _selected_families(selected_families),
    }


def _selected_families(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return MODEL_FAMILIES
    return tuple(normalize_model_family(value) for value in _csv_values(raw))


def _csv_values(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("CSV argument must include at least one value")
    return values


def _resource_from_query(
    *,
    internal_threads: int,
    parallel_workers: int,
    xgboost_process_workers: int,
) -> dict:
    return resource_payload(
        validated_resource_config(
            ModelSearchResourceConfig(
                internal_threads=_query_int(internal_threads, DEFAULT_INTERNAL_THREADS) or DEFAULT_INTERNAL_THREADS,
                parallel_workers=_query_int(parallel_workers, DEFAULT_PARALLEL_WORKERS) or DEFAULT_PARALLEL_WORKERS,
                xgboost_process_workers=_query_int(
                    xgboost_process_workers,
                    DEFAULT_XGBOOST_PROCESS_WORKERS,
                ) or DEFAULT_XGBOOST_PROCESS_WORKERS,
            )
        )
    )


def _batch_search_message(queued: dict, worker: dict) -> str:
    if queued.get("jobs") and worker["state"] in {"queued", "running"}:
        return "全部模型族重训任务已入队，model search worker 正在执行队列。"
    if queued.get("jobs"):
        return f"全部模型族重训任务已入队；请启动：{worker['workerRequiredCommand']}"
    return "没有可入队的模型族重训任务。"
