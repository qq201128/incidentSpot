from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.feature_integrity_service import timeframe_feature_integrity_report
from app.services.forward_validation_service import forward_validation_summary
from app.services.model_metrics import metric_summary


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"
ARCHIVE_DIR = MODEL_DIR / "archive"
CANDIDATE_DIR = MODEL_DIR / "candidates"
SYMBOL = "BTCUSDT"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    duration: str
    suffix: str
    family: str
    default_min_move_bps: float
    default_window_days: int


@dataclass(frozen=True)
class VersionRecord:
    spec: ModelSpec
    version_id: str
    artifact_dir: Path
    status: str
    metrics: dict[str, Any]
    trigger: str
    decision: dict[str, Any]


MODEL_SPECS = (
    ModelSpec("10m_enhanced", "10分钟增强模型", "10m", "10m_enhanced", "enhanced", 3.0, 60),
    ModelSpec("10m", "10分钟基础模型", "10m", "10m", "legacy", 3.0, 45),
    ModelSpec("30m", "30分钟基础模型", "30m", "30m", "legacy", 5.0, 45),
    ModelSpec("60m", "60分钟基础模型", "60m", "60m", "legacy", 5.0, 45),
    ModelSpec("1d", "1天基础模型", "1d", "1d", "legacy", 5.0, 60),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_specs() -> tuple[ModelSpec, ...]:
    return MODEL_SPECS


def spec_by_key(model_key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == model_key:
            return spec
    raise ValueError(f"unsupported model key: {model_key}")


def artifact_paths(base_dir: Path, spec: ModelSpec) -> dict[str, Path]:
    prefix = f"model_{spec.suffix}"
    return {
        "model": base_dir / f"{prefix}.pkl",
        "calibrator": base_dir / f"{prefix}_calibrator.pkl",
        "meta": base_dir / f"{prefix}_meta.json",
    }


def active_meta(spec: ModelSpec) -> dict[str, Any] | None:
    return read_json_file(artifact_paths(MODEL_DIR, spec)["meta"])


def training_params(spec: ModelSpec) -> dict[str, float | int]:
    meta = active_meta(spec) or {}
    return {
        "min_move_bps": float(meta.get("min_move_bps", spec.default_min_move_bps)),
        "train_window_days": int(meta.get("train_window_days", spec.default_window_days)),
        "trade_confidence_threshold": float(meta.get("trade_confidence_threshold", 0.85)),
        "min_trade_gap_minutes": int(meta.get("min_trade_gap_minutes", 0)),
    }


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


def read_registry() -> dict[str, Any]:
    data = read_json_file(REGISTRY_PATH)
    if data is None:
        return {"versions": [], "runs": []}
    data.setdefault("versions", [])
    data.setdefault("runs", [])
    return data


def write_registry(registry: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    tmp_path.replace(REGISTRY_PATH)


def active_snapshot(spec: ModelSpec) -> dict[str, Any]:
    paths = artifact_paths(MODEL_DIR, spec)
    meta = read_json_file(paths["meta"]) or {}
    metrics = metric_summary(meta)
    metrics["forward_validation"] = forward_validation_summary(SYMBOL, spec.duration, meta)
    if spec.family == "enhanced":
        metrics["feature_integrity"] = timeframe_feature_integrity_report()
    return {
        "key": spec.key,
        "label": spec.label,
        "duration": spec.duration,
        "family": spec.family,
        "exists": paths["model"].exists() and paths["meta"].exists(),
        "metrics": metrics,
        "meta": meta,
        "updatedAt": _file_updated_at(paths["meta"]),
    }


def list_versions() -> list[dict[str, Any]]:
    versions = read_registry()["versions"]
    return sorted(versions, key=lambda item: item.get("createdAt", ""), reverse=True)


def archive_active_model(spec: ModelSpec, run_id: str, trigger: str) -> dict[str, Any] | None:
    paths = artifact_paths(MODEL_DIR, spec)
    if not paths["model"].exists() or not paths["meta"].exists():
        return None
    version_id = f"{run_id}-{spec.key}-previous"
    dest_dir = version_dir(spec.key, version_id)
    copy_artifacts(MODEL_DIR, dest_dir, spec)
    meta = read_json_file(dest_dir / paths["meta"].name) or {}
    entry = version_entry(VersionRecord(spec, version_id, dest_dir, "archived", meta, trigger, {}))
    append_version(entry)
    return entry


def record_candidate_version(record: VersionRecord) -> dict[str, Any]:
    dest_dir = version_dir(record.spec.key, record.version_id)
    copy_artifacts(record.artifact_dir, dest_dir, record.spec)
    stored = VersionRecord(
        record.spec,
        record.version_id,
        dest_dir,
        record.status,
        record.metrics,
        record.trigger,
        record.decision,
    )
    entry = version_entry(stored)
    append_version(entry)
    return entry


def activate_version(model_key: str, version_id: str) -> dict[str, Any]:
    spec = spec_by_key(model_key)
    entry = find_version(model_key, version_id)
    if entry.get("status") == "rejected":
        raise ValueError("rejected model versions cannot be activated")
    archive_active_model(spec, _activation_run_id(), "manual-activate")
    copy_artifacts(Path(entry["artifactDir"]), MODEL_DIR, spec)
    mark_active(model_key, version_id)
    return active_snapshot(spec)


def copy_artifacts(source_dir: Path, dest_dir: Path, spec: ModelSpec) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    for path in artifact_paths(source_dir, spec).values():
        if path.exists():
            shutil.copy2(path, dest_dir / path.name)
            copied = True
    if not copied:
        raise FileNotFoundError(f"no artifacts found for {spec.key} in {source_dir}")


def publish_artifacts(source_dir: Path, spec: ModelSpec) -> None:
    copied = False
    for path in artifact_paths(source_dir, spec).values():
        if path.exists():
            _atomic_copy(path, MODEL_DIR / path.name)
            copied = True
    if not copied:
        raise FileNotFoundError(f"no artifacts found for {spec.key} in {source_dir}")


def append_version(entry: dict[str, Any]) -> None:
    registry = read_registry()
    registry["versions"] = _without_duplicate(registry["versions"], entry)
    registry["versions"].append(entry)
    write_registry(registry)


def mark_active(model_key: str, version_id: str) -> None:
    registry = read_registry()
    for entry in registry["versions"]:
        if entry.get("modelKey") != model_key:
            continue
        entry["status"] = "active" if entry.get("versionId") == version_id else _archived_status(entry)
    write_registry(registry)


def record_run(run: dict[str, Any]) -> None:
    registry = read_registry()
    registry["lastRun"] = run
    registry["runs"] = [*registry.get("runs", [])[-19:], run]
    write_registry(registry)


def find_version(model_key: str, version_id: str) -> dict[str, Any]:
    for entry in read_registry()["versions"]:
        if entry.get("modelKey") == model_key and entry.get("versionId") == version_id:
            return entry
    raise ValueError(f"model version not found: {model_key}/{version_id}")


def version_dir(model_key: str, version_id: str) -> Path:
    return ARCHIVE_DIR / model_key / version_id


def version_entry(record: VersionRecord) -> dict[str, Any]:
    return {
        "versionId": record.version_id,
        "modelKey": record.spec.key,
        "label": record.spec.label,
        "duration": record.spec.duration,
        "family": record.spec.family,
        "status": record.status,
        "trigger": record.trigger,
        "artifactDir": str(record.artifact_dir),
        "createdAt": utc_now_iso(),
        "metrics": metric_summary(record.metrics),
        "decision": record.decision,
    }


def _without_duplicate(versions: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in versions
        if not (
            item.get("modelKey") == entry.get("modelKey")
            and item.get("versionId") == entry.get("versionId")
        )
    ]


def _archived_status(entry: dict[str, Any]) -> str:
    return "rejected" if entry.get("status") == "rejected" else "archived"


def _file_updated_at(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _activation_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(target)
