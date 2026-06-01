from __future__ import annotations

import json
import os
import errno
import hashlib
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_MODEL_DIR = MODELS_ROOT / "ml"
LEGACY_LSTM_MODEL_DIR = MODELS_ROOT / "lstm"
MODEL_DIR = ML_MODEL_DIR / "lstm"
JSON_ARTIFACT_LOCK_DIR = (
    Path(tempfile.gettempdir()) / f"incidentSpot-json-artifact-locks-{hashlib.sha256(str(PROJECT_ROOT).encode('utf-8')).hexdigest()[:12]}"
)
LOCK_BYTE_COUNT = 1
LOCK_FILE_PREFIX = "."
LOCK_FILE_SUFFIX = ".lock"
JSON_REPLACE_MAX_ATTEMPTS = 5
JSON_REPLACE_RETRY_SECONDS = 0.05
LOCK_ACQUIRE_MAX_ATTEMPTS = 5
LOCK_ACQUIRE_RETRY_SECONDS = 0.05
LOCK_OPEN_MAX_ATTEMPTS = 240
LOCK_OPEN_RETRY_SECONDS = 0.05
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}


@dataclass(frozen=True)
class LstmArtifactPaths:
    root: Path
    model: Path
    report: Path
    scaler: Path
    features: Path
    version: Path
    status: Path
    attempt: Path


def artifact_paths(
    symbol: str,
    duration: str,
    root: Path | None = None,
    *,
    family: str = "lstm",
) -> LstmArtifactPaths:
    base = artifact_base_path(symbol, duration, root, family=family)
    model_name = "model.pt" if family in {"lstm", "gru", "cnn", "transformer"} else "model.joblib"
    return LstmArtifactPaths(
        root=base,
        model=base / model_name,
        report=base / "training_report.json",
        scaler=base / "scaler.json",
        features=base / "features.json",
        version=base / "model_version.json",
        status=base / "status.json",
        attempt=base / "last_training_attempt.json",
    )


def artifact_base_path(
    symbol: str,
    duration: str,
    root: Path | None = None,
    *,
    family: str = "lstm",
) -> Path:
    normalized = family.strip().lower()
    if root is not None:
        return root / symbol.strip().upper() / duration.strip()
    sym = symbol.strip().upper()
    dur = duration.strip()
    if normalized == "lstm":
        _migrate_legacy_lstm_dir(sym, dur)
    return ML_MODEL_DIR / normalized / sym / dur


def artifact_paths_for_root(root: Path) -> LstmArtifactPaths:
    model_name = "model.joblib" if root.name.endswith("_joblib") else "model.pt"
    return LstmArtifactPaths(
        root=root,
        model=root / model_name,
        report=root / "training_report.json",
        scaler=root / "scaler.json",
        features=root / "features.json",
        version=root / "model_version.json",
        status=root / "status.json",
        attempt=root / "last_training_attempt.json",
    )


def artifact_paths_for_family_root(root: Path, family: str) -> LstmArtifactPaths:
    model_name = "model.pt" if family in {"lstm", "gru", "cnn", "transformer"} else "model.joblib"
    return LstmArtifactPaths(
        root=root,
        model=root / model_name,
        report=root / "training_report.json",
        scaler=root / "scaler.json",
        features=root / "features.json",
        version=root / "model_version.json",
        status=root / "status.json",
        attempt=root / "last_training_attempt.json",
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.parent.exists():
        return None
    with _json_artifact_lock(path):
        return _read_json_unlocked(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _json_artifact_lock(path):
        _write_json_unlocked(path, payload)


def update_json(
    path: Path,
    updater: Callable[[dict[str, Any] | None], dict[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _json_artifact_lock(path):
        payload = updater(_read_json_unlocked(path))
        _write_json_unlocked(path, payload)
        return payload


def _read_json_unlocked(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"LSTM artifact is not a JSON object: {path}")
    return payload


def _write_json_unlocked(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    _replace_json_file(tmp, path)


def _replace_json_file(source: Path, target: Path) -> None:
    for attempt in range(1, JSON_REPLACE_MAX_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt >= JSON_REPLACE_MAX_ATTEMPTS:
                raise
            time.sleep(JSON_REPLACE_RETRY_SECONDS)


@contextmanager
def _json_artifact_lock(path: Path) -> Iterator[None]:
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = _process_lock(lock_path)
    with process_lock:
        with _open_lock_file(lock_path) as handle:
            _ensure_lock_byte(handle)
            _lock_file(handle)
            try:
                yield
            finally:
                _unlock_file(handle)


def _lock_path(path: Path) -> Path:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return JSON_ARTIFACT_LOCK_DIR / f"{LOCK_FILE_PREFIX}{digest}{LOCK_FILE_SUFFIX}"


def _open_lock_file(lock_path: Path):
    for attempt in range(1, LOCK_OPEN_MAX_ATTEMPTS + 1):
        try:
            return lock_path.open("a+b")
        except PermissionError:
            if attempt >= LOCK_OPEN_MAX_ATTEMPTS:
                raise
            time.sleep(LOCK_OPEN_RETRY_SECONDS)


def _process_lock(lock_path: Path) -> threading.RLock:
    key = lock_path.resolve()
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _ensure_lock_byte(handle) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() > 0:
        return
    handle.write(b"\0")
    handle.flush()


def _lock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        _lock_windows_file(handle, msvcrt)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _lock_windows_file(handle, msvcrt_module) -> None:
    for attempt in range(1, LOCK_ACQUIRE_MAX_ATTEMPTS + 1):
        try:
            msvcrt_module.locking(handle.fileno(), msvcrt_module.LK_LOCK, LOCK_BYTE_COUNT)
            return
        except OSError as exc:
            if exc.errno != errno.EDEADLK or attempt >= LOCK_ACQUIRE_MAX_ATTEMPTS:
                raise
            time.sleep(LOCK_ACQUIRE_RETRY_SECONDS)


def _unlock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, LOCK_BYTE_COUNT)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publish_artifacts(staging: LstmArtifactPaths, active: LstmArtifactPaths) -> None:
    active.root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staging.model, active.model)
    for source, target in (
        (staging.scaler, active.scaler),
        (staging.features, active.features),
        (staging.version, active.version),
        (staging.report, active.report),
        (staging.status, active.status),
    ):
        payload = require_json(source, source.name)
        write_json(target, payload)


def require_json(path: Path, name: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload is None:
        raise ValueError(f"missing LSTM {name} artifact: {path}")
    return payload


def required_artifacts_exist(paths: LstmArtifactPaths) -> bool:
    return all(
        path.exists()
        for path in (paths.model, paths.report, paths.scaler, paths.features, paths.version)
    )


def _migrate_legacy_lstm_dir(symbol: str, duration: str) -> None:
    legacy_base = LEGACY_LSTM_MODEL_DIR / symbol / duration
    active_base = MODEL_DIR / symbol / duration
    if not legacy_base.exists() or active_base.exists():
        return
    active_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_base), str(active_base))
