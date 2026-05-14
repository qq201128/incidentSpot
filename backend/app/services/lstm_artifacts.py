from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "lstm"


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


def artifact_paths(symbol: str, duration: str, root: Path | None = None) -> LstmArtifactPaths:
    base = (root or MODEL_DIR) / symbol.strip().upper() / duration.strip()
    return LstmArtifactPaths(
        root=base,
        model=base / "model.pt",
        report=base / "training_report.json",
        scaler=base / "scaler.json",
        features=base / "features.json",
        version=base / "model_version.json",
        status=base / "status.json",
        attempt=base / "last_training_attempt.json",
    )


def artifact_paths_for_root(root: Path) -> LstmArtifactPaths:
    return LstmArtifactPaths(
        root=root,
        model=root / "model.pt",
        report=root / "training_report.json",
        scaler=root / "scaler.json",
        features=root / "features.json",
        version=root / "model_version.json",
        status=root / "status.json",
        attempt=root / "last_training_attempt.json",
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"LSTM artifact is not a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


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
