from __future__ import annotations

from pathlib import Path

from app.services.lstm_artifacts import write_json
from app.services.model_family_candidates import candidate_library_path, candidate_progress_path
from app.services.model_family_config import normalize_model_family


def reset_model_candidate_history(
    family: str,
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, dict]:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    dur = duration.strip()
    library = _empty_library(selected, sym, dur)
    progress = _empty_progress(selected, sym, dur)
    write_json(candidate_library_path(selected, sym, dur, artifact_root=artifact_root), library)
    write_json(candidate_progress_path(selected, sym, dur, artifact_root=artifact_root), progress)
    return {"library": library, "progress": progress}


def _empty_library(family: str, symbol: str, duration: str) -> dict:
    return {
        "modelFamily": family,
        "symbol": symbol,
        "duration": duration,
        "updatedAt": None,
        "total": 0,
        "records": [],
    }


def _empty_progress(family: str, symbol: str, duration: str) -> dict:
    return {
        "status": "idle",
        "modelFamily": family,
        "symbol": symbol,
        "duration": duration,
        "total": 0,
        "completed": 0,
        "percent": 0.0,
        "counts": _empty_counts(),
        "latestCompleted": None,
        "recent": [],
    }


def _empty_counts() -> dict[str, int]:
    return {
        "tradeActive": 0,
        "shadowActive": 0,
        "initialBaseline": 0,
        "validationFailed": 0,
        "insufficientSamples": 0,
        "failed": 0,
    }
