from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_REJECTED = "rejected"
JOB_STATUS_CANCELLED = "cancelled"

JOB_STAGE_QUEUED = "queued"
JOB_STAGE_COARSE = "coarse"
JOB_STAGE_FULL = "full"
JOB_STAGE_WALK_FORWARD = "walk_forward"
JOB_STAGE_PAPER_LIVE = "paper_live"
JOB_STAGE_SETTLED_REVIEW = "settled_review"

DEFAULT_MODEL_SEARCH_PRIORITY = 100
DEFAULT_MAX_RUNNING_JOBS = 1
DEFAULT_STALE_AFTER_SECONDS = 3600
JOB_ID_NAMESPACE = "5f0e23bf-06fc-4bb2-a68d-1d777f69aa08"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_loads(raw: Any) -> Any:
    if not raw:
        return None
    return json.loads(str(raw))
