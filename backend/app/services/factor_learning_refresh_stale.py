from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

REFRESH_TASK_STALE_SECONDS = 600


def is_refresh_task_stale(task: dict[str, Any], *, now: datetime | None = None) -> bool:
    status = str(task.get("status") or "")
    if status not in {"queued", "running"}:
        return False
    stamp = str(task.get("updatedAt") or "")
    if not stamp:
        return True
    try:
        started = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return (current - started).total_seconds() >= REFRESH_TASK_STALE_SECONDS


def stale_refresh_task_error(task: dict[str, Any]) -> str:
    stamp = str(task.get("updatedAt") or "")
    action = "复盘+联网挖掘" if task.get("runAgent") else "本地复盘"
    return (
        f"因子学习{action}超时或中断（refreshTask 停留在 {task.get('status') or '未知'}）。"
        f" 更新于 {stamp or '未知'}，超过 {REFRESH_TASK_STALE_SECONDS} 秒未完成。"
        " 请重新点击刷新；若反复失败，请查看后台日志或重启后端。"
    )
