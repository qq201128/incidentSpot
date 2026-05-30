from __future__ import annotations


def exception_details(exc: Exception, stage: str) -> dict[str, str]:
    return {"stage": stage, "error": str(exc), "exceptionType": type(exc).__name__}
