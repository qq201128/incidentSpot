from __future__ import annotations

from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report


def data_coverage_summary(symbol: str, duration: str) -> dict:
    report = build_data_coverage_report(CoverageOptions(symbol=symbol, interval=duration))
    return {
        "mainRange": report["mainRange"],
        "missingFeatureSources": _missing_feature_sources(report["tables"]),
    }


def _missing_feature_sources(tables: list[dict]) -> list[dict]:
    missing = []
    for table in tables:
        for row in table.get("rows") or []:
            if row.get("status") in {"healthy"}:
                continue
            missing.append(
                {
                    "table": table.get("table"),
                    "status": row.get("status"),
                    "coveragePct": row.get("coveragePct"),
                    "missingReason": row.get("missingReason"),
                }
            )
    return missing
