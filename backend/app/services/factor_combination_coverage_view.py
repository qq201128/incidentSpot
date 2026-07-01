from __future__ import annotations

from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report
from app.services.factor_page_alerts import coverage_gaps


def data_coverage_summary(symbol: str, duration: str) -> dict:
    report = build_data_coverage_report(CoverageOptions(symbol=symbol, interval=duration, primary_only=True))
    return {
        "mainRange": report["mainRange"],
        "missingFeatureSources": coverage_gaps(report, primary_interval=duration),
    }
