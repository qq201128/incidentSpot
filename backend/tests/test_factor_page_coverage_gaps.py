from __future__ import annotations

from app.services.factor_page_alerts import coverage_gaps


def test_coverage_gaps_skip_cross_interval_klines_and_ticks() -> None:
    report = {
        "mainRange": {"rowCount": 100, "status": "healthy"},
        "interval": "10m",
        "tables": [
            {
                "table": "klines",
                "rows": [
                    {"interval": "1m", "status": "partial", "missingReason": "partial"},
                    {"interval": "10m", "status": "healthy", "missingReason": None},
                ],
            },
            {
                "table": "orderbook_ticks",
                "rows": [{"status": "unavailable", "missingReason": "no_open_time_column"}],
            },
            {
                "table": "orderbook_features",
                "rows": [{"status": "partial", "missingReason": "partial", "coveragePct": 12.0}],
            },
        ],
    }

    gaps = coverage_gaps(report, primary_interval="10m")

    assert len(gaps) == 1
    assert gaps[0]["table"] == "orderbook_features"


def test_coverage_gaps_flags_low_orderbook_coverage() -> None:
    report = {
        "mainRange": {"rowCount": 100, "status": "healthy"},
        "interval": "10m",
        "tables": [
            {
                "table": "orderbook_features",
                "rows": [{"status": "partial", "missingReason": "partial", "coveragePct": 12.0}],
            },
        ],
    }

    gaps = coverage_gaps(report, primary_interval="10m")

    assert len(gaps) == 1
    assert gaps[0]["table"] == "orderbook_features"
