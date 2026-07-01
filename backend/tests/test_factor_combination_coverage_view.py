from __future__ import annotations

from app.services import factor_combination_coverage_view as view


def test_data_coverage_summary_uses_blocking_gap_rules(monkeypatch) -> None:
    monkeypatch.setattr(
        view,
        "build_data_coverage_report",
        lambda _options: {
            "mainRange": {"rowCount": 100, "status": "healthy"},
            "interval": "10m",
            "tables": [
                {
                    "table": "klines",
                    "rows": [
                        {"interval": "1m", "status": "partial", "missingReason": None},
                        {"interval": "10m", "status": "healthy", "missingReason": None},
                    ],
                },
                {
                    "table": "klines_multi",
                    "rows": [{"interval": "10m", "status": "partial", "coveragePct": 99.0}],
                },
                {
                    "table": "orderbook_ticks",
                    "rows": [{"status": "unavailable", "missingReason": "no_open_time_column"}],
                },
                {
                    "table": "funding_features",
                    "rows": [{"status": "partial", "coveragePct": 99.0}],
                },
            ],
        },
    )

    summary = view.data_coverage_summary("BTCUSDT", "10m")

    assert summary["missingFeatureSources"] == []


def test_data_coverage_summary_keeps_true_missing_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        view,
        "build_data_coverage_report",
        lambda _options: {
            "mainRange": {"rowCount": 100, "status": "healthy"},
            "interval": "10m",
            "tables": [
                {
                    "table": "klines_multi",
                    "rows": [{"interval": "10m", "status": "missing", "missingReason": "no_rows"}],
                },
            ],
        },
    )

    summary = view.data_coverage_summary("BTCUSDT", "10m")

    assert summary["missingFeatureSources"] == [
        {
            "table": "klines_multi",
            "status": "missing",
            "missingReason": "no_rows",
            "group": "10m",
            "coveragePct": None,
            "rowIndex": 0,
        }
    ]
