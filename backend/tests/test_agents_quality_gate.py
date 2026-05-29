from __future__ import annotations

from pathlib import Path

from scripts import agents_quality_gate as gate


def test_quality_gate_detects_function_length_and_empty_except(tmp_path: Path) -> None:
    target = tmp_path / "backend" / "app" / "services" / "bad_module.py"
    target.parent.mkdir(parents=True)
    body = "\n".join(f"    value_{index} = {index}" for index in range(52))
    target.write_text(
        "def too_long(a, b, c, d):\n"
        f"{body}\n"
        "    try:\n"
        "        return value_0\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )

    violations = gate.scan_project(tmp_path)
    codes = {item.code for item in violations}

    assert "FUNCTION_LENGTH" in codes
    assert "POSITIONAL_PARAMS" in codes
    assert "EMPTY_EXCEPT_PASS" in codes


def test_quality_gate_baseline_accepts_existing_violation(tmp_path: Path) -> None:
    violation = gate.Violation("FILE_SIZE", "backend/app/services/large.py", 1, "file has 301 lines")
    baseline = tmp_path / "baseline.json"

    gate.write_baseline(baseline, [violation])

    assert violation.key in gate.read_baseline(baseline)
