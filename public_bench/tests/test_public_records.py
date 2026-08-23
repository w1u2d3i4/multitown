from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"


def test_formal_summary_matches_frozen_teambench_claim() -> None:
    summary = json.loads(
        (RECORDS / "teambench-test-v1.2-summary.json").read_text(encoding="utf-8")
    )
    assert summary["paired_tasks"] == 89
    assert summary["A4"]["passes"] == 14
    assert summary["A8"]["passes"] == 11
    assert summary["A4"]["mean_partial_score"] == pytest.approx(0.6337456929)
    assert summary["A8"]["mean_partial_score"] == pytest.approx(0.5825097378)
    assert summary["mean_token_reduction"] == pytest.approx(0.3705713492)
    paired = summary["paired_partial_difference_a8_minus_a4"]
    assert paired["mean"] == pytest.approx(-0.0512359551)
    assert paired["ci95_lower"] == pytest.approx(-0.0895124157)
    assert paired["ci95_upper"] == pytest.approx(-0.0167756742)
    assert summary["preregistered_gate"]["passed"] is False


def test_public_paired_table_has_89_unique_tasks() -> None:
    with (RECORDS / "teambench-test-v1.2-paired.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    task_ids = [row["task_id"] for row in rows]
    assert len(rows) == 89
    assert len(set(task_ids)) == 89
    assert all(row["a4_route"] and row["a8_route"] for row in rows)


def test_public_records_do_not_expose_machine_local_paths() -> None:
    for path in RECORDS.iterdir():
        if path.suffix not in {".json", ".md", ".csv"}:
            continue
        content = path.read_text(encoding="utf-8")
        assert "/home/dilab/" not in content
        assert "USTC@" not in content
