import csv
import json
from pathlib import Path

from general_mas_bench.report import build_report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_report_writes_paired_table_and_monitoring_plots(tmp_path: Path) -> None:
    task_ids = ["task-b", "task-a"]
    a4_dir = tmp_path / "a4"
    a8_dir = tmp_path / "a8"
    output = tmp_path / "report"
    a4_dir.mkdir()
    a8_dir.mkdir()
    _write_json(a4_dir / "config.json", {"tasks": task_ids, "method": "A4"})
    _write_json(a8_dir / "config.json", {"tasks": task_ids, "method": "A8"})
    common = {
        "category": "Test",
        "difficulty": "easy",
        "request_errors": 0,
        "role_activations": {"executor": 1},
    }
    _write_jsonl(a4_dir / "results.jsonl", [
        {**common, "task_id": "task-a", "passed": False, "partial_score": 0.4, "total_tokens": 100, "latency_s": 10, "route": "fixed"},
        {**common, "task_id": "task-b", "passed": True, "partial_score": 1.0, "total_tokens": 200, "latency_s": 20, "route": "fixed"},
    ])
    _write_jsonl(a8_dir / "results.jsonl", [
        {**common, "task_id": "task-a", "passed": True, "partial_score": 1.0, "total_tokens": 50, "latency_s": 5, "route": "weak"},
        {**common, "task_id": "task-b", "passed": False, "partial_score": 0.5, "total_tokens": 80, "latency_s": 8, "route": "strong"},
    ])
    metrics = [
        {"elapsed_s": 0, "cpu_percent": 10, "ram_used_bytes": 2 * 1024**3, "gpu_power_w": 20},
        {"elapsed_s": 10, "cpu_percent": 20, "ram_used_bytes": 3 * 1024**3, "gpu_power_w": 30},
    ]
    _write_jsonl(a4_dir / "system_metrics.jsonl", metrics)
    _write_jsonl(a8_dir / "system_metrics.jsonl", metrics)

    report = build_report(
        a4_dir=a4_dir,
        a8_dir=a8_dir,
        output=output,
        expected_count=2,
        allow_incomplete=False,
        bootstrap_samples=100,
        seed=7,
    )

    assert report["schema_version"] == "general-mas-teambench-comparison-v2"
    assert report["mean_token_reduction"] == 1 - 65 / 150
    assert report["paired_partial_outcomes"] == {"a8_better": 1, "tied": 0, "a4_better": 1}
    assert report["system_monitoring"]["A4"]["samples"] == 2
    with (output / "paired_tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == task_ids
    for name in (
        "quality_cost.png", "category_partial.png", "cumulative_tokens.png",
        "per_task_metrics.png", "paired_quality_difference.png", "system_monitoring.png",
    ):
        assert (output / name).stat().st_size > 0
