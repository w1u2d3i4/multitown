import csv
import json
from pathlib import Path

from general_mas_bench.baseline_report import build_baseline_report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_three_way_baseline_report_is_paired_and_auditable(tmp_path: Path) -> None:
    task_ids = ["task-b", "task-a"]
    directories = {method: tmp_path / method.lower() for method in ("Solo", "A4", "A8")}
    for method, path in directories.items():
        path.mkdir()
        _write_json(path / "config.json", {"tasks": task_ids, "method": method})
        _write_jsonl(path / "system_metrics.jsonl", [
            {"elapsed_s": 0, "gpu_power_w": 20},
            {"elapsed_s": 10, "gpu_power_w": 30},
        ])
    common = {
        "category": "Test", "difficulty": "easy", "request_errors": 0,
        "role_activations": {}, "route": "test",
    }
    values = {
        "Solo": [(False, 0.4, 100, 10), (True, 1.0, 200, 20)],
        "A4": [(False, 0.5, 150, 12), (True, 1.0, 250, 25)],
        "A8": [(True, 1.0, 50, 5), (False, 0.5, 80, 8)],
    }
    for method, rows in values.items():
        _write_jsonl(directories[method] / "results.jsonl", [
            {
                **common, "task_id": task_id, "passed": passed,
                "partial_score": partial, "total_tokens": tokens,
                "latency_s": latency,
            }
            for task_id, (passed, partial, tokens, latency) in zip(
                ["task-a", "task-b"], rows, strict=True
            )
        ])

    output = tmp_path / "report"
    report = build_baseline_report(
        solo_dir=directories["Solo"], a4_dir=directories["A4"],
        a8_dir=directories["A8"], output=output, expected_count=2,
        bootstrap_samples=100, seed=7,
    )

    assert report["schema_version"] == "general-mas-teambench-baseline-comparison-v1"
    assert report["paired_tasks"] == 2
    assert set(report["methods"]) == {"Solo", "A4", "A8"}
    assert "A8_minus_Solo" in report["pairwise"]
    assert report["pairwise"]["A8_minus_A4"]["mean_token_reduction"] == 1 - 65 / 200
    with (output / "paired_tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == task_ids
    assert (output / "quality_cost_three_way.png").stat().st_size > 0
    assert (output / "REPORT.md").stat().st_size > 0
