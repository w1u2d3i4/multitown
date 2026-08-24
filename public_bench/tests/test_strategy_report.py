import csv
import json
from pathlib import Path

from general_mas_bench.strategy_report import METHODS, build_strategy_report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def test_five_strategy_report_is_paired_and_auditable(tmp_path: Path) -> None:
    task_ids = ["task-b", "task-a"]
    directories = {method: tmp_path / method.lower() for method in METHODS}
    values = {
        "Solo": [(True, 1.0, 200, 20), (False, 0.4, 100, 10)],
        "PlanExecute": [(True, 1.0, 160, 16), (False, 0.5, 90, 9)],
        "ExecuteReview": [(False, 0.7, 140, 14), (True, 1.0, 110, 11)],
        "A4": [(True, 1.0, 250, 25), (False, 0.5, 150, 12)],
        "A8": [(False, 0.5, 80, 8), (True, 1.0, 50, 5)],
    }
    for method, path in directories.items():
        path.mkdir()
        _write_json(path / "config.json", {"tasks": task_ids, "method": method})
        _write_jsonl(
            path / "system_metrics.jsonl",
            [
                {"elapsed_s": 0, "gpu_power_w": 20},
                {"elapsed_s": 10, "gpu_power_w": 30},
            ],
        )
        _write_jsonl(
            path / "results.jsonl",
            [
                {
                    "task_id": task_id,
                    "category": "Test",
                    "difficulty": "easy",
                    "request_errors": 0,
                    "role_activations": {},
                    "route": "test",
                    "passed": passed,
                    "partial_score": partial,
                    "total_tokens": tokens,
                    "latency_s": latency,
                }
                for task_id, (passed, partial, tokens, latency) in zip(
                    ["task-a", "task-b"], values[method], strict=True
                )
            ],
        )

    output = tmp_path / "report"
    report = build_strategy_report(
        solo_dir=directories["Solo"],
        plan_execute_dir=directories["PlanExecute"],
        execute_review_dir=directories["ExecuteReview"],
        a4_dir=directories["A4"],
        a8_dir=directories["A8"],
        output=output,
        expected_count=2,
        bootstrap_samples=100,
        seed=11,
    )

    assert report["schema_version"] == "general-mas-teambench-strategy-comparison-v2"
    assert report["comparison_scope"] == "same_harness_all_metrics"
    assert report["paired_tasks"] == 2
    assert set(report["methods"]) == set(METHODS)
    assert len(report["pairwise"]) == 10
    assert "A8_minus_PlanExecute" in report["pairwise"]
    assert report["pairwise"]["A8_minus_A4"]["mean_token_reduction"] == 1 - 65 / 200
    with (output / "paired_tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == task_ids
    assert (output / "quality_cost_five_way.png").stat().st_size > 0
    assert (output / "REPORT.md").stat().st_size > 0


def test_report_rejects_mixed_revisions_unless_quality_only(tmp_path: Path) -> None:
    directories = {method: tmp_path / method.lower() for method in METHODS}
    for index, (method, path) in enumerate(directories.items()):
        path.mkdir()
        _write_json(
            path / "config.json",
            {
                "tasks": ["task-a"],
                "method": method,
                "source": {"revision": f"rev-{index}", "dirty": False},
                "split_sha256": "split",
                "docker_image_id": "image",
                "temperature": 0,
                "max_tokens": 100,
                "strong": {"model": "strong"},
                "weak": {"model": "weak"},
            },
        )
        _write_jsonl(path / "system_metrics.jsonl", [])
        _write_jsonl(
            path / "results.jsonl",
            [
                {
                    "task_id": "task-a",
                    "request_errors": 0,
                    "role_activations": {},
                    "passed": True,
                    "partial_score": 1.0,
                    "total_tokens": 10,
                    "latency_s": 1,
                }
            ],
        )

    kwargs = {
        "solo_dir": directories["Solo"],
        "plan_execute_dir": directories["PlanExecute"],
        "execute_review_dir": directories["ExecuteReview"],
        "a4_dir": directories["A4"],
        "a8_dir": directories["A8"],
        "expected_count": 1,
        "bootstrap_samples": 10,
        "seed": 1,
    }
    import pytest

    with pytest.raises(ValueError, match="runtime provenance differs"):
        build_strategy_report(output=tmp_path / "strict", **kwargs)

    report = build_strategy_report(
        output=tmp_path / "quality", quality_only=True, **kwargs
    )
    assert report["comparison_scope"] == "quality_and_tokens_only"
    assert report["provenance_compatibility"]["runtime_compatible"] is False
    assert report["system_monitoring"] == {method: None for method in METHODS}
    assert "mean_latency_reduction" not in report["pairwise"]["A8_minus_A4"]
    header = (tmp_path / "quality" / "paired_tasks.csv").read_text().splitlines()[0]
    assert "latency" not in header
