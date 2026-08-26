import json
from pathlib import Path

from general_mas_bench.capacity_route_report import build_capacity_route_report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _make_run(
    root: Path,
    *,
    method: str,
    values: list[tuple[bool, float, int, float, str]],
) -> Path:
    root.mkdir()
    tasks = ["task-a", "task-b"]
    config = {
        "method": method,
        "tasks": tasks,
        "split": "test",
        "split_sha256": "split-hash",
        "seed_override": 5,
        "sampling_seed": 11,
        "temperature": 0,
        "max_tokens": 2048,
        "task_instances_sha256": "instance-hash",
        "source": {"revision": "upstream", "dirty": False},
        "docker_image_id": "sha256:image",
        "strong": {"model": "strong"},
        "weak": {"model": "weak"},
        "runner_source": {"revision": "runner", "dirty": False},
        "controller_sha256": "controller-hash",
    }
    _write_json(root / "config.json", config)
    _write_jsonl(
        root / "results.jsonl",
        [
            {
                "task_id": task,
                "category": "Test",
                "difficulty": "easy",
                "passed": passed,
                "partial_score": partial,
                "total_tokens": tokens,
                "latency_s": latency,
                "route": route,
                "role_activations": {},
                "request_errors": 0,
                "failure_modes": [],
            }
            for task, (passed, partial, tokens, latency, route) in zip(
                tasks, values, strict=True
            )
        ],
    )
    _write_jsonl(
        root / "system_metrics.jsonl",
        [
            {"elapsed_s": 0, "gpu_power_w": 20},
            {"elapsed_s": 10, "gpu_power_w": 30},
        ],
    )
    return root


def test_capacity_route_report_separates_route_effect_and_repeat_noise(
    tmp_path: Path,
) -> None:
    plan = _make_run(
        tmp_path / "plan",
        method="PlanExecute",
        values=[
            (False, 0.4, 100, 10, "baseline"),
            (False, 0.5, 100, 10, "baseline"),
        ],
    )
    solo = _make_run(
        tmp_path / "solo",
        method="Solo",
        values=[
            (False, 0.6, 180, 15, "solo"),
            (False, 0.5, 180, 15, "solo"),
        ],
    )
    capacity = _make_run(
        tmp_path / "capacity",
        method="MTCapacityRoute",
        values=[
            (True, 1.0, 90, 9, "capacity_route:strong_executor"),
            (False, 0.5, 90, 9, "capacity_route:weak_executor"),
        ],
    )

    report = build_capacity_route_report(
        plan_execute_dir=plan,
        solo_dir=solo,
        capacity_route_dir=capacity,
        output=tmp_path / "report",
        expected_count=2,
        bootstrap_samples=100,
        seed=1,
    )

    assert report["claim_checks"]["frozen_confirmation_gate"] is True
    assert report["claim_checks"]["strongest_local_point_on_pass_partial_tokens"]
    assert report["claim_checks"]["literature_sota_supported"] is False
    assert report["route_audit"]["strong_route"]["n"] == 1
    assert report["route_audit"]["weak_repeat"]["n"] == 1
    assert (
        report["route_audit"]["baseline_anchored_counterfactual"]["summary"]["passes"]
        == 1
    )
    assert report["grader_timeout_rows"] == []
    assert (tmp_path / "report" / "summary.json").is_file()
    assert (tmp_path / "report" / "REPORT.md").is_file()
