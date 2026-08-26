import csv
import json
from pathlib import Path

import pytest
from general_mas_bench.paired_report import build_paired_report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    task_ids = ["task-b", "task-a"]
    config = {
        "tasks": task_ids,
        "split": "dev",
        "split_sha256": "split-hash",
        "seed_override": 5,
        "sampling_seed": 9,
        "temperature": 0,
        "max_tokens": 2048,
        "task_instances_sha256": "instances-hash",
        "source": {"revision": "upstream", "dirty": False},
        "docker_image_id": "sha256:image",
        "strong": {"model": "strong"},
        "weak": {"model": "weak"},
        "runner_source": {"revision": "runner", "dirty": False},
        "controller_sha256": "controller-hash",
    }
    common = {
        "category": "Test",
        "difficulty": "easy",
        "request_errors": 0,
        "role_activations": {},
        "route": "test",
    }
    for path, method, values in (
        (baseline, "PlanExecute", [(False, 0.4, 100, 10), (True, 1.0, 200, 20)]),
        (candidate, "StrongPlanExecute", [(True, 1.0, 80, 8), (False, 0.5, 100, 10)]),
    ):
        path.mkdir()
        _write_json(path / "config.json", {**config, "method": method})
        _write_jsonl(
            path / "results.jsonl",
            [
                {
                    **common,
                    "task_id": task_id,
                    "passed": passed,
                    "partial_score": partial,
                    "total_tokens": tokens,
                    "latency_s": latency,
                }
                for task_id, (passed, partial, tokens, latency) in zip(
                    ["task-a", "task-b"], values, strict=True
                )
            ],
        )
        _write_jsonl(
            path / "system_metrics.jsonl",
            [
                {"elapsed_s": 0, "gpu_power_w": 20},
                {"elapsed_s": 10, "gpu_power_w": 30},
            ],
        )
    return baseline, candidate


def test_paired_report_is_matched_and_auditable(tmp_path: Path) -> None:
    baseline, candidate = _fixture(tmp_path)
    output = tmp_path / "report"
    report = build_paired_report(
        baseline_dir=baseline,
        candidate_dir=candidate,
        baseline_label="PlanExecute",
        candidate_label="StrongPlanExecute",
        output=output,
        expected_count=2,
        bootstrap_samples=100,
        seed=7,
    )

    assert report["schema_version"] == "general-mas-teambench-paired-comparison-v1"
    assert report["paired_tasks"] == 2
    assert report["methods"]["PlanExecute"]["passes"] == 1
    assert report["paired"]["mean_token_reduction"] == 0.4
    assert report["paired"]["partial_score_outcomes"] == {
        "candidate_better": 1,
        "tied": 0,
        "baseline_better": 1,
    }
    with (output / "paired_tasks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == ["task-b", "task-a"]
    assert (output / "quality_cost.png").stat().st_size > 0
    assert (output / "REPORT.md").stat().st_size > 0


def test_paired_report_rejects_unmatched_config(tmp_path: Path) -> None:
    baseline, candidate = _fixture(tmp_path)
    config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    config["sampling_seed"] = 10
    _write_json(candidate / "config.json", config)

    with pytest.raises(ValueError, match="unmatched experiment configuration"):
        build_paired_report(
            baseline_dir=baseline,
            candidate_dir=candidate,
            baseline_label="PlanExecute",
            candidate_label="StrongPlanExecute",
            output=tmp_path / "report",
            expected_count=2,
            bootstrap_samples=100,
            seed=7,
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("split_sha256", "other-split"),
        ("seed_override", 6),
        ("task_instances_sha256", "other-instances"),
        ("docker_image_id", "sha256:other-image"),
        ("runner_source", {"revision": "other", "dirty": False}),
        ("controller_sha256", "other-controller"),
    ],
)
def test_paired_report_rejects_hard_provenance_mismatch(
    tmp_path: Path, field: str, replacement: object
) -> None:
    baseline, candidate = _fixture(tmp_path)
    config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    config[field] = replacement
    _write_json(candidate / "config.json", config)

    with pytest.raises(ValueError, match="unmatched experiment configuration"):
        build_paired_report(
            baseline_dir=baseline,
            candidate_dir=candidate,
            baseline_label="PlanExecute",
            candidate_label="StrongPlanExecute",
            output=tmp_path / "report",
            expected_count=2,
            bootstrap_samples=100,
            seed=7,
        )
