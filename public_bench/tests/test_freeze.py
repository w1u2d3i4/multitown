from pathlib import Path

from general_mas_bench.freeze import resolve_test_task, round_robin_stratified


def test_round_robin_is_deterministic_and_spans_categories() -> None:
    rows = [
        {"task_id": f"a{i}", "category": "A"} for i in range(4)
    ] + [
        {"task_id": f"b{i}", "category": "B"} for i in range(4)
    ]
    first = round_robin_stratified(rows, 4, "seed")
    second = round_robin_stratified(list(reversed(rows)), 4, "seed")
    assert first == second
    assert {row["category"] for row in first} == {"A", "B"}


def test_resolver_prefers_exact_and_supports_seeded_fallback(tmp_path: Path) -> None:
    exact = tmp_path / "EXACT"
    exact.mkdir()
    (exact / "task.yaml").write_text("task_id: EXACT\n")
    (exact / "grade.sh").write_text("#!/bin/sh\n")
    assert resolve_test_task(tmp_path, "EXACT")["source_task"] == "EXACT"
    assert resolve_test_task(tmp_path, "EXACT")["generated_at_runtime"] is True
    assert resolve_test_task(
        tmp_path, "EXACT", has_generator=False
    )["generated_at_runtime"] is False

    seeded = tmp_path / "SEEDED_seed0"
    (seeded / "workspace").mkdir(parents=True)
    (seeded / "task.yaml").write_text("task_id: SEEDED\n")
    (seeded / "spec.md").write_text("spec")
    (seeded / "brief.md").write_text("brief")
    (seeded / "workspace" / "check_solution.py").write_text("print('ok')\n")
    value = resolve_test_task(tmp_path, "SEEDED")
    assert value["source_task"] == "SEEDED_seed0"
    assert value["grader"] == "workspace/check_solution.py"
