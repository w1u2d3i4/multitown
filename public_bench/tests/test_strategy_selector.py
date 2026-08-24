from __future__ import annotations

import json
from pathlib import Path

import pytest

from general_mas_bench.strategy_selector import (
    METHODS,
    fit_policy,
    leave_one_out_summary,
    load_paired_runs,
    outcome_reward,
    select_method,
)


def _row(task_id: str, category: str, partial: float, tokens: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "category": category,
        "passed": partial == 1.0,
        "partial_score": partial,
        "total_tokens": tokens,
        "request_errors": 0,
    }


def _matrix() -> tuple[list[str], dict[str, dict[str, dict[str, object]]]]:
    task_ids = ["safe-1", "safe-2", "hard-1", "hard-2"]
    rows = {method: {} for method in METHODS}
    for task_id in task_ids:
        category = "Safe" if task_id.startswith("safe") else "Hard"
        for method in METHODS:
            partial, tokens = 0.4, 200
            if method == "PlanExecute":
                partial, tokens = (0.9, 100) if category == "Safe" else (0.2, 100)
            elif method == "Solo":
                partial, tokens = (0.3, 180) if category == "Safe" else (1.0, 180)
            rows[method][task_id] = _row(task_id, category, partial, tokens)
    return task_ids, rows


def test_reward_penalizes_cost_and_errors() -> None:
    cheap = _row("a", "x", 0.5, 100)
    costly = _row("a", "x", 0.5, 1000)
    assert outcome_reward(cheap, 100) > outcome_reward(costly, 100)
    cheap["request_errors"] = 1
    assert outcome_reward(cheap, 100) < outcome_reward(costly, 100)


def test_policy_uses_observable_category_and_falls_back() -> None:
    task_ids, rows = _matrix()
    policy = fit_policy(task_ids, rows, alpha=0)
    assert policy["claim_class"] == "contextual_bandit_not_agentic_rl"
    assert select_method(policy, {"category": "Safe"}) == "PlanExecute"
    assert select_method(policy, {"category": "Hard"}) == "Solo"
    assert select_method(policy, {"category": "Unseen"}) in METHODS
    loo = leave_one_out_summary(task_ids, rows, alpha=0)
    assert loo["task_count"] == 4
    assert loo["passes"] == 2


def test_loader_rejects_nonpaired_runs(tmp_path: Path) -> None:
    directories = {}
    for method in METHODS:
        directory = tmp_path / method
        directory.mkdir()
        tasks = ["one", "two"] if method != "A8" else ["one"]
        config = {
            "method": method,
            "split": "dev",
            "tasks": tasks,
            "split_sha256": "split",
            "docker_image_id": "image",
            "max_tokens": 2048,
            "temperature": 0,
            "strong": {"model": "strong"},
            "weak": {"model": "weak"},
        }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with (directory / "results.jsonl").open("w", encoding="utf-8") as handle:
            for task_id in tasks:
                handle.write(json.dumps(_row(task_id, "x", 0.5, 100)) + "\n")
        directories[method] = directory
    with pytest.raises(ValueError, match="same task IDs"):
        load_paired_runs(directories)
