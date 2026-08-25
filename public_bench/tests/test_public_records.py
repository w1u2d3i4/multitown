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


def test_mainstream_strategy_summary_preserves_negative_result() -> None:
    summary = json.loads(
        (RECORDS / "teambench-strategy-quality-v2-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["paired_tasks"] == 89
    assert summary["comparison_scope"] == "quality_and_tokens_only"
    assert summary["runtime_compatible_across_all_five"] is False
    assert summary["quality_token_pareto_frontier"] == [
        "Solo-TB",
        "PlanExecute-TB",
    ]
    methods = summary["methods"]
    assert methods["PlanExecute-TB"]["passes"] == 18
    assert methods["MultiTown-TB"]["passes"] == 11
    assert methods["ExecuteReview-TB"]["mean_partial_score"] < methods[
        "PlanExecute-TB"
    ]["mean_partial_score"]
    assert summary["invocation_errors"] == 0


def test_seed2_three_way_record_preserves_negative_controller_result() -> None:
    summary = json.loads(
        (RECORDS / "teambench-sequential-seed2-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "three_way_complete_negative_controller"
    assert summary["paired_tasks"] == 89
    assert summary["baseline"]["passes"] == 17
    assert summary["candidate"]["passes"] == 16
    assert summary["solo"]["passes"] == 13
    assert summary["quality_token_pareto_frontier"] == [
        "PlanExecute",
        "MTSequential",
    ]
    assert summary["benchmark_best_supported"] is False
    assert summary["integrity"]["unique_tasks_each"] == 89
    assert summary["integrity"]["invocation_errors_each"] == 0
    assert summary["integrity"]["grader_timeout_rows_each"] == 0


def test_replan_dev_record_does_not_overstate_advancement() -> None:
    summary = json.loads(
        (RECORDS / "teambench-replan-dev-v2.1-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["paired_tasks"] == 30
    assert summary["baseline"]["passes"] == 6
    assert summary["candidate"]["passes"] == 4
    assert summary["same_trajectory_shadow_effect"]["pass_delta"] == 0
    assert summary["same_trajectory_shadow_effect"]["positive_repair"][
        "partial_delta"
    ] == pytest.approx(0.3334)
    assert summary["decision"] == {
        "advance_to_seed3": False,
        "benchmark_best_supported": False,
        "agentic_rl_supported": False,
    }


def test_agentic_rl_dev_record_preserves_learned_noop_result() -> None:
    summary = json.loads(
        (RECORDS / "teambench-agentic-rl-dev-v1-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["training_exploration"]["tasks"] == 30
    assert summary["training_exploration"]["complete_counterfactual_episodes"] == 28
    assert summary["training"]["algorithm"].startswith("finite-horizon fitted Q")
    online = summary["online_development_evaluation"]
    assert online["paired_tasks"] == 30
    assert online["candidate"]["passes"] == 6
    assert online["same_trajectory_plan_execute"]["passes"] == 6
    assert online["candidate_minus_plan_execute"]["partial_score"]["mean"] == 0
    assert online["candidate_minus_plan_execute"]["tokens"]["mean"] == 0
    assert online["selected_hash_mismatches"] == 0
    assert summary["decision"] == {
        "trained_agentic_rl_supported": True,
        "performance_advantage_supported": False,
        "benchmark_best_supported": False,
        "advance_to_seed3": False,
    }


def test_public_records_do_not_expose_machine_local_paths() -> None:
    for path in RECORDS.iterdir():
        if path.suffix not in {".json", ".md", ".csv"}:
            continue
        content = path.read_text(encoding="utf-8")
        assert "/home/dilab/" not in content
        assert "USTC@" not in content
