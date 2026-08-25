from __future__ import annotations

from general_mas_bench.agentic_rl import (
    DATASET_SCHEMA,
    outcome_reward,
    select_action,
    train_policy,
    validate_policy,
)
from general_mas_bench.agentic_rl_v2 import DATASET_SCHEMA as DATASET_SCHEMA_V2
from general_mas_bench.agentic_rl_v2 import select_action as select_action_v2
from general_mas_bench.agentic_rl_v2 import train_policy as train_policy_v2
from general_mas_bench.agentic_rl_v2 import validate_policy as validate_policy_v2


def _outcome(partial: float, tokens: int, latency: float, passed: bool = False):
    return {
        "passed": passed,
        "partial_score": partial,
        "total_tokens": tokens,
        "latency_s": latency,
        "invocation_error": False,
        "safety_violation": False,
    }


def _state(failed: int, tokens: int, latency: float):
    return {
        "category": "Testing",
        "difficulty": "hard",
        "workspace_changed": True,
        "successful_commands": 0,
        "failed_commands": failed,
        "timed_out_commands": 0,
        "max_failed_command_repetitions": failed,
        "turns_used": 8,
        "turn_budget_exhausted": False,
        "reliability_score": 0.4,
        "hard_fail": failed > 0,
        "plan_delivered": True,
        "plan_retry_used": False,
        "recovery_plan_delivered": True,
        "consumed_tokens": tokens,
        "remaining_token_budget": 90_000 - tokens,
        "consumed_latency_s": latency,
    }


def _dataset():
    episodes = []
    for index in range(6):
        failed = index % 3
        prefix = _outcome(0.4 + 0.05 * index, 30_000 + index * 100, 30 + index)
        recovery = _outcome(
            prefix["partial_score"] + (0.3 if failed >= 2 else -0.1),
            45_000 + index * 100,
            45 + index,
        )
        reviewed = recovery if failed >= 2 else prefix | {
            "total_tokens": 50_000 + index * 100,
            "latency_s": 50 + index,
        }
        episodes.append({
            "task_id": f"task-{index}",
            "states": {
                "post_execution": _state(failed, prefix["total_tokens"], prefix["latency_s"]),
                "post_replan": _state(failed, 35_000 + index * 100, 35 + index),
                "post_recovery": _state(failed, recovery["total_tokens"], recovery["latency_s"]),
            },
            "outcomes": {
                "prefix": prefix,
                "recovery": recovery,
                "reviewed": reviewed,
            },
        })
    return {"schema_version": DATASET_SCHEMA, "episodes": episodes}


def test_reward_penalizes_tokens_latency_and_safety() -> None:
    config = {
        "pass_bonus": 0.25,
        "token_penalty": 0.05,
        "latency_penalty": 0.02,
        "invocation_error_penalty": 0.25,
        "safety_penalty": 0.5,
        "human_penalty": 0.1,
    }
    cheap = _outcome(0.8, 20_000, 20)
    costly = _outcome(0.8, 80_000, 80)
    unsafe = cheap | {"safety_violation": True}
    assert outcome_reward(
        cheap, reference_tokens=40_000, reference_latency_s=40, reward_config=config
    ) > outcome_reward(
        costly, reference_tokens=40_000, reference_latency_s=40, reward_config=config
    )
    assert outcome_reward(
        cheap, reference_tokens=40_000, reference_latency_s=40, reward_config=config
    ) > outcome_reward(
        unsafe, reference_tokens=40_000, reference_latency_s=40, reward_config=config
    )


def test_train_policy_produces_hashed_sequential_policy() -> None:
    policy = train_policy(_dataset())
    assert policy["claim_class"] == "trained_offline_sequential_fitted_q_candidate"
    assert len(policy["policy_sha256"]) == 64
    assert set(policy["phase_actions"]) == {
        "post_execution", "post_replan", "post_recovery"
    }
    decision = select_action(
        policy, "post_execution", _state(2, 30_000, 30)
    )
    assert decision["action"] in {"stop", "escalate", "human/abstain"}
    assert decision["policy_sha256"] == policy["policy_sha256"]
    validate_policy(policy)
    policy["conservative_margin"] = 99.0
    try:
        validate_policy(policy)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered policy was accepted")


def _v2_state(failed: int, tokens: int, latency: float, *, recovery: bool = False):
    execution = {
        "workspace_changed": True,
        "successful_commands": 1,
        "failed_commands": failed,
        "timed_out_commands": 0,
        "repeated_commands": 0,
        "max_failed_command_repetitions": failed,
        "turns_used": 8,
        "turn_budget_exhausted": True,
        "reliability_score": 0.5,
        "hard_fail": False,
        "test_commands": 1,
        "successful_test_commands": 0,
        "failed_test_commands": 1,
        "last_test_exit_code": 1,
    }
    current = execution | ({
        "successful_commands": 2,
        "failed_commands": 0,
        "reliability_score": 0.9,
        "successful_test_commands": 1,
        "failed_test_commands": 0,
        "last_test_exit_code": 0,
    } if recovery else {})
    return {
        "category": "Testing",
        "difficulty": "hard",
        "validator": current,
        "execution_validator": execution,
        "plan_delivered": True,
        "plan_retry_used": False,
        "recovery_plan_delivered": True,
        "consumed_tokens": tokens,
        "remaining_token_budget": 90_000 - tokens,
        "consumed_latency_s": latency,
        "brief_word_count": 10,
        **{f"brief_hash_{index:02d}": float(index == 0) for index in range(16)},
        "workspace_file_count": 5,
        "workspace_test_file_count": 1,
        "workspace_language_count": 1,
        "workspace_has_python": 1,
    }


def _v2_dataset():
    episodes = []
    for seed in range(3):
        for index in range(3):
            beneficial = index == 2
            prefix = _outcome(0.4, 30_000, 30)
            recovery = _outcome(0.9 if beneficial else 0.4, 42_000, 42)
            episodes.append({
                "instance_id": f"task-{index}::seed={seed}",
                "task_id": f"task-{index}",
                "seed": seed,
                "states": {
                    "post_execution": _v2_state(index, 30_000, 30),
                    "post_replan": _v2_state(index, 34_000, 34),
                    "post_recovery": _v2_state(index, 42_000, 42, recovery=True),
                },
                "outcomes": {"prefix": prefix, "recovery": recovery},
            })
    return {"schema_version": DATASET_SCHEMA_V2, "episodes": episodes}


def test_v2_trains_pessimistic_hashed_ensemble() -> None:
    policy = train_policy_v2(_v2_dataset())
    validate_policy_v2(policy)
    assert policy["claim_class"] == "trained_offline_sequential_pessimistic_q_candidate"
    assert policy["training"]["seeds"] == [0, 1, 2]
    assert policy["post_recovery_review_mode"] == "deterministic_rollback_to_prefix"
    decision = select_action_v2(
        policy, "post_execution", _v2_state(2, 30_000, 30)
    )
    assert decision["action"] in {"stop", "escalate"}
    assert set(decision["advantage"]) == {"mean", "std", "lcb", "min", "max"}
    assert policy["uncertainty_beta"] > 0


def test_v2_rejects_non_pessimistic_or_malformed_policy() -> None:
    policy = train_policy_v2(_v2_dataset())
    for key, value, expected in (
        ("uncertainty_beta", 0.0, "positive uncertainty"),
        ("budget_reserve_tokens", {"post_execution": -1}, "reserve phases"),
        ("ensemble_size", 1, "at least two"),
    ):
        broken = dict(policy)
        broken[key] = value
        broken.pop("policy_sha256")
        from general_mas_bench.agentic_rl_v2 import _canonical_sha256

        broken["policy_sha256"] = _canonical_sha256(broken)
        try:
            validate_policy_v2(broken)
        except (TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"malformed {key} was accepted")
