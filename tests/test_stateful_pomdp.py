import pytest

from multitown.stateful_model_protocol import ModelAction
from multitown.stateful_ops import build_scenario
from multitown.stateful_groups import variants_for_family
from multitown.stateful_ops import FAMILIES
from multitown.stateful_pomdp import (
    CPOMDP_SPEC_VERSION,
    GYM_ADAPTER_VERSION,
    StatefulPOMDPEnv,
    constrained_pomdp_spec,
)


def scenario(branch: str = "grant_before_deadline"):
    return build_scenario(
        "permit_transaction", world_seed=140,
        variant_id="asynchronous_authority_timeout",
        dynamics_branch=branch,
    )


def call(tool_name: str, **arguments: str) -> dict[str, object]:
    return {
        "action": "call_tool", "tool_name": tool_name,
        "arguments": arguments, "idempotency_key": None,
    }


def public_context(observation: dict[str, object]) -> dict[str, str]:
    world = observation["world"]
    assert isinstance(world, dict)
    permits = world["permits"]
    assert isinstance(permits, dict)
    permit_id = next(iter(permits))
    permit = permits[permit_id]
    assert isinstance(permit, dict)
    protected = world["protected"]
    assert isinstance(protected, dict)
    return {
        "permit_id": permit_id,
        "scope": str(permit["authority_scope"]),
        "policy_version": str(protected["policy_version"]),
        "request_id": str(permit["authority_request_id"] or "INVALID"),
        "expiry": str(permit["authority_expiry"]),
        "capability": str(permit["authority_capability"] or "INVALID"),
    }


def test_cpomdp_spec_defines_s_o_a_t_z_r_c_h_rho() -> None:
    spec = constrained_pomdp_spec(scenario()).to_dict()
    assert spec["schema_version"] == CPOMDP_SPEC_VERSION
    assert set(spec) == {
        "schema_version", "state", "observation", "action", "transition",
        "observation_kernel", "reward", "public_costs", "private_audit_costs",
        "horizon", "initial_world_distribution", "robust_objective",
    }
    assert "every member" in spec["robust_objective"]
    assert "collateral_mutation" in spec["private_audit_costs"]
    assert "point mass" in spec["initial_world_distribution"]
    assert "multi-instance runner" in spec["initial_world_distribution"]


def test_adapter_has_zero_intermediate_reward_and_public_cost_deltas() -> None:
    env = StatefulPOMDPEnv(scenario())
    observation, reset_info = env.reset()
    context = public_context(observation)
    observation, reward, terminated, truncated, info = env.step(call(
        "submit_authority_request", permit_id=context["permit_id"],
        scope=context["scope"], expected_policy_version=context["policy_version"],
    ))
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["schema_version"] == GYM_ADAPTER_VERSION
    assert info["terminal_result"] is None
    assert info["cost"]["tool_calls"] == 1
    assert info["cost"]["steps"] == 1
    assert all(value == 0 for value in reset_info["cost"].values())
    rendered = repr((observation, info, env.spec, env.action_contract))
    for private_name in (
        "private_instance_id", "private_evaluator", "private_state",
        "dynamics_branch", "world_seed", "scenario_group_id", "failure_codes",
    ):
        assert private_name not in rendered


def test_adapter_success_reward_appears_only_on_stop() -> None:
    env = StatefulPOMDPEnv(scenario())
    observation, _ = env.reset()
    context = public_context(observation)
    actions = [call(
        "submit_authority_request", permit_id=context["permit_id"],
        scope=context["scope"], expected_policy_version=context["policy_version"],
    )]
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)
        assert reward == 0.0 and not terminated and not truncated
    for _ in range(2):
        context = public_context(observation)
        observation, reward, terminated, truncated, _ = env.step(call(
            "check_authority_status", permit_id=context["permit_id"],
            request_id=context["request_id"],
        ))
        assert reward == 0.0 and not terminated and not truncated
    context = public_context(observation)
    for action in (
        call(
            "issue_permit_with_capability", permit_id=context["permit_id"],
            request_id=context["request_id"], scope=context["scope"],
            policy_version=context["policy_version"], expiry=context["expiry"],
            capability=context["capability"],
        ),
        call("send_permit_status_notice", permit_id=context["permit_id"]),
    ):
        observation, reward, terminated, truncated, _ = env.step(action)
        assert reward == 0.0 and not terminated and not truncated
    observation, reward, terminated, truncated, info = env.step({"action": "stop"})
    assert reward == 1.0 and terminated and not truncated
    assert observation["terminal"]
    assert info["terminal_result"] == {
        "terminal": True, "success": True,
        "safety_violations": 0, "budget_violations": 0,
    }
    assert "failure_codes" not in repr(info)
    with pytest.raises(RuntimeError):
        env.step({"action": "stop"})


def test_adapter_budget_rejection_is_truncation_not_active_abstention() -> None:
    env = StatefulPOMDPEnv(scenario())
    observation, _ = env.reset()
    context = public_context(observation)
    action = call(
        "check_authority_status", permit_id=context["permit_id"],
        request_id="INVALID",
    )
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            break
    assert reward == 0.0
    assert not terminated and truncated
    assert info["termination_reason"] == "budget_exhausted"
    assert info["cost"]["budget_violations"] == 1
    assert observation["terminal"]


def test_adapter_validates_action_before_mutating_episode() -> None:
    env = StatefulPOMDPEnv(scenario())
    before, _ = env.reset()
    with pytest.raises(ValueError):
        env.step({"action": "call_tool", "tool_name": "get_permit"})
    context = public_context(before)
    attacked = env.step(call("get_permit", permit_id=context["permit_id"]))
    control = StatefulPOMDPEnv(scenario())
    control.reset()
    expected = control.step(call("get_permit", permit_id=context["permit_id"]))
    assert attacked == expected
    with pytest.raises(ValueError):
        env.reset(seed=1)
    control.step(ModelAction("stop", None, {}, None))


def test_adapter_public_contract_resets_all_sixteen_structural_groups() -> None:
    rows = []
    for family in FAMILIES:
        for variant in variants_for_family(family):
            env = StatefulPOMDPEnv(build_scenario(
                family, world_seed=141, variant_id=variant,
            ))
            observation, info = env.reset()
            assert observation["task"]["family"] == family
            assert set(env.action_contract["tools"])
            assert all(value == 0 for value in info["cost"].values())
            _, reward, terminated, truncated, terminal_info = env.step({
                "action": "stop",
            })
            assert reward == 0.0 and terminated and not truncated
            assert terminal_info["terminal_result"]["success"] is False
            rows.append((family, variant))
    assert len(rows) == 16
