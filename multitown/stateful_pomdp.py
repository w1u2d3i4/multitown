"""Public-policy interface and formal contract for stateful MultiTown CPOMDPs.

The adapter intentionally depends on no Gym package.  Its ``reset`` and ``step``
signatures follow the Gymnasium API, while every value returned to a policy is
already part of the audited :class:`PolicySession` surface.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from .stateful_model_protocol import ModelAction, parse_model_action
from .stateful_ops import (
    MultiTownStatefulOpsEnv, PolicySession, StatefulScenario, tool_profile,
)


CPOMDP_SPEC_VERSION = "multitown-stateful-cpomdp-spec-v1"
GYM_ADAPTER_VERSION = "multitown-stateful-gym-adapter-v1"


@dataclass(frozen=True)
class ConstrainedPOMDPSpec:
    """Finite-horizon constrained-POMDP contract, independent of training."""

    schema_version: str
    state: str
    observation: str
    action: str
    transition: str
    observation_kernel: str
    reward: str
    public_costs: tuple[str, ...]
    private_audit_costs: tuple[str, ...]
    horizon: dict[str, int]
    initial_world_distribution: str
    robust_objective: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def constrained_pomdp_spec(scenario: StatefulScenario) -> ConstrainedPOMDPSpec:
    """Describe S/O/A/T/Z/R/C/H/rho without exposing a private instance."""

    budget = scenario.public_task.budget
    return ConstrainedPOMDPSpec(
        schema_version=CPOMDP_SPEC_VERSION,
        state=(
            "S=(business state, private dynamics, pending exogenous events, "
            "public runtime, remaining budgets)"
        ),
        observation=(
            "O is exactly the JSON value returned by PolicySession.observation"
        ),
        action=(
            "A is one exact typed ModelAction: call_tool(tool_name, arguments, "
            "idempotency_key) or stop"
        ),
        transition=(
            "T is the deterministic tool transition composed with guarded, "
            "deterministic exogenous events for one frozen private world"
        ),
        observation_kernel="Z is the audited public projection of post-action state",
        reward=(
            "R=1 only for strict terminal success and 0 otherwise; every "
            "non-terminal transition has reward 0"
        ),
        public_costs=(
            "tool_calls", "steps", "logical_latency", "irreversible_risk",
            "attempted_policy_violations", "executed_safety_violations",
            "budget_violations",
        ),
        private_audit_costs=("collateral_mutation", "forbidden_terminal_outcome"),
        horizon={
            "max_steps": budget.max_steps,
            "tool_calls": budget.tool_calls,
            "logical_latency": budget.logical_latency,
            "irreversible_risk": budget.irreversible_risk,
        },
        initial_world_distribution=(
            "rho is a point mass on this adapter's trusted frozen private world; "
            "a separate trusted multi-instance runner supplies compatible worlds "
            "for robust synthesis and never exposes their identities to the policy"
        ),
        robust_objective=(
            "primary: strict success in every member of a structural pair/triplet; "
            "expectation under rho is secondary"
        ),
    )


def _parse_action(action: ModelAction | Mapping[str, Any], family: str) -> ModelAction:
    if isinstance(action, ModelAction):
        raw = {
            "action": action.action,
            **({} if action.action == "stop" else {
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "idempotency_key": action.idempotency_key,
            }),
        }
    elif isinstance(action, Mapping):
        raw = dict(action)
    else:
        raise TypeError("action must be a ModelAction or mapping")
    try:
        content = json.dumps(
            raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be finite JSON") from exc
    return parse_model_action(content, family=family)


def _cost_delta(
    before: dict[str, Any], after: dict[str, Any],
) -> dict[str, int]:
    left, right = before["runtime"], after["runtime"]
    return {
        "tool_calls": (
            before["tool_calls_remaining"] - after["tool_calls_remaining"]
        ),
        "steps": after["steps"] - before["steps"],
        "logical_latency": (
            right["logical_latency_used"] - left["logical_latency_used"]
        ),
        "irreversible_risk": (
            right["irreversible_risk_used"] - left["irreversible_risk_used"]
        ),
        "attempted_policy_violations": (
            right["attempted_policy_violations"]
            - left["attempted_policy_violations"]
        ),
        "executed_safety_violations": (
            right["executed_safety_violations"]
            - left["executed_safety_violations"]
        ),
        "budget_violations": (
            right["budget_violations"] - left["budget_violations"]
        ),
    }


class StatefulPOMDPEnv:
    """Gymnasium-shaped API contract with public observations and sparse reward.

    This in-process Python object is trusted runner infrastructure, not a
    security sandbox.  Untrusted policies must receive serialized return values
    only and must not receive or introspect the adapter object itself.
    """

    __slots__ = (
        "__scenario", "__session", "__done", "__family", "__env_factory",
    )

    metadata = {"render_modes": []}

    def __init__(
        self, scenario: StatefulScenario, *,
        env_factory: Callable[[StatefulScenario], MultiTownStatefulOpsEnv]
        = MultiTownStatefulOpsEnv,
    ):
        self.__scenario = scenario
        self.__family = scenario.public_task.family
        self.__env_factory = env_factory
        self.__session = PolicySession(scenario, env_factory=env_factory)
        self.__done = False

    @property
    def spec(self) -> dict[str, Any]:
        return constrained_pomdp_spec(self.__scenario).to_dict()

    @property
    def action_contract(self) -> dict[str, Any]:
        return copy.deepcopy(tool_profile(self.__family))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            raise ValueError("scenario identity is frozen; construct a seeded scenario")
        if options not in (None, {}):
            raise ValueError("reset options are unsupported")
        self.__session = PolicySession(
            self.__scenario, env_factory=self.__env_factory,
        )
        self.__done = False
        observation = self.__session.observation()
        return observation, {
            "schema_version": GYM_ADAPTER_VERSION,
            "cost": {name: 0 for name in self.spec["public_costs"]},
        }

    def step(
        self, action: ModelAction | Mapping[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.__done:
            raise RuntimeError("episode is done; call reset before another step")
        parsed = _parse_action(action, self.__family)
        before = self.__session.observation()
        tool_result: dict[str, Any] | None = None
        terminal_result: dict[str, Any] | None = None
        terminated = False
        truncated = False
        reason: str | None = None
        if parsed.action == "stop":
            terminal_result = self.__session.stop()
            terminated = True
            reason = "agent_stop"
        else:
            assert parsed.tool_name is not None
            tool_result = self.__session.call_tool(
                parsed.tool_name, parsed.arguments,
                idempotency_key=parsed.idempotency_key,
            )
            if tool_result.get("error_code") == "BUDGET_EXHAUSTED":
                terminal_result = self.__session.stop()
                truncated = True
                reason = "budget_exhausted"
        after = self.__session.observation()
        self.__done = terminated or truncated
        reward = float(bool(terminal_result and terminal_result["success"]))
        info: dict[str, Any] = {
            "schema_version": GYM_ADAPTER_VERSION,
            "cost": _cost_delta(before, after),
            "tool_result": copy.deepcopy(tool_result),
            "terminal_result": copy.deepcopy(terminal_result),
            "termination_reason": reason,
        }
        return after, reward, terminated, truncated, info

    def close(self) -> None:
        """Provided for Gymnasium compatibility; there are no external resources."""
