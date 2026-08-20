"""Audit-only real transition mutants for the G2 resource vertical slice."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from .stateful_ops import (
    MultiTownStatefulOpsEnv, ScheduledEvent, StatefulScenario,
)


RELAXED_TRIGGER_MUTATION_ID = "event_trigger_arguments_relaxed"
RELAXED_TRIGGER_OPERATOR_VERSION = "multitown-g2-relaxed-trigger-v1"
FAILED_CAS_MUTATION_ID = "failed_cas_partial_booking_bind"
FAILED_CAS_OPERATOR_VERSION = "multitown-g2-failed-cas-partial-bind-v1"
EXACT_REPLAY_MUTATION_ID = "exact_replay_reexecutes_versioned_hold"
EXACT_REPLAY_OPERATOR_VERSION = "multitown-g2-exact-replay-reexecute-v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _at(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


class RelaxedTriggerArgumentsEnv(MultiTownStatefulOpsEnv):
    """Single-fault mutant: ignore an otherwise exact trigger-argument map."""

    def __init__(self, scenario: StatefulScenario):
        self.mutation_activations: list[dict[str, Any]] = []
        super().__init__(scenario)

    def reset(self) -> dict[str, Any]:
        result = super().reset()
        self.mutation_activations.clear()
        return result

    def _scheduled_event_arguments_match(
        self, event: ScheduledEvent, arguments: Mapping[str, Any] | None,
    ) -> bool:
        expected = dict(event.trigger_arguments)
        if not expected or expected == arguments:
            return super()._scheduled_event_arguments_match(event, arguments)
        guard_matches = bool(
            not event.public_guard_path
            or _at(self.state, event.public_guard_path) == event.public_guard_value
        )
        # All other event predicates (identity, phase, due tick, trigger tool,
        # and not-yet-applied) have already passed before this hook.  The public
        # guard is the sole predicate evaluated afterward; only a true guard is
        # therefore an actual mutation activation/event application.
        if guard_matches:
            self.mutation_activations.append({
                "mutation_id": RELAXED_TRIGGER_MUTATION_ID,
                "operator_version": RELAXED_TRIGGER_OPERATOR_VERSION,
                "event_id": event.event_id,
                "phase": event.phase,
                "eligible_at_tick": event.logical_tick,
                "actual_tick": self.logical_tick,
                "trigger_tool": event.trigger_tool,
                "expected_arguments_sha256": _sha256(expected),
                "actual_arguments_sha256": _sha256(arguments),
                "exact_arguments_match": False,
                "public_guard_matches": True,
                "relaxed_predicate_taken": True,
                "event_already_applied": False,
                "event_target_path": "/".join(event.path),
            })
        return True

    def mutation_sidecar(self) -> dict[str, Any]:
        return {
            "schema_version": "multitown-g2-private-mutation-sidecar-v1",
            "mutation_id": RELAXED_TRIGGER_MUTATION_ID,
            "operator_version": RELAXED_TRIGGER_OPERATOR_VERSION,
            "private_instance_id": self.scenario.private_instance_id,
            "activation_count": len(self.mutation_activations),
            "activations": copy.deepcopy(self.mutation_activations),
        }


class CapturingRelaxedTriggerFactory:
    """Fresh-env factory that retains audit-only private mutation sidecars."""

    def __init__(self) -> None:
        self.instances: list[RelaxedTriggerArgumentsEnv] = []

    def __call__(self, scenario: StatefulScenario) -> RelaxedTriggerArgumentsEnv:
        env = RelaxedTriggerArgumentsEnv(scenario)
        self.instances.append(env)
        return env


class FailedCASPartialBindingEnv(MultiTownStatefulOpsEnv):
    """Single-fault mutant: a failed CAS partially binds its booking target."""

    def __init__(self, scenario: StatefulScenario):
        self.mutation_activations: list[dict[str, Any]] = []
        super().__init__(scenario)

    def reset(self) -> dict[str, Any]:
        result = super().reset()
        self.mutation_activations.clear()
        return result

    def _apply_version_conflict_effect(
        self, tool_name: str, arguments: Mapping[str, Any],
    ) -> None:
        if tool_name != "create_versioned_hold":
            raise AssertionError("failed-CAS mutant reached an unexpected tool")
        booking_id = str(arguments["booking_id"])
        resource_id = str(arguments["resource_id"])
        before = self.state["bookings"][booking_id]["resource_id"]
        self.state["bookings"][booking_id]["resource_id"] = resource_id
        state_infected = before != resource_id
        self.mutation_activations.append({
            "mutation_id": FAILED_CAS_MUTATION_ID,
            "operator_version": FAILED_CAS_OPERATOR_VERSION,
            "tool_name": tool_name,
            "booking_id_sha256": _sha256(booking_id),
            "resource_id_sha256": _sha256(resource_id),
            "before_value_sha256": _sha256(before),
            "after_value_sha256": _sha256(resource_id),
            "changed_path": f"bookings/{booking_id}/resource_id",
            "state_infected": state_infected,
        })

    def mutation_sidecar(self) -> dict[str, Any]:
        return {
            "schema_version": "multitown-g2-private-mutation-sidecar-v1",
            "mutation_id": FAILED_CAS_MUTATION_ID,
            "operator_version": FAILED_CAS_OPERATOR_VERSION,
            "private_instance_id": self.scenario.private_instance_id,
            "activation_count": len(self.mutation_activations),
            "infection_count": sum(
                row["state_infected"] for row in self.mutation_activations
            ),
            "activations": copy.deepcopy(self.mutation_activations),
        }


class CapturingFailedCASFactory:
    def __init__(self) -> None:
        self.instances: list[FailedCASPartialBindingEnv] = []

    def __call__(self, scenario: StatefulScenario) -> FailedCASPartialBindingEnv:
        env = FailedCASPartialBindingEnv(scenario)
        self.instances.append(env)
        return env


class ExactReplayReexecutesVersionedHoldEnv(MultiTownStatefulOpsEnv):
    """Single-fault mutant: exact successful replay dispatches the write again."""

    def __init__(self, scenario: StatefulScenario):
        self.mutation_activations: list[dict[str, Any]] = []
        super().__init__(scenario)

    def reset(self) -> dict[str, Any]:
        result = super().reset()
        self.mutation_activations.clear()
        return result

    def _apply_exact_replay_effect(
        self, tool_name: str, arguments: Mapping[str, Any], *,
        previous_result: str, idempotency_key: str,
        call_fingerprint: str,
    ) -> None:
        if tool_name != "create_versioned_hold" or previous_result != "ok":
            return
        booking_id = str(arguments["booking_id"])
        resource_id = str(arguments["resource_id"])
        resource = self.state["resources"][resource_id]
        before_version = resource["version"]
        # Reuse the real handler to model one erroneous duplicate dispatch.
        self._write(tool_name, dict(arguments))
        after_version = resource["version"]
        state_infected = before_version != after_version
        self.mutation_activations.append({
            "mutation_id": EXACT_REPLAY_MUTATION_ID,
            "operator_version": EXACT_REPLAY_OPERATOR_VERSION,
            "tool_name": tool_name,
            "booking_id_sha256": _sha256(booking_id),
            "resource_id_sha256": _sha256(resource_id),
            "idempotency_key_sha256": _sha256(idempotency_key),
            "call_fingerprint": call_fingerprint,
            "changed_path": f"resources/{resource_id}/version",
            "before_version": before_version,
            "after_version": after_version,
            "state_infected": state_infected,
        })

    def mutation_sidecar(self) -> dict[str, Any]:
        return {
            "schema_version": "multitown-g2-private-mutation-sidecar-v1",
            "mutation_id": EXACT_REPLAY_MUTATION_ID,
            "operator_version": EXACT_REPLAY_OPERATOR_VERSION,
            "private_instance_id": self.scenario.private_instance_id,
            "activation_count": len(self.mutation_activations),
            "infection_count": sum(
                row["state_infected"] for row in self.mutation_activations
            ),
            "activations": copy.deepcopy(self.mutation_activations),
        }


class CapturingExactReplayFactory:
    def __init__(self) -> None:
        self.instances: list[ExactReplayReexecutesVersionedHoldEnv] = []

    def __call__(
        self, scenario: StatefulScenario,
    ) -> ExactReplayReexecutesVersionedHoldEnv:
        env = ExactReplayReexecutesVersionedHoldEnv(scenario)
        self.instances.append(env)
        return env
