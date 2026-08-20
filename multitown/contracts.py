"""Versioned contracts shared by A7/A8 controllers and trajectory tooling.

The contracts intentionally use only the Python standard library.  They are
small enough to remain stable at the benchmark boundary while individual
controller or training implementations can change independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "multitown-trajectory-v1"


class MessageKind(str, Enum):
    TASK = "task"
    NEED = "need"
    OFFER = "offer"
    RESULT = "result"
    REVIEW = "review"
    CONTROL = "control"


class ControllerActionKind(str, Enum):
    ACTIVATE = "activate"
    DELEGATE = "delegate"
    CONNECT = "connect"
    CALL_TOOL = "call_tool"
    VERIFY = "verify"
    SUBMIT = "submit"
    STOP = "stop"
    ESCALATE_HUMAN = "escalate_human"


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class JsonContract:
    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class Budget(JsonContract):
    token_limit: int | None = None
    latency_limit_s: float | None = None
    communication_token_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("token_limit", "communication_token_limit"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.latency_limit_s is not None and self.latency_limit_s < 0:
            raise ValueError("latency_limit_s cannot be negative")


@dataclass(frozen=True, slots=True)
class TaskContract(JsonContract):
    task_id: str
    family: str
    instruction: str
    allowed_actions: tuple[str, ...]
    validator_id: str
    parent_task_id: str | None = None
    dependency_ids: tuple[str, ...] = ()
    created_step: int = 0
    deadline_step: int | None = None
    budget: Budget = field(default_factory=Budget)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.task_id, "task_id")
        _nonempty(self.family, "family")
        _nonempty(self.instruction, "instruction")
        _nonempty(self.validator_id, "validator_id")
        if not self.allowed_actions or any(not item for item in self.allowed_actions):
            raise ValueError("allowed_actions must contain non-empty actions")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed_actions must be unique")
        if self.parent_task_id == self.task_id or self.task_id in self.dependency_ids:
            raise ValueError("a task cannot be its own parent or dependency")
        if self.created_step < 0:
            raise ValueError("created_step cannot be negative")
        if self.deadline_step is not None and self.deadline_step < self.created_step:
            raise ValueError("deadline_step cannot precede created_step")


@dataclass(frozen=True, slots=True)
class StateFact(JsonContract):
    key: str
    value: Any
    source: str
    observed_by: str
    state_version: int
    updated_step: int
    stale_after_step: int | None = None

    def __post_init__(self) -> None:
        _nonempty(self.key, "key")
        _nonempty(self.source, "source")
        _nonempty(self.observed_by, "observed_by")
        if self.state_version < 0 or self.updated_step < 0:
            raise ValueError("state_version and updated_step cannot be negative")
        if self.stale_after_step is not None and self.stale_after_step < self.updated_step:
            raise ValueError("stale_after_step cannot precede updated_step")


@dataclass(frozen=True, slots=True)
class StateSnapshot(JsonContract):
    episode_id: str
    step_index: int
    state_version: int
    observer_id: str
    facts: tuple[StateFact, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        _nonempty(self.observer_id, "observer_id")
        if self.step_index < 0 or self.state_version < 0:
            raise ValueError("step_index and state_version cannot be negative")
        if any(fact.state_version > self.state_version for fact in self.facts):
            raise ValueError("a fact cannot be newer than its enclosing snapshot")


@dataclass(frozen=True, slots=True)
class CommunicationEdge(JsonContract):
    sender: str
    receiver: str
    reason: str

    def __post_init__(self) -> None:
        _nonempty(self.sender, "sender")
        _nonempty(self.receiver, "receiver")
        _nonempty(self.reason, "reason")
        if self.sender == self.receiver:
            raise ValueError("communication edges must connect different agents")


@dataclass(frozen=True, slots=True)
class ControllerAction(JsonContract):
    kind: ControllerActionKind
    controller_id: str
    task_id: str
    selected_action: str | None = None
    activated_agents: tuple[str, ...] = ()
    assigned_role: str | None = None
    model_tier: str | None = None
    communication_edges: tuple[CommunicationEdge, ...] = ()
    reason_codes: tuple[str, ...] = ()
    stop_reason: str | None = None
    propensity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.controller_id, "controller_id")
        _nonempty(self.task_id, "task_id")
        if len(set(self.activated_agents)) != len(self.activated_agents):
            raise ValueError("activated_agents must be unique")
        if self.propensity is not None and not 0 < self.propensity <= 1:
            raise ValueError("propensity must be in (0, 1]")
        if self.kind == ControllerActionKind.SUBMIT and not self.selected_action:
            raise ValueError("submit actions require selected_action")
        if self.kind == ControllerActionKind.CONNECT and not self.communication_edges:
            raise ValueError("connect actions require communication_edges")


@dataclass(frozen=True, slots=True)
class AgentMessage(JsonContract):
    message_id: str
    episode_id: str
    step_index: int
    sender: str
    receivers: tuple[str, ...]
    kind: MessageKind
    task_id: str
    state_version: int
    payload: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("message_id", "episode_id", "sender", "task_id"):
            _nonempty(getattr(self, name), name)
        if not self.receivers or any(not receiver for receiver in self.receivers):
            raise ValueError("receivers must contain non-empty agent ids")
        if self.sender in self.receivers:
            raise ValueError("a sender cannot also be a receiver")
        if self.step_index < 0 or self.state_version < 0:
            raise ValueError("step_index and state_version cannot be negative")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts cannot be negative")


@dataclass(frozen=True, slots=True)
class RewardComponents(JsonContract):
    final_success: float = 0.0
    subgoal_progress: float = 0.0
    invalid_action: float = 0.0
    budget_violation: float = 0.0
    tool_failure_recovery: float = 0.0
    unnecessary_delegation: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.final_success
            + self.subgoal_progress
            + self.invalid_action
            + self.budget_violation
            + self.tool_failure_recovery
            + self.unnecessary_delegation
        )

    def to_dict(self) -> dict[str, Any]:
        payload = JsonContract.to_dict(self)
        payload["total"] = self.total
        return payload


@dataclass(frozen=True, slots=True)
class TrajectoryStep(JsonContract):
    trajectory_id: str
    episode_id: str
    architecture: str
    step_index: int
    timestamp_utc: str
    task_id: str
    observation: StateSnapshot
    controller_action: ControllerAction
    messages: tuple[AgentMessage, ...]
    tool_result: dict[str, Any]
    reward: RewardComponents
    metrics: dict[str, Any]
    terminated: bool
    legacy_source: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "episode_id", "architecture", "timestamp_utc", "task_id"):
            _nonempty(getattr(self, name), name)
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        if self.episode_id != self.observation.episode_id:
            raise ValueError("step and observation episode ids must match")
        if self.step_index != self.observation.step_index:
            raise ValueError("step and observation indexes must match")
        if self.task_id != self.controller_action.task_id:
            raise ValueError("step and controller action task ids must match")
        if any(message.episode_id != self.episode_id for message in self.messages):
            raise ValueError("all messages must belong to the same episode")

    def to_dict(self) -> dict[str, Any]:
        payload = JsonContract.to_dict(self)
        # dataclasses.asdict recursively flattens nested dataclasses, so retain
        # the derived reward total explicitly at the storage boundary.
        payload["reward"] = self.reward.to_dict()
        return payload
