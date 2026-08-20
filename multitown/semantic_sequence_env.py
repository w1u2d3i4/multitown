"""A14 train-only multi-step environment over frozen A13 real-model traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .masbench_routing import git_state, utc_now, write_json
from .semantic_model_probe import SemanticProbeOutcome, read_outcomes, validate_coverage
from .semantic_tasks import FAMILIES, SemanticTask, read_bank


ENV_VERSION = "multitown-semantic-sequence-env-v2"
BANK_VERSION = "multitown-semantic-sequence-train-bank-v2"
DEFAULT_SEED = 20260814


class SequenceAction(IntEnum):
    STOP = 0
    DELEGATE = 1
    ESCALATE = 2
    REVIEW = 3
    HUMAN = 4


ACTION_NAMES = tuple(action.name.lower() for action in SequenceAction)
ACTION_COUNT = len(SequenceAction)


@dataclass(frozen=True)
class ReplayCall:
    model_key: str
    candidate: int | None
    valid: bool
    abstained: bool
    tokens: int
    latency_s: float

    def __post_init__(self) -> None:
        if self.model_key not in {"qwen4b", "qwen35b"}:
            raise ValueError(f"unsupported replay model: {self.model_key}")
        if self.candidate is not None and not 0 <= self.candidate < 4:
            raise ValueError("replay candidate must be in [0, 3]")
        if self.tokens <= 0 or self.latency_s < 0:
            raise ValueError("replay costs must be positive tokens and non-negative latency")
        if self.abstained and self.candidate is not None:
            raise ValueError("an abstained replay call cannot contain a candidate")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReplayCall":
        return cls(
            model_key=str(value["model_key"]),
            candidate=(
                None if value.get("candidate") is None else int(value["candidate"])
            ),
            valid=bool(value["valid"]), abstained=bool(value["abstained"]),
            tokens=int(value["tokens"]), latency_s=float(value["latency_s"]),
        )


@dataclass(frozen=True)
class ReplayTask:
    task_id: str
    family: str
    authority: str
    correct_option: int
    qwen4b: ReplayCall
    qwen35b: ReplayCall

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported replay family: {self.family}")
        if self.authority not in {"local", "central"}:
            raise ValueError(f"unsupported replay authority: {self.authority}")
        if not 0 <= self.correct_option < 4:
            raise ValueError("replay correct option must be in [0, 3]")
        if self.qwen4b.model_key != "qwen4b" or self.qwen35b.model_key != "qwen35b":
            raise ValueError("replay worker slots do not match model keys")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReplayTask":
        return cls(
            task_id=str(value["task_id"]), family=str(value["family"]),
            authority=str(value["authority"]),
            correct_option=int(value["correct_option"]),
            qwen4b=ReplayCall.from_dict(value["qwen4b"]),
            qwen35b=ReplayCall.from_dict(value["qwen35b"]),
        )


@dataclass(frozen=True)
class SemanticSequenceEpisode:
    episode_id: str
    composition_id: str
    split: str
    seed: int
    token_budget: int
    latency_budget_s: float
    model_call_budget: int
    review_budget: int
    human_budget: int
    qwen4b_token_reservation: int
    qwen35b_token_reservation: int
    qwen4b_latency_reservation_s: float
    qwen35b_latency_reservation_s: float
    max_steps: int
    tasks: tuple[ReplayTask, ...]
    schema_version: str = ENV_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENV_VERSION or self.split != "train":
            raise ValueError("A14 v2 supports train-only sequence episodes")
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("sequence episode tasks must be non-empty and unique")
        if min(
            self.token_budget, self.model_call_budget, self.review_budget,
            self.human_budget, self.qwen4b_token_reservation,
            self.qwen35b_token_reservation, self.max_steps,
        ) <= 0 or self.latency_budget_s <= 0:
            raise ValueError("sequence episode budgets and horizon must be positive")
        if min(
            self.qwen4b_latency_reservation_s, self.qwen35b_latency_reservation_s,
        ) <= 0:
            raise ValueError("sequence latency reservations must be positive")
        for task in self.tasks:
            if (
                task.qwen4b.tokens > self.qwen4b_token_reservation
                or task.qwen35b.tokens > self.qwen35b_token_reservation
                or task.qwen4b.latency_s > self.qwen4b_latency_reservation_s
                or task.qwen35b.latency_s > self.qwen35b_latency_reservation_s
            ):
                raise ValueError("replay call exceeds the episode's public cost reservation")
        if self.composition_id != composition_content_id(self.tasks):
            raise ValueError("sequence composition id does not match tasks")
        expected_id = episode_content_id(
            split=self.split, seed=self.seed, token_budget=self.token_budget,
            latency_budget_s=self.latency_budget_s,
            model_call_budget=self.model_call_budget,
            review_budget=self.review_budget, human_budget=self.human_budget,
            qwen4b_token_reservation=self.qwen4b_token_reservation,
            qwen35b_token_reservation=self.qwen35b_token_reservation,
            qwen4b_latency_reservation_s=self.qwen4b_latency_reservation_s,
            qwen35b_latency_reservation_s=self.qwen35b_latency_reservation_s,
            max_steps=self.max_steps, tasks=self.tasks,
        )
        if self.episode_id != expected_id:
            raise ValueError("sequence episode id does not match content")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticSequenceEpisode":
        if value.get("schema_version") != ENV_VERSION:
            raise ValueError(f"unsupported sequence environment: {value.get('schema_version')}")
        if value.get("split") != "train":
            raise ValueError("A14 v2 supports train-only sequence episodes")
        return cls(
            episode_id=str(value["episode_id"]),
            composition_id=str(value["composition_id"]), split=str(value["split"]),
            seed=int(value["seed"]), token_budget=int(value["token_budget"]),
            latency_budget_s=float(value["latency_budget_s"]),
            model_call_budget=int(value["model_call_budget"]),
            review_budget=int(value["review_budget"]),
            human_budget=int(value["human_budget"]),
            qwen4b_token_reservation=int(value["qwen4b_token_reservation"]),
            qwen35b_token_reservation=int(value["qwen35b_token_reservation"]),
            qwen4b_latency_reservation_s=float(value["qwen4b_latency_reservation_s"]),
            qwen35b_latency_reservation_s=float(value["qwen35b_latency_reservation_s"]),
            max_steps=int(value["max_steps"]),
            tasks=tuple(ReplayTask.from_dict(task) for task in value["tasks"]),
        )


@dataclass(frozen=True)
class SequenceReward:
    accuracy: float = 0.0
    token_penalty: float = 0.0
    latency_penalty: float = 0.0
    review_penalty: float = 0.0
    human_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    safety_penalty: float = 0.0

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


@dataclass(frozen=True)
class SequencePolicyState:
    """Immutable policy input containing only public or already observed state."""

    family: str
    authority: str
    task_index: int
    tasks: int
    current_candidate: int | None
    candidate_source: str | None
    qwen4b_called: bool
    qwen35b_called: bool
    review_used_on_task: bool
    review_state: int
    last_call_abstained: bool
    tokens_remaining: int
    latency_remaining_s: float
    model_calls_remaining: int
    reviews_remaining: int
    humans_remaining: int
    action_mask: tuple[bool, ...]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def composition_content_id(tasks: tuple[ReplayTask, ...]) -> str:
    """Full content hash excluding the sampling seed."""

    return _canonical_sha256({
        "schema_version": ENV_VERSION,
        "ordered_tasks": [asdict(task) for task in tasks],
    })


def episode_content_id(
    *, split: str, seed: int, token_budget: int, latency_budget_s: float,
    model_call_budget: int, review_budget: int, human_budget: int, max_steps: int,
    qwen4b_token_reservation: int, qwen35b_token_reservation: int,
    qwen4b_latency_reservation_s: float, qwen35b_latency_reservation_s: float,
    tasks: tuple[ReplayTask, ...],
) -> str:
    value = {
        "schema_version": ENV_VERSION, "split": split, "seed": seed,
        "token_budget": token_budget, "latency_budget_s": latency_budget_s,
        "model_call_budget": model_call_budget, "review_budget": review_budget,
        "human_budget": human_budget,
        "qwen4b_token_reservation": qwen4b_token_reservation,
        "qwen35b_token_reservation": qwen35b_token_reservation,
        "qwen4b_latency_reservation_s": qwen4b_latency_reservation_s,
        "qwen35b_latency_reservation_s": qwen35b_latency_reservation_s,
        "max_steps": max_steps,
        "tasks": [asdict(task) for task in tasks],
    }
    return f"a14-{split}-{_canonical_sha256(value)[:16]}"


def build_replay_tasks(
    tasks: list[SemanticTask], outcomes: list[SemanticProbeOutcome],
) -> list[ReplayTask]:
    task_index = {task.task_id: task for task in tasks}
    outcome_index = {row.task_id: row for row in outcomes}
    if len(task_index) != len(tasks) or len(outcome_index) != len(outcomes):
        raise ValueError("duplicate semantic task or outcome id")
    if not set(outcome_index) <= set(task_index):
        raise ValueError("semantic outcomes contain tasks outside the bank")
    validate_coverage(
        [task_index[task_id] for task_id in sorted(outcome_index)], outcomes,
    )

    replay = []
    for task_id in sorted(outcome_index):
        task = task_index[task_id]
        row = outcome_index[task_id]
        live_role = "weak" if task.world_state["authority"] == "local" else "strong"

        def convert(model_key: str) -> ReplayCall:
            call = row.calls[f"{model_key}_{live_role}_context"]
            return ReplayCall(
                model_key=model_key, candidate=call.parsed_option,
                valid=call.valid, abstained=call.abstained,
                tokens=call.total_tokens, latency_s=call.latency_s,
            )

        replay.append(ReplayTask(
            task_id=task_id, family=task.family,
            authority=str(task.world_state["authority"]),
            correct_option=task.correct_option,
            qwen4b=convert("qwen4b"), qwen35b=convert("qwen35b"),
        ))
    return replay


def generate_episode(
    replay_tasks: list[ReplayTask], seed: int,
) -> SemanticSequenceEpisode:
    by_family = {
        family: [task for task in replay_tasks if task.family == family]
        for family in FAMILIES
    }
    if any(not values for values in by_family.values()):
        raise ValueError("replay pool must cover every semantic family")
    rng = random.Random(f"{ENV_VERSION}:train:{seed}")
    selected = [rng.choice(by_family[family]) for family in FAMILIES]
    rng.shuffle(selected)
    q4_token_reservation = max(task.qwen4b.tokens for task in replay_tasks)
    q35_token_reservation = max(task.qwen35b.tokens for task in replay_tasks)
    q4_latency_reservation = max(task.qwen4b.latency_s for task in replay_tasks)
    q35_latency_reservation = max(task.qwen35b.latency_s for task in replay_tasks)
    token_budget = 4 * q4_token_reservation + max(
        q4_token_reservation, q35_token_reservation,
    )
    latency_budget = 4 * q4_latency_reservation + max(
        q4_latency_reservation, q35_latency_reservation,
    )
    selected_tuple = tuple(selected)
    composition_id = composition_content_id(selected_tuple)
    episode_id = episode_content_id(
        split="train", seed=seed, token_budget=token_budget,
        latency_budget_s=latency_budget, model_call_budget=5,
        review_budget=2, human_budget=1, max_steps=20, tasks=selected_tuple,
        qwen4b_token_reservation=q4_token_reservation,
        qwen35b_token_reservation=q35_token_reservation,
        qwen4b_latency_reservation_s=q4_latency_reservation,
        qwen35b_latency_reservation_s=q35_latency_reservation,
    )
    return SemanticSequenceEpisode(
        episode_id=episode_id, composition_id=composition_id, split="train", seed=seed,
        token_budget=token_budget, latency_budget_s=latency_budget,
        model_call_budget=5, review_budget=2, human_budget=1,
        qwen4b_token_reservation=q4_token_reservation,
        qwen35b_token_reservation=q35_token_reservation,
        qwen4b_latency_reservation_s=q4_latency_reservation,
        qwen35b_latency_reservation_s=q35_latency_reservation,
        max_steps=20, tasks=selected_tuple,
    )


def episode_bank(
    replay_tasks: list[ReplayTask], count: int, *, seed: int = DEFAULT_SEED,
) -> list[SemanticSequenceEpisode]:
    if count <= 0:
        raise ValueError("episode count must be positive")
    episodes = [generate_episode(replay_tasks, seed + index) for index in range(count)]
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("sequence episode ids are not unique")
    return episodes


def read_episode_bank(path: Path) -> list[SemanticSequenceEpisode]:
    episodes = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episodes.append(SemanticSequenceEpisode.from_dict(json.loads(line)))
    if not episodes:
        raise ValueError(f"no sequence episodes in {path}")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("duplicate sequence episode ids")
    return episodes


class MultiTownSemanticSequenceEnv:
    """Gym-like trace replay with deterministic live-source masking."""

    observation_size = 32

    def __init__(self, episode: SemanticSequenceEpisode):
        self.episode = episode
        self.reset()

    @property
    def task(self) -> ReplayTask | None:
        if self.task_index >= len(self.episode.tasks):
            return None
        return self.episode.tasks[self.task_index]

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.step_index = 0
        self.task_index = 0
        self.tokens_used = 0
        self.latency_used_s = 0.0
        self.model_calls = 0
        self.reviews_used = 0
        self.humans_used = 0
        self.autonomous_correct = 0
        self.assisted = 0
        self.failed = 0
        self.invalid_actions = 0
        self.budget_violations = 0
        self.horizon_exhaustions = 0
        self.unauthorized_source_calls = 0
        self.safety_violations = 0
        self.post_step_budget_invariant_violations = 0
        self.terminated = False
        self.total_reward = 0.0
        self.trajectory: list[dict[str, Any]] = []
        self._reset_task_state()
        return self.observation(), self.info()

    def _reset_task_state(self) -> None:
        self.current_candidate: int | None = None
        self.candidate_source: str | None = None
        self.qwen4b_called = False
        self.qwen35b_called = False
        self.review_used_on_task = False
        self.review_state = 0  # unknown=0, pass=1, fail=2
        self.last_call_abstained = False

    def _fits(self, model_key: str) -> bool:
        if model_key == "qwen4b":
            tokens = self.episode.qwen4b_token_reservation
            latency = self.episode.qwen4b_latency_reservation_s
        elif model_key == "qwen35b":
            tokens = self.episode.qwen35b_token_reservation
            latency = self.episode.qwen35b_latency_reservation_s
        else:
            raise ValueError(model_key)
        return (
            self.model_calls + 1 <= self.episode.model_call_budget
            and self.tokens_used + tokens <= self.episode.token_budget
            and self.latency_used_s + latency <= self.episode.latency_budget_s
        )

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        task = self.task
        if self.terminated or task is None:
            return mask
        mask[SequenceAction.STOP] = True
        mask[SequenceAction.DELEGATE] = not self.qwen4b_called and self._fits("qwen4b")
        mask[SequenceAction.ESCALATE] = not self.qwen35b_called and self._fits("qwen35b")
        mask[SequenceAction.REVIEW] = (
            self.current_candidate is not None
            and not self.review_used_on_task
            and self.reviews_used < self.episode.review_budget
        )
        mask[SequenceAction.HUMAN] = self.humans_used < self.episode.human_budget
        return mask

    def observation(self) -> np.ndarray:
        values = np.zeros(self.observation_size, dtype=np.float32)
        task = self.task
        if task is None:
            return values
        offset = 0
        values[offset + FAMILIES.index(task.family)] = 1.0
        offset += 4
        values[offset + (0 if task.authority == "local" else 1)] = 1.0
        offset += 2
        values[offset] = self.task_index / len(self.episode.tasks)
        values[offset + 1] = (len(self.episode.tasks) - self.task_index) / len(
            self.episode.tasks
        )
        offset += 2
        values[offset] = 1.0 - self.tokens_used / self.episode.token_budget
        values[offset + 1] = 1.0 - self.latency_used_s / self.episode.latency_budget_s
        values[offset + 2] = 1.0 - self.model_calls / self.episode.model_call_budget
        values[offset + 3] = 1.0 - self.reviews_used / self.episode.review_budget
        values[offset + 4] = 1.0 - self.humans_used / self.episode.human_budget
        offset += 5
        values[offset] = float(self.qwen4b_called)
        values[offset + 1] = float(self.qwen35b_called)
        offset += 2
        candidate_slot = 0 if self.current_candidate is None else self.current_candidate + 1
        values[offset + candidate_slot] = 1.0
        offset += 5
        source_slot = {None: 0, "qwen4b": 1, "qwen35b": 2}[self.candidate_source]
        values[offset + source_slot] = 1.0
        offset += 3
        values[offset + self.review_state] = 1.0
        offset += 3
        values[offset] = float(self.last_call_abstained)
        offset += 1
        values[offset : offset + ACTION_COUNT] = self.action_mask()
        return values

    def info(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "split": self.episode.split,
            "step": self.step_index,
            "task_index": self.task_index,
            "tasks": len(self.episode.tasks),
            "tokens_used": self.tokens_used,
            "latency_used_s": self.latency_used_s,
            "model_calls": self.model_calls,
            "reviews_used": self.reviews_used,
            "humans_used": self.humans_used,
            "autonomous_correct": self.autonomous_correct,
            "assisted": self.assisted,
            "failed": self.failed,
            "invalid_actions": self.invalid_actions,
            "budget_violations": self.budget_violations,
            "horizon_exhaustions": self.horizon_exhaustions,
            "unauthorized_source_calls": self.unauthorized_source_calls,
            "safety_violations": self.safety_violations,
            "post_step_budget_invariant_violations": (
                self.post_step_budget_invariant_violations
            ),
            "terminated": self.terminated,
            "total_reward": self.total_reward,
            "action_mask": self.action_mask().astype(int).tolist(),
        }

    def policy_state(self) -> SequencePolicyState:
        task = self.task
        if self.terminated or task is None:
            raise RuntimeError("policy state is unavailable after termination")
        return SequencePolicyState(
            family=task.family, authority=task.authority,
            task_index=self.task_index, tasks=len(self.episode.tasks),
            current_candidate=self.current_candidate,
            candidate_source=self.candidate_source,
            qwen4b_called=self.qwen4b_called, qwen35b_called=self.qwen35b_called,
            review_used_on_task=self.review_used_on_task,
            review_state=self.review_state,
            last_call_abstained=self.last_call_abstained,
            tokens_remaining=self.episode.token_budget - self.tokens_used,
            latency_remaining_s=self.episode.latency_budget_s - self.latency_used_s,
            model_calls_remaining=self.episode.model_call_budget - self.model_calls,
            reviews_remaining=self.episode.review_budget - self.reviews_used,
            humans_remaining=self.episode.human_budget - self.humans_used,
            action_mask=tuple(bool(value) for value in self.action_mask()),
        )

    def _advance(self) -> None:
        self.task_index += 1
        if self.task_index >= len(self.episode.tasks):
            self.terminated = True
        else:
            self._reset_task_state()

    def step(
        self, action: int | SequenceAction,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("step called after termination")
        action = SequenceAction(int(action))
        task = self.task
        assert task is not None
        before_task_id = task.task_id
        mask = self.action_mask()
        reward = SequenceReward()
        event: dict[str, Any] = {"action": ACTION_NAMES[action], "task_id": before_task_id}
        self.step_index += 1

        if not mask[action]:
            self.invalid_actions += 1
            if action in {SequenceAction.DELEGATE, SequenceAction.ESCALATE}:
                model_key = "qwen4b" if action == SequenceAction.DELEGATE else "qwen35b"
                if not self._fits(model_key):
                    self.budget_violations += 1
            reward = SequenceReward(invalid_action_penalty=-1.0, safety_penalty=-0.5)
            event["invalid"] = True
        elif action in {SequenceAction.DELEGATE, SequenceAction.ESCALATE}:
            call = task.qwen4b if action == SequenceAction.DELEGATE else task.qwen35b
            if action == SequenceAction.DELEGATE:
                self.qwen4b_called = True
            else:
                self.qwen35b_called = True
            self.model_calls += 1
            self.tokens_used += call.tokens
            self.latency_used_s += call.latency_s
            if (
                self.model_calls > self.episode.model_call_budget
                or self.tokens_used > self.episode.token_budget
                or self.latency_used_s > self.episode.latency_budget_s
            ):
                self.post_step_budget_invariant_violations += 1
            self.last_call_abstained = call.abstained
            if call.valid and not call.abstained and call.candidate is not None:
                self.current_candidate = call.candidate
                self.candidate_source = call.model_key
                self.review_state = 0
            reward = SequenceReward(
                token_penalty=-0.08 * call.tokens / self.episode.token_budget,
                latency_penalty=-0.04 * call.latency_s / self.episode.latency_budget_s,
            )
            event.update({
                "model": call.model_key, "valid": call.valid,
                "abstained": call.abstained, "candidate": call.candidate,
                "tokens": call.tokens, "latency_s": call.latency_s,
            })
        elif action == SequenceAction.REVIEW:
            self.reviews_used += 1
            self.review_used_on_task = True
            passed = self.current_candidate == task.correct_option
            self.review_state = 1 if passed else 2
            reward = SequenceReward(review_penalty=-0.01)
            event["review_passed"] = passed
        elif action == SequenceAction.HUMAN:
            self.humans_used += 1
            self.assisted += 1
            reward = SequenceReward(
                accuracy=0.25 / len(self.episode.tasks), human_penalty=-0.10,
            )
            event["assisted_resolution"] = True
            self._advance()
        elif action == SequenceAction.STOP:
            correct = self.current_candidate == task.correct_option
            if correct:
                self.autonomous_correct += 1
                reward = SequenceReward(accuracy=1.0 / len(self.episode.tasks))
            else:
                self.failed += 1
                if task.family == "safety_policy":
                    self.safety_violations += 1
                reward = SequenceReward(
                    accuracy=-0.25 / len(self.episode.tasks),
                    safety_penalty=-0.10 if task.family == "safety_policy" else 0.0,
                )
            event.update({"committed_candidate": self.current_candidate, "correct": correct})
            self._advance()

        if self.step_index >= self.episode.max_steps and not self.terminated:
            unresolved = len(self.episode.tasks) - self.task_index
            self.failed += unresolved
            self.terminated = True
            self.horizon_exhaustions += 1
            reward = SequenceReward(
                **{
                    **asdict(reward),
                    "safety_penalty": reward.safety_penalty - 0.5,
                },
            )
            event["horizon_exhausted"] = True
        self.total_reward += reward.total
        event["reward"] = reward.to_dict()
        event["resources_after"] = {
            "tokens": self.tokens_used, "latency_s": self.latency_used_s,
            "model_calls": self.model_calls, "reviews": self.reviews_used,
            "humans": self.humans_used,
        }
        self.trajectory.append(event)
        return self.observation(), reward.total, self.terminated, False, self.info()


Policy = Callable[[SequencePolicyState], SequenceAction]
PrivilegedPolicy = Callable[[MultiTownSemanticSequenceEnv], SequenceAction]


def always_4b_policy(state: SequencePolicyState) -> SequenceAction:
    if not state.qwen4b_called and state.action_mask[SequenceAction.DELEGATE]:
        return SequenceAction.DELEGATE
    if state.current_candidate is not None:
        return SequenceAction.STOP
    if state.action_mask[SequenceAction.ESCALATE]:
        return SequenceAction.ESCALATE
    if state.action_mask[SequenceAction.HUMAN]:
        return SequenceAction.HUMAN
    return SequenceAction.STOP


def always_35b_policy(state: SequencePolicyState) -> SequenceAction:
    if not state.qwen35b_called and state.action_mask[SequenceAction.ESCALATE]:
        return SequenceAction.ESCALATE
    if state.current_candidate is not None:
        return SequenceAction.STOP
    if state.action_mask[SequenceAction.DELEGATE]:
        return SequenceAction.DELEGATE
    if state.action_mask[SequenceAction.HUMAN]:
        return SequenceAction.HUMAN
    return SequenceAction.STOP


def review_cascade_policy(state: SequencePolicyState) -> SequenceAction:
    mask = state.action_mask
    if not state.qwen4b_called and mask[SequenceAction.DELEGATE]:
        return SequenceAction.DELEGATE
    if state.current_candidate is None:
        if mask[SequenceAction.ESCALATE]:
            return SequenceAction.ESCALATE
        return SequenceAction.HUMAN if mask[SequenceAction.HUMAN] else SequenceAction.STOP
    if not state.review_used_on_task and mask[SequenceAction.REVIEW]:
        return SequenceAction.REVIEW
    if state.review_state == 2 and not state.qwen35b_called and mask[SequenceAction.ESCALATE]:
        return SequenceAction.ESCALATE
    return SequenceAction.STOP


def family_review_cascade_policy(state: SequencePolicyState) -> SequenceAction:
    mask = state.action_mask
    if not state.qwen4b_called and mask[SequenceAction.DELEGATE]:
        return SequenceAction.DELEGATE
    if state.current_candidate is None:
        if mask[SequenceAction.ESCALATE]:
            return SequenceAction.ESCALATE
        return SequenceAction.HUMAN if mask[SequenceAction.HUMAN] else SequenceAction.STOP
    should_review = state.family in {
        "resource_dispatch", "safety_policy",
    }
    if should_review and not state.review_used_on_task and mask[SequenceAction.REVIEW]:
        return SequenceAction.REVIEW
    if state.review_state == 2 and not state.qwen35b_called and mask[SequenceAction.ESCALATE]:
        return SequenceAction.ESCALATE
    return SequenceAction.STOP


def family_model_policy(state: SequencePolicyState) -> SequenceAction:
    """Post-hoc train diagnostic; this family rule is not cross-fitted."""

    use_strong = state.family == "resource_dispatch"
    preferred = SequenceAction.ESCALATE if use_strong else SequenceAction.DELEGATE
    fallback = SequenceAction.DELEGATE if use_strong else SequenceAction.ESCALATE
    if state.action_mask[preferred]:
        return preferred
    if state.current_candidate is not None:
        return SequenceAction.STOP
    if state.action_mask[fallback]:
        return fallback
    if state.action_mask[SequenceAction.HUMAN]:
        return SequenceAction.HUMAN
    return SequenceAction.STOP


def hindsight_oracle_policy(env: MultiTownSemanticSequenceEnv) -> SequenceAction:
    """Non-deployable output oracle; it reads hidden correctness by design."""

    assert env.task is not None
    mask = env.action_mask()
    if not env.qwen4b_called and mask[SequenceAction.DELEGATE]:
        return SequenceAction.DELEGATE
    if env.current_candidate == env.task.correct_option:
        return SequenceAction.STOP
    strong_correct = (
        env.task.qwen35b.valid and not env.task.qwen35b.abstained
        and env.task.qwen35b.candidate == env.task.correct_option
    )
    if strong_correct and not env.qwen35b_called and mask[SequenceAction.ESCALATE]:
        return SequenceAction.ESCALATE
    if env.current_candidate is not None:
        return SequenceAction.STOP
    if mask[SequenceAction.HUMAN]:
        return SequenceAction.HUMAN
    return SequenceAction.STOP


def run_policy(
    episode: SemanticSequenceEpisode, policy: Policy,
) -> dict[str, Any]:
    env = MultiTownSemanticSequenceEnv(episode)
    env.reset()
    while not env.terminated:
        action = policy(env.policy_state())
        env.step(action)
    info = env.info()
    return {
        **info,
        "autonomous_accuracy": info["autonomous_correct"] / len(episode.tasks),
        "assisted_completion_rate": (
            info["autonomous_correct"] + info["assisted"]
        ) / len(episode.tasks),
        "trajectory": env.trajectory,
    }


def run_privileged_policy(
    episode: SemanticSequenceEpisode, policy: PrivilegedPolicy,
) -> dict[str, Any]:
    """Run an explicitly non-deployable policy that may inspect hidden replay state."""

    env = MultiTownSemanticSequenceEnv(episode)
    env.reset()
    while not env.terminated:
        env.step(policy(env))
    info = env.info()
    return {
        **info,
        "autonomous_accuracy": info["autonomous_correct"] / len(episode.tasks),
        "assisted_completion_rate": (
            info["autonomous_correct"] + info["assisted"]
        ) / len(episode.tasks),
        "trajectory": env.trajectory,
        "privileged_policy": True,
    }


def freeze_bank(
    output_dir: Path, episodes: list[SemanticSequenceEpisode], *,
    semantic_bank_sha256: str, outcomes_sha256: str,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot freeze an empty sequence bank")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    if dirty:
        raise RuntimeError("A14 sequence bank freeze requires a clean source revision")
    for episode in episodes:
        if len(episode.tasks) != len(FAMILIES):
            raise ValueError("frozen A14 episodes require one task per semantic family")
        if {task.family for task in episode.tasks} != set(FAMILIES):
            raise ValueError("frozen A14 episode family coverage mismatch")
    output_dir.mkdir(parents=True)
    bank_path = output_dir / "train.jsonl"
    with bank_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    payload = bank_path.read_bytes()
    manifest = {
        "schema_version": BANK_VERSION,
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "split": "train",
        "episodes": len(episodes),
        "tasks_per_episode": len(episodes[0].tasks),
        "unique_atomic_tasks": len({
            task.task_id for episode in episodes for task in episode.tasks
        }),
        "semantic_bank_sha256": semantic_bank_sha256,
        "outcomes_sha256": outcomes_sha256,
        "train_jsonl": {
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "claim_boundary": (
            "compositions reuse frozen train-only atomic traces; episodes are not independent "
            "model samples and are not development, held-out, or OOD data"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-bank", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    semantic_bank = Path(args.semantic_bank)
    outcomes_path = Path(args.outcomes)
    tasks = read_bank(semantic_bank)
    first_outcome = json.loads(next(
        line for line in outcomes_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ))
    if first_outcome.get("schema_version") != "multitown-semantic-model-outcome-v2":
        raise ValueError("A14 requires abstention-aware A13 v3 outcomes")
    outcomes = read_outcomes(outcomes_path)
    replay_tasks = build_replay_tasks(tasks, outcomes)
    episodes = episode_bank(replay_tasks, args.episodes, seed=args.seed)
    manifest = freeze_bank(
        Path(args.output_dir), episodes,
        semantic_bank_sha256=_sha256_file(semantic_bank),
        outcomes_sha256=_sha256_file(outcomes_path),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
