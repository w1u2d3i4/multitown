"""Deterministic 20--50 step partially observable MultiTown RL environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ENV_VERSION = "multitown-long-horizon-pomdp-v1"
ACTION_COUNT = 8
FAMILY_COUNT = 4
CANDIDATE_COUNT = 4
HELD_OUT_COMBINATIONS = ((0, 3), (1, 0), (2, 1), (3, 2))


class RLAction(IntEnum):
    OBSERVE = 0
    DELEGATE = 1
    ESCALATE = 2
    CONNECT = 3
    REVIEW = 4
    EXECUTE = 5
    HUMAN = 6
    STOP = 7


ACTION_NAMES = tuple(action.name.lower() for action in RLAction)
ACTION_COSTS: dict[RLAction, tuple[int, float]] = {
    RLAction.OBSERVE: (32, 0.08),
    RLAction.DELEGATE: (210, 0.48),
    RLAction.ESCALATE: (330, 0.82),
    RLAction.CONNECT: (48, 0.10),
    RLAction.REVIEW: (260, 0.62),
    RLAction.EXECUTE: (20, 0.05),
    RLAction.HUMAN: (520, 0.20),
    RLAction.STOP: (0, 0.0),
}


@dataclass(frozen=True)
class LongHorizonReward:
    final_success: float = 0.0
    subgoal_progress: float = 0.0
    action_cost: float = 0.0
    invalid_action: float = 0.0
    budget_violation: float = 0.0
    safety_penalty: float = 0.0
    tool_failure_recovery: float = 0.0
    unnecessary_delegation: float = 0.0
    human_penalty: float = 0.0

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


@dataclass(frozen=True)
class IncidentSpec:
    family: int
    failure_mode: int
    severity: float
    correct_action: int
    sensor_candidate: int
    weak_candidate: int
    strong_candidate: int
    reviewer_pass: tuple[bool, ...]
    fail_first_actions: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IncidentSpec":
        return cls(
            family=int(value["family"]),
            failure_mode=int(value["failure_mode"]),
            severity=float(value["severity"]),
            correct_action=int(value["correct_action"]),
            sensor_candidate=int(value["sensor_candidate"]),
            weak_candidate=int(value["weak_candidate"]),
            strong_candidate=int(value["strong_candidate"]),
            reviewer_pass=tuple(bool(item) for item in value["reviewer_pass"]),
            fail_first_actions=tuple(int(item) for item in value["fail_first_actions"]),
        )


@dataclass(frozen=True)
class LongHorizonEpisode:
    episode_id: str
    split: str
    seed: int
    token_budget: int
    latency_budget_s: float
    max_steps: int
    incidents: tuple[IncidentSpec, ...]
    schema_version: str = ENV_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["incidents"] = [asdict(item) for item in self.incidents]
        return payload

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, expected_schema_version: str = ENV_VERSION,
    ) -> "LongHorizonEpisode":
        if value.get("schema_version") != expected_schema_version:
            raise ValueError(f"unsupported environment schema: {value.get('schema_version')}")
        return cls(
            episode_id=str(value["episode_id"]),
            split=str(value["split"]),
            seed=int(value["seed"]),
            token_budget=int(value["token_budget"]),
            latency_budget_s=float(value["latency_budget_s"]),
            max_steps=int(value["max_steps"]),
            incidents=tuple(IncidentSpec.from_dict(item) for item in value["incidents"]),
            schema_version=str(value["schema_version"]),
        )


def _wrong_candidate(rng: random.Random, correct: int) -> int:
    choices = [item for item in range(CANDIDATE_COUNT) if item != correct]
    return rng.choice(choices)


def _noisy_candidate(rng: random.Random, correct: int, probability: float) -> int:
    return correct if rng.random() < probability else _wrong_candidate(rng, correct)


def generate_episode(seed: int, split: str) -> LongHorizonEpisode:
    if split not in {"train", "dev", "test"}:
        raise ValueError(split)
    rng = random.Random(f"{ENV_VERSION}:{split}:{seed}")
    all_combinations = [(family, mode) for family in range(4) for mode in range(4)]
    if split == "test":
        combinations = list(HELD_OUT_COMBINATIONS)
        tool_failure_probability = 0.24
    else:
        combinations = [item for item in all_combinations if item not in HELD_OUT_COMBINATIONS]
        tool_failure_probability = 0.10 if split == "train" else 0.14
    incident_count = rng.randint(4, 7)
    incidents: list[IncidentSpec] = []
    for _ in range(incident_count):
        family, failure_mode = rng.choice(combinations)
        severity = rng.uniform(0.15, 1.0)
        correct = (family + 2 * failure_mode + int(severity >= 0.7)) % CANDIDATE_COUNT
        sensor_accuracy = 0.50 + 0.18 * float(severity < 0.55)
        # The fixed worker pool is deliberately heterogeneous: the weak local
        # specialist is best on field/evidence families, while the strong
        # generalist is best on dispatch/dependency families. This creates a
        # real organization decision instead of a globally dominant model.
        weak_accuracy = (
            0.90 if family in {0, 2} else 0.58
        ) - 0.06 * float(severity >= 0.8)
        strong_accuracy = (
            0.72 if family in {0, 2} else 0.90
        ) - 0.04 * float(split == "test")
        reviewer_pass = tuple(
            (rng.random() < 0.91) if candidate == correct else (rng.random() < 0.08)
            for candidate in range(CANDIDATE_COUNT)
        )
        failure_candidates = (
            RLAction.OBSERVE, RLAction.DELEGATE, RLAction.ESCALATE,
            RLAction.CONNECT, RLAction.REVIEW,
        )
        fail_first = tuple(
            int(action) for action in failure_candidates
            if rng.random() < tool_failure_probability
        )
        incidents.append(IncidentSpec(
            family=family,
            failure_mode=failure_mode,
            severity=severity,
            correct_action=correct,
            sensor_candidate=_noisy_candidate(rng, correct, sensor_accuracy),
            weak_candidate=_noisy_candidate(rng, correct, weak_accuracy),
            strong_candidate=_noisy_candidate(rng, correct, strong_accuracy),
            reviewer_pass=reviewer_pass,
            fail_first_actions=fail_first,
        ))
    max_steps = min(50, max(20, incident_count * 6 + 6))
    return LongHorizonEpisode(
        episode_id=f"lh-{split}-{seed:08d}",
        split=split,
        seed=seed,
        token_budget=incident_count * 650,
        latency_budget_s=incident_count * 1.65,
        max_steps=max_steps,
        incidents=tuple(incidents),
    )


def episode_bank(split: str, count: int, *, seed_offset: int) -> list[LongHorizonEpisode]:
    if count <= 0:
        raise ValueError("episode count must be positive")
    return [generate_episode(seed_offset + index, split) for index in range(count)]


def read_episode_bank(path: Path, *, split: str | None = None) -> list[LongHorizonEpisode]:
    episodes = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                episode = LongHorizonEpisode.from_dict(json.loads(line))
                if split is None or episode.split == split:
                    episodes.append(episode)
    if not episodes:
        raise ValueError(f"no episodes loaded from {path} for split={split!r}")
    return episodes


class MultiTownLongHorizonEnv:
    """Small Gym-like environment with hidden world state and fixed agent tools."""

    observation_size = 47

    def __init__(self, episode: LongHorizonEpisode):
        self.episode = episode
        self.reset()

    @property
    def incident(self) -> IncidentSpec | None:
        if self.incident_index >= len(self.episode.incidents):
            return None
        return self.episode.incidents[self.incident_index]

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.step_index = 0
        self.incident_index = 0
        self.tokens_used = 0
        self.latency_used_s = 0.0
        self.resolved = 0
        self.failed = 0
        self.human_escalations = 0
        self.tool_failures = 0
        self.tool_recoveries = 0
        self.invalid_actions = 0
        self.budget_violations = 0
        self.terminated = False
        self.episode_success = False
        self.assisted_episode_success = False
        self.trajectory: list[dict[str, Any]] = []
        self._reset_incident_state()
        return self.observation(), self.info()

    def _reset_incident_state(self) -> None:
        self.observed = False
        self.current_candidate: int | None = None
        self.weak_candidate: int | None = None
        self.strong_candidate: int | None = None
        self.connected = False
        self.review_state = 0  # 0 unknown, 1 pass, 2 fail
        self.last_tool_failed = False
        self.last_action: int | None = None
        self.action_attempts = [0] * ACTION_COUNT

    def action_cost(self, action: RLAction) -> tuple[int, float]:
        """Return the token and latency cost for an action.

        Subclasses may replace fixed simulator costs with measured model-call
        telemetry while preserving the same transition and reward contract.
        """
        return ACTION_COSTS[action]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        if self.terminated:
            return mask
        if self.incident is None:
            mask[RLAction.STOP] = True
            return mask
        mask[RLAction.HUMAN] = True
        mask[RLAction.STOP] = True
        if not self.observed:
            mask[RLAction.OBSERVE] = True
        if self.weak_candidate is None:
            mask[RLAction.DELEGATE] = True
        if self.strong_candidate is None:
            mask[RLAction.ESCALATE] = True
        if (self.weak_candidate is not None or self.strong_candidate is not None) and not self.connected:
            mask[RLAction.CONNECT] = True
        if self.current_candidate is not None and self.review_state == 0:
            mask[RLAction.REVIEW] = True
        if self.current_candidate is not None:
            mask[RLAction.EXECUTE] = True
        for action in RLAction:
            token_cost, latency_cost = self.action_cost(action)
            if (
                self.tokens_used + token_cost > self.episode.token_budget
                or self.latency_used_s + latency_cost > self.episode.latency_budget_s
            ):
                mask[action] = False
        if not mask.any():
            mask[RLAction.STOP] = True
        return mask

    def observation(self) -> np.ndarray:
        values: list[float] = []
        incident = self.incident
        family = incident.family if incident else -1
        failure = incident.failure_mode if incident else -1
        values.extend(float(index == family) for index in range(FAMILY_COUNT))
        # Failure-mode identity is observable as a coarse tool/error category.
        values.extend(float(index == failure) for index in range(4))
        values.append(float(incident.severity) if incident else 0.0)
        values.extend([
            self.incident_index / len(self.episode.incidents),
            self.step_index / self.episode.max_steps,
            self.tokens_used / self.episode.token_budget,
            self.latency_used_s / self.episode.latency_budget_s,
            self.resolved / len(self.episode.incidents),
            self.failed / len(self.episode.incidents),
            float(self.observed),
            float(self.connected),
            float(self.last_tool_failed),
        ])
        for candidate in (self.current_candidate, self.weak_candidate, self.strong_candidate):
            values.extend(float(candidate == index) for index in range(CANDIDATE_COUNT))
            values.append(float(candidate is None))
        values.extend(float(self.review_state == index) for index in range(3))
        values.extend(float(self.last_action == index) for index in range(ACTION_COUNT))
        values.extend([
            min(1.0, self.tool_failures / 5.0),
            min(1.0, self.tool_recoveries / 5.0),
            min(1.0, self.human_escalations / len(self.episode.incidents)),
        ])
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.observation_size,):
            raise RuntimeError(f"observation shape mismatch: {result.shape}")
        return result

    def _tool_fails(self, action: RLAction) -> bool:
        incident = self.incident
        return bool(
            incident
            and int(action) in incident.fail_first_actions
            and self.action_attempts[int(action)] == 1
        )

    def _fuse_candidates(self) -> int:
        candidates = [
            item for item in (self.current_candidate, self.weak_candidate, self.strong_candidate)
            if item is not None
        ]
        counts = {item: candidates.count(item) for item in set(candidates)}
        best_count = max(counts.values())
        winners = {item for item, count in counts.items() if count == best_count}
        for preferred in (self.strong_candidate, self.weak_candidate, self.current_candidate):
            if preferred in winners:
                return int(preferred)
        raise RuntimeError("cannot fuse an empty candidate set")

    def _advance_incident(self, correct: bool) -> None:
        if correct:
            self.resolved += 1
        else:
            self.failed += 1
        self.incident_index += 1
        self._reset_incident_state()

    def step(self, action_value: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("step called after termination")
        action = RLAction(action_value)
        before_mask = self.action_mask()
        incident_before = self.incident_index
        token_cost, latency_cost = self.action_cost(action)
        reward_values = {name: 0.0 for name in LongHorizonReward.__dataclass_fields__}
        reward_values["action_cost"] = -0.04 * token_cost / 1000.0 - 0.01 * latency_cost
        self.step_index += 1
        previous_last_action = self.last_action
        self.last_action = int(action)

        if not before_mask[int(action)]:
            self.invalid_actions += 1
            reward_values["invalid_action"] = -1.0
        else:
            self.tokens_used += token_cost
            self.latency_used_s += latency_cost
            if (
                self.tokens_used > self.episode.token_budget
                or self.latency_used_s > self.episode.latency_budget_s
            ):
                self.budget_violations += 1
                reward_values["budget_violation"] = -2.0
                self.terminated = True
            elif action == RLAction.STOP:
                self.assisted_episode_success = (
                    self.incident is None
                    and self.failed == 0
                    and self.resolved == len(self.episode.incidents)
                )
                self.episode_success = self.assisted_episode_success and self.human_escalations == 0
                if self.episode_success:
                    reward_values["final_success"] = 2.0
                elif self.assisted_episode_success:
                    reward_values["final_success"] = -0.5
                else:
                    reward_values["final_success"] = -2.0
                self.terminated = True
            elif action == RLAction.HUMAN:
                self.human_escalations += 1
                reward_values["human_penalty"] = -0.80
                self._advance_incident(True)
            else:
                self.action_attempts[int(action)] += 1
                was_recovery = self.last_tool_failed and previous_last_action == int(action)
                if self._tool_fails(action):
                    self.tool_failures += 1
                    self.last_tool_failed = True
                else:
                    if was_recovery:
                        self.tool_recoveries += 1
                        reward_values["tool_failure_recovery"] = 0.08
                    self.last_tool_failed = False
                    incident = self.incident
                    if incident is None:
                        raise RuntimeError("missing incident")
                    if action == RLAction.OBSERVE:
                        self.observed = True
                        self.current_candidate = incident.sensor_candidate
                        self.review_state = 0
                    elif action == RLAction.DELEGATE:
                        if self.current_candidate == incident.correct_action and self.review_state == 1:
                            reward_values["unnecessary_delegation"] = -0.10
                        self.weak_candidate = incident.weak_candidate
                        self.connected = False
                    elif action == RLAction.ESCALATE:
                        if self.current_candidate == incident.correct_action and self.review_state == 1:
                            reward_values["unnecessary_delegation"] = -0.16
                        self.strong_candidate = incident.strong_candidate
                        self.connected = False
                    elif action == RLAction.CONNECT:
                        self.current_candidate = self._fuse_candidates()
                        self.connected = True
                        self.review_state = 0
                    elif action == RLAction.REVIEW:
                        self.review_state = 1 if incident.reviewer_pass[self.current_candidate] else 2
                    elif action == RLAction.EXECUTE:
                        correct = self.current_candidate == incident.correct_action
                        if correct:
                            reward_values["subgoal_progress"] = 1.0
                        else:
                            reward_values["safety_penalty"] = -1.0 - 0.5 * incident.severity
                        self._advance_incident(correct)

        truncated = False
        if not self.terminated and self.step_index >= self.episode.max_steps:
            reward_values["final_success"] = -1.5
            self.terminated = True
            truncated = True
        reward = LongHorizonReward(**reward_values)
        row = {
            "step_index": self.step_index - 1,
            "incident_index": incident_before,
            "action": ACTION_NAMES[int(action)],
            "reward": reward.to_dict(),
            "tokens_used": self.tokens_used,
            "latency_used_s": self.latency_used_s,
            "tool_failed": self.last_tool_failed,
            "terminated": self.terminated,
            "truncated": truncated,
        }
        self.trajectory.append(row)
        return self.observation(), reward.total, self.terminated, truncated, self.info()

    def info(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "split": self.episode.split,
            "episode_success": self.episode_success,
            "assisted_episode_success": self.assisted_episode_success,
            "resolved": self.resolved,
            "failed": self.failed,
            "incidents": len(self.episode.incidents),
            "steps": self.step_index,
            "tokens_used": self.tokens_used,
            "latency_used_s": self.latency_used_s,
            "human_escalations": self.human_escalations,
            "tool_failures": self.tool_failures,
            "tool_recoveries": self.tool_recoveries,
            "invalid_actions": self.invalid_actions,
            "budget_violations": self.budget_violations,
        }


Policy = Callable[[MultiTownLongHorizonEnv, np.ndarray, np.ndarray], int]


def run_policy(episode: LongHorizonEpisode, policy: Policy) -> dict[str, Any]:
    env = MultiTownLongHorizonEnv(episode)
    observation, _ = env.reset()
    total_reward = 0.0
    while not env.terminated:
        mask = env.action_mask()
        action = int(policy(env, observation, mask))
        observation, reward, _, _, _ = env.step(action)
        total_reward += reward
    return {**env.info(), "return": total_reward, "trajectory": env.trajectory}


def a8_heuristic_policy(env: MultiTownLongHorizonEnv, _: np.ndarray, mask: np.ndarray) -> int:
    if env.incident is None:
        return int(RLAction.STOP)
    if env.last_tool_failed and env.last_action is not None and mask[env.last_action]:
        return env.last_action
    if not env.observed and mask[RLAction.OBSERVE]:
        return int(RLAction.OBSERVE)
    if env.current_candidate is not None and env.review_state == 0 and mask[RLAction.REVIEW]:
        return int(RLAction.REVIEW)
    if env.review_state == 1 and mask[RLAction.EXECUTE]:
        return int(RLAction.EXECUTE)
    if env.weak_candidate is None and mask[RLAction.DELEGATE]:
        return int(RLAction.DELEGATE)
    if not env.connected and mask[RLAction.CONNECT]:
        return int(RLAction.CONNECT)
    if env.strong_candidate is None and mask[RLAction.ESCALATE]:
        return int(RLAction.ESCALATE)
    if not env.connected and mask[RLAction.CONNECT]:
        return int(RLAction.CONNECT)
    if env.review_state == 0 and mask[RLAction.REVIEW]:
        return int(RLAction.REVIEW)
    if mask[RLAction.HUMAN]:
        return int(RLAction.HUMAN)
    return int(RLAction.STOP)


def strong_only_policy(env: MultiTownLongHorizonEnv, _: np.ndarray, mask: np.ndarray) -> int:
    if env.incident is None:
        return int(RLAction.STOP)
    if env.last_tool_failed and env.last_action is not None and mask[env.last_action]:
        return env.last_action
    if env.strong_candidate is None and mask[RLAction.ESCALATE]:
        return int(RLAction.ESCALATE)
    if not env.connected and mask[RLAction.CONNECT]:
        return int(RLAction.CONNECT)
    if mask[RLAction.EXECUTE]:
        return int(RLAction.EXECUTE)
    if mask[RLAction.HUMAN]:
        return int(RLAction.HUMAN)
    return int(RLAction.STOP)


def weak_only_policy(env: MultiTownLongHorizonEnv, _: np.ndarray, mask: np.ndarray) -> int:
    if env.incident is None:
        return int(RLAction.STOP)
    if env.last_tool_failed and env.last_action is not None and mask[env.last_action]:
        return env.last_action
    if env.weak_candidate is None and mask[RLAction.DELEGATE]:
        return int(RLAction.DELEGATE)
    if not env.connected and mask[RLAction.CONNECT]:
        return int(RLAction.CONNECT)
    if mask[RLAction.EXECUTE]:
        return int(RLAction.EXECUTE)
    if mask[RLAction.HUMAN]:
        return int(RLAction.HUMAN)
    return int(RLAction.STOP)


def oracle_policy(env: MultiTownLongHorizonEnv, _: np.ndarray, mask: np.ndarray) -> int:
    if env.incident is None:
        return int(RLAction.STOP)
    if env.last_tool_failed and env.last_action is not None and mask[env.last_action]:
        return env.last_action
    incident = env.incident
    if env.current_candidate == incident.correct_action and mask[RLAction.EXECUTE]:
        return int(RLAction.EXECUTE)
    if not env.observed and incident.sensor_candidate == incident.correct_action and mask[RLAction.OBSERVE]:
        return int(RLAction.OBSERVE)
    if env.weak_candidate is None and incident.weak_candidate == incident.correct_action and mask[RLAction.DELEGATE]:
        return int(RLAction.DELEGATE)
    if env.strong_candidate is None and incident.strong_candidate == incident.correct_action and mask[RLAction.ESCALATE]:
        return int(RLAction.ESCALATE)
    if not env.connected and mask[RLAction.CONNECT]:
        available = [item for item in (env.weak_candidate, env.strong_candidate) if item is not None]
        if incident.correct_action in available:
            return int(RLAction.CONNECT)
    return int(RLAction.HUMAN if mask[RLAction.HUMAN] else RLAction.STOP)


def random_policy_factory(seed: int) -> Policy:
    rng = random.Random(seed)

    def policy(_: MultiTownLongHorizonEnv, __: np.ndarray, mask: np.ndarray) -> int:
        return rng.choice(np.flatnonzero(mask).tolist())

    return policy


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "episodes": count,
        "episode_success_rate": sum(bool(row["episode_success"]) for row in rows) / count,
        "assisted_episode_success_rate": sum(
            bool(row["assisted_episode_success"]) for row in rows
        ) / count,
        "subgoal_completion_rate": sum(int(row["resolved"]) for row in rows) /
            sum(int(row["incidents"]) for row in rows),
        "mean_return": sum(float(row["return"]) for row in rows) / count,
        "tokens_per_episode": sum(int(row["tokens_used"]) for row in rows) / count,
        "tokens_per_success": (
            sum(int(row["tokens_used"]) for row in rows) /
            max(1, sum(bool(row["episode_success"]) for row in rows))
        ),
        "latency_per_episode_s": sum(float(row["latency_used_s"]) for row in rows) / count,
        "energy_per_episode_j": sum(float(row.get("energy_used_j", 0.0)) for row in rows) / count,
        "steps_per_episode": sum(int(row["steps"]) for row in rows) / count,
        "human_rate": sum(int(row["human_escalations"]) for row in rows) /
            sum(int(row["incidents"]) for row in rows),
        "tool_recovery_rate": (
            sum(int(row["tool_recoveries"]) for row in rows) /
            max(1, sum(int(row["tool_failures"]) for row in rows))
        ),
        "budget_violations": sum(int(row["budget_violations"]) for row in rows),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_bank(output: Path, *, train: int, dev: int, test: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()
    # Capture provenance before creating the bank. Otherwise the newly-created,
    # intentionally tracked output makes an initially clean checkout look dirty.
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip())
    output.mkdir(parents=True, exist_ok=False)
    counts = {"train": train, "dev": dev, "test": test}
    offsets = {"train": 10_000_000, "dev": 20_000_000, "test": 30_000_000}
    files = {}
    for split in ("train", "dev", "test"):
        episodes = episode_bank(split, counts[split], seed_offset=offsets[split])
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")
        files[path.name] = {
            "episodes": len(episodes),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "mean_incidents": sum(len(item.incidents) for item in episodes) / len(episodes),
            "min_horizon": min(item.max_steps for item in episodes),
            "max_horizon": max(item.max_steps for item in episodes),
        }
    manifest = {
        "schema_version": "multitown-long-horizon-bank-manifest-v1",
        "environment_version": ENV_VERSION,
        "source_revision": revision,
        "source_dirty": dirty,
        "split_protocol": {
            "train_dev_combinations": "12/16 family x failure-mode combinations",
            "held_out_test_combinations": [list(item) for item in HELD_OUT_COMBINATIONS],
            "test_is_ood": True,
            "seed_offsets": offsets,
        },
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/multitown-long-horizon-v0.1"))
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--dev", type=int, default=500)
    parser.add_argument("--test", type=int, default=1000)
    args = parser.parse_args()
    manifest = freeze_bank(args.output, train=args.train, dev=args.dev, test=args.test)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
