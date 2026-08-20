"""Collect real model worker outcomes and replay them in the long-horizon environment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import httpx

from .client import ModelResponse, stream_chat_completion
from .long_horizon_env import (
    ACTION_COSTS,
    CANDIDATE_COUNT,
    IncidentSpec,
    LongHorizonEpisode,
    MultiTownLongHorizonEnv,
    RLAction,
    a8_heuristic_policy,
    oracle_policy,
    read_episode_bank,
    run_policy,
    strong_only_policy,
    summarize_results,
    weak_only_policy,
)


SCHEMA_VERSION = "multitown-live-model-outcome-v1"
MANIFEST_VERSION = "multitown-live-model-manifest-v1"
PROMPT_VERSION = "multitown-live-prompt-v4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EndpointSpec:
    role: str
    endpoint: str
    model: str


@dataclass(frozen=True)
class MeasuredCall:
    role: str
    endpoint_role: str
    model: str
    request_sha256: str
    response_content: str
    parsed: Any
    valid: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    ttft_s: float | None
    finish_reason: str | None
    reasoning_chars: int
    error: str | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MeasuredCall":
        return cls(
            role=str(value["role"]), endpoint_role=str(value["endpoint_role"]),
            model=str(value["model"]), request_sha256=str(value["request_sha256"]),
            response_content=str(value["response_content"]), parsed=value.get("parsed"),
            valid=bool(value["valid"]), prompt_tokens=int(value["prompt_tokens"]),
            completion_tokens=int(value["completion_tokens"]),
            total_tokens=int(value["total_tokens"]), latency_s=float(value["latency_s"]),
            ttft_s=None if value.get("ttft_s") is None else float(value["ttft_s"]),
            finish_reason=value.get("finish_reason"),
            reasoning_chars=int(value.get("reasoning_chars", 0)), error=value.get("error"),
        )


@dataclass(frozen=True)
class MeasuredIncidentOutcome:
    episode_id: str
    split: str
    incident_index: int
    incident_sha256: str
    correct_action: int
    weak: MeasuredCall
    strong: MeasuredCall
    reviewer: MeasuredCall
    schema_version: str = SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, int]:
        return self.episode_id, self.incident_index

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MeasuredIncidentOutcome":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported live outcome schema: {value.get('schema_version')}")
        return cls(
            episode_id=str(value["episode_id"]), split=str(value["split"]),
            incident_index=int(value["incident_index"]),
            incident_sha256=str(value["incident_sha256"]),
            correct_action=int(value["evaluator"]["correct_action"]),
            weak=MeasuredCall.from_dict(value["calls"]["weak"]),
            strong=MeasuredCall.from_dict(value["calls"]["strong"]),
            reviewer=MeasuredCall.from_dict(value["calls"]["reviewer"]),
        )


def incident_sha256(incident: IncidentSpec) -> str:
    return _canonical_hash(asdict(incident))


def _common_incident_text(incident: IncidentSpec) -> str:
    return (
        "Incident record (use the numeric fields exactly):\n"
        f"family_code={incident.family}\n"
        f"failure_code={incident.failure_mode}\n"
        f"severity={incident.severity:.4f}\n"
        f"high_severity={int(incident.severity >= 0.7)}"
    )


def worker_messages(role: str, incident: IncidentSpec) -> list[dict[str, str]]:
    if role == "weak":
        system = (
            "Compute candidate = (family_code + 2*failure_code + high_severity) mod 4. "
            "Use only the numeric incident fields. Return only one JSON object of the form "
            "{\"candidate\": INTEGER}."
        )
    elif role == "strong":
        system = (
            "Compute candidate = (family_code + 2*failure_code + high_severity) mod 4. "
            "Use only the numeric incident fields. Return only one JSON object of the form "
            "{\"candidate\": INTEGER}."
        )
    else:
        raise ValueError(role)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _common_incident_text(incident)},
    ]


def reviewer_messages(incident: IncidentSpec) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Independently compute candidate = "
                "(family_code + 2*failure_code + high_severity) mod 4. Use only the numeric "
                "incident fields. Return only one JSON object of the form "
                "{\"candidate\": INTEGER}."
            ),
        },
        {"role": "user", "content": _common_incident_text(incident)},
    ]


def _parse_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def _validated_parsed(role: str, content: str) -> tuple[Any, bool, str | None]:
    try:
        payload = _parse_content(content)
        if role in {"weak", "strong"}:
            value = payload.get("candidate") if isinstance(payload, dict) else None
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
                raise ValueError("candidate must be an integer in [0, 3]")
            return int(value), True, None
        approved = payload.get("candidate") if isinstance(payload, dict) else None
        if isinstance(approved, bool) or not isinstance(approved, int) or not 0 <= approved < 4:
            raise ValueError("approved_candidate must be an integer in [0, 3]")
        return [index == approved for index in range(CANDIDATE_COUNT)], True, None
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"


async def call_role(
    client: httpx.AsyncClient,
    endpoint: EndpointSpec,
    incident: IncidentSpec,
    *,
    seed: int,
) -> MeasuredCall:
    messages = (
        worker_messages(endpoint.role, incident)
        if endpoint.role in {"weak", "strong"}
        else reviewer_messages(incident)
    )
    request = {
        "messages": messages, "seed": seed, "max_tokens": 64,
        "temperature": 0.0, "top_p": 1.0, "response_format": None,
    }
    response: ModelResponse = await stream_chat_completion(
        client,
        endpoint=endpoint.endpoint,
        model=endpoint.model,
        api_key="local-no-secret",
        messages=messages,
        seed=seed,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        response_format=None,
    )
    parsed, valid, validation_error = _validated_parsed(endpoint.role, response.content)
    if response.error is None and response.total_tokens <= 0:
        valid = False
        validation_error = "ValueError: missing positive token usage"
    errors = [item for item in (response.error, validation_error) if item]
    return MeasuredCall(
        role=endpoint.role, endpoint_role=endpoint.role, model=endpoint.model,
        request_sha256=_canonical_hash(request), response_content=response.content,
        parsed=parsed, valid=valid and response.error is None,
        prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens, latency_s=response.latency_s,
        ttft_s=response.ttft_s, finish_reason=response.finish_reason,
        reasoning_chars=response.reasoning_chars, error="; ".join(errors) or None,
    )


async def collect_incident(
    client: httpx.AsyncClient,
    episode: LongHorizonEpisode,
    incident_index: int,
    endpoints: dict[str, EndpointSpec],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    incident = episode.incidents[incident_index]
    seed = episode.seed * 17 + incident_index * 101

    async def limited(role: str, offset: int) -> MeasuredCall:
        async with semaphore:
            return await call_role(client, endpoints[role], incident, seed=seed + offset)

    weak, strong, reviewer = await asyncio.gather(
        limited("weak", 1), limited("strong", 2), limited("reviewer", 3),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode.episode_id,
        "split": episode.split,
        "incident_index": incident_index,
        "incident_sha256": incident_sha256(incident),
        "evaluator": {"correct_action": incident.correct_action},
        "calls": {
            "weak": asdict(weak), "strong": asdict(strong), "reviewer": asdict(reviewer),
        },
    }


def read_outcomes(path: Path) -> list[MeasuredIncidentOutcome]:
    outcomes = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                outcomes.append(MeasuredIncidentOutcome.from_dict(json.loads(line)))
    if not outcomes:
        raise ValueError(f"no measured outcomes in {path}")
    keys = [item.key for item in outcomes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode/incident keys in measured outcomes")
    return outcomes


def summarize_outcomes(outcomes: Iterable[MeasuredIncidentOutcome]) -> dict[str, Any]:
    rows = list(outcomes)
    result: dict[str, Any] = {"incidents": len(rows), "roles": {}}
    for role in ("weak", "strong", "reviewer"):
        calls = [getattr(item, role) for item in rows]
        valid = [call for call in calls if call.valid]
        if role == "reviewer":
            correct = sum(
                call.valid and call.parsed == [index == item.correct_action for index in range(4)]
                for item, call in zip(rows, calls, strict=True)
            )
        else:
            correct = sum(call.valid and call.parsed == item.correct_action for item, call in zip(rows, calls, strict=True))
        result["roles"][role] = {
            "model": sorted({call.model for call in calls}),
            "valid_rate": len(valid) / len(calls),
            "accuracy": correct / len(calls),
            "mean_total_tokens": statistics.mean(call.total_tokens for call in calls),
            "mean_latency_s": statistics.mean(call.latency_s for call in calls),
            "p95_latency_s": sorted(call.latency_s for call in calls)[min(len(calls) - 1, int(len(calls) * 0.95))],
            "errors": sum(call.error is not None for call in calls),
        }
    return result


def summarize_outcomes_with_episodes(
    outcomes: list[MeasuredIncidentOutcome], episodes: list[LongHorizonEpisode],
) -> dict[str, Any]:
    summary = summarize_outcomes(outcomes)
    episode_index = {item.episode_id: item for item in episodes}
    for role in ("weak", "strong"):
        family_rows: dict[int, list[bool]] = defaultdict(list)
        for item in outcomes:
            incident = episode_index[item.episode_id].incidents[item.incident_index]
            call = getattr(item, role)
            family_rows[incident.family].append(call.valid and call.parsed == item.correct_action)
        summary["roles"][role]["accuracy_by_family"] = {
            str(family): sum(values) / len(values) for family, values in sorted(family_rows.items())
        }
    return summary


class ModelBackedLongHorizonEnv(MultiTownLongHorizonEnv):
    """Long-horizon transitions using cached real-model candidates and costs."""

    def __init__(
        self, episode: LongHorizonEpisode, outcomes: list[MeasuredIncidentOutcome],
        *,
        energy_profile_j: dict[str, float] | None = None,
        energy_penalty_per_j: float = 0.0,
    ):
        if energy_penalty_per_j < 0:
            raise ValueError("energy_penalty_per_j cannot be negative")
        self.energy_profile_j = {
            "weak": 0.0, "strong": 0.0, "reviewer": 0.0,
            **(energy_profile_j or {}),
        }
        if any(value < 0 for value in self.energy_profile_j.values()):
            raise ValueError("energy profile values cannot be negative")
        self.energy_penalty_per_j = energy_penalty_per_j
        self.energy_used_j = 0.0
        by_index = {item.incident_index: item for item in outcomes if item.episode_id == episode.episode_id}
        if set(by_index) != set(range(len(episode.incidents))):
            raise ValueError(f"incomplete measured outcomes for {episode.episode_id}")
        measured: list[MeasuredIncidentOutcome] = []
        incidents: list[IncidentSpec] = []
        for index, incident in enumerate(episode.incidents):
            outcome = by_index[index]
            if outcome.incident_sha256 != incident_sha256(incident):
                raise ValueError(f"incident hash mismatch for {episode.episode_id}/{index}")
            weak = int(outcome.weak.parsed) if outcome.weak.valid else (incident.correct_action + 1) % 4
            strong = int(outcome.strong.parsed) if outcome.strong.valid else (incident.correct_action + 1) % 4
            reviewer = (
                tuple(bool(item) for item in outcome.reviewer.parsed)
                if outcome.reviewer.valid else (False,) * 4
            )
            incidents.append(replace(
                incident, weak_candidate=weak, strong_candidate=strong, reviewer_pass=reviewer,
            ))
            measured.append(outcome)
        self.measured_outcomes = tuple(measured)
        super().__init__(replace(episode, incidents=tuple(incidents)))

    def reset(self):
        self.energy_used_j = 0.0
        return super().reset()

    def action_cost(self, action: RLAction) -> tuple[int, float]:
        if self.incident is None:
            return ACTION_COSTS[action]
        outcome = self.measured_outcomes[self.incident_index]
        call = {
            RLAction.DELEGATE: outcome.weak,
            RLAction.ESCALATE: outcome.strong,
            RLAction.REVIEW: outcome.reviewer,
        }.get(action)
        if call is None:
            return ACTION_COSTS[action]
        return max(1, call.total_tokens), max(0.0, call.latency_s)

    def action_energy_j(self, action: RLAction) -> float:
        return {
            RLAction.DELEGATE: self.energy_profile_j["weak"],
            RLAction.ESCALATE: self.energy_profile_j["strong"],
            RLAction.REVIEW: self.energy_profile_j["reviewer"],
        }.get(action, 0.0)

    def step(self, action_value: int):
        action = RLAction(action_value)
        valid = bool(self.action_mask()[int(action)])
        action_energy = self.action_energy_j(action) if valid else 0.0
        observation, reward, terminated, truncated, _ = super().step(action_value)
        compute_cost = -self.energy_penalty_per_j * action_energy
        self.energy_used_j += action_energy
        row = self.trajectory[-1]
        row["action_energy_j"] = action_energy
        row["energy_used_j"] = self.energy_used_j
        row["reward"]["compute_cost"] = compute_cost
        row["reward"]["total"] += compute_cost
        return observation, reward + compute_cost, terminated, truncated, self.info()

    def info(self) -> dict[str, Any]:
        return {**super().info(), "energy_used_j": self.energy_used_j}


def group_outcomes(
    outcomes: Iterable[MeasuredIncidentOutcome],
) -> dict[str, list[MeasuredIncidentOutcome]]:
    grouped: dict[str, list[MeasuredIncidentOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.episode_id].append(outcome)
    return {key: sorted(value, key=lambda item: item.incident_index) for key, value in grouped.items()}


async def collect_command(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "outcomes.jsonl"
    model_files = {}
    for role in ("weak", "strong", "reviewer"):
        value = getattr(args, f"{role}_model_file")
        if value is not None:
            path = value.resolve()
            model_files[role] = {
                "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
    config = {
        "schema_version": MANIFEST_VERSION,
        "prompt_version": PROMPT_VERSION,
        "collector_source_sha256": sha256_file(Path(__file__)),
        "bank": str(args.bank.resolve()), "bank_sha256": sha256_file(args.bank.resolve()),
        "split": args.split,
        "max_episodes": args.max_episodes, "concurrency": args.concurrency,
        "endpoints": {
            role: {"endpoint": getattr(args, f"{role}_endpoint"), "model": getattr(args, f"{role}_model")}
            for role in ("weak", "strong", "reviewer")
        },
        "model_files": model_files,
        "prompt_contract": "role-specific manual; prompted JSON with strict post-validation; deterministic hidden environment reward",
        "decoding": "unconstrained JSON; llama.cpp json_schema constrained decoding failed semantic smoke and is retained as a negative artifact",
        "http_proxy_policy": "trust_env=false; localhost never traverses download proxy",
    }
    config_hash = _canonical_hash(config)
    config_path = output / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != config_hash:
            raise ValueError("existing collection config differs; use a new output directory")
    else:
        config_path.write_text(
            json.dumps({**config, "config_sha256": config_hash}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    episodes = read_episode_bank(args.bank.resolve(), split=args.split)[: args.max_episodes]
    completed: set[tuple[str, int]] = set()
    if rows_path.exists() and rows_path.stat().st_size:
        completed = {item.key for item in read_outcomes(rows_path)}
    pending = [
        (episode, index) for episode in episodes for index in range(len(episode.incidents))
        if (episode.episode_id, index) not in completed
    ]
    endpoints = {
        role: EndpointSpec(role, getattr(args, f"{role}_endpoint"), getattr(args, f"{role}_model"))
        for role in ("weak", "strong", "reviewer")
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    # Disabling environment proxies is mandatory because the download proxy also
    # captures localhost unless NO_PROXY was configured in the login shell.
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), trust_env=False) as client:
        with rows_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.batch_incidents):
                batch = pending[start : start + args.batch_incidents]
                rows = await asyncio.gather(*(
                    collect_incident(client, episode, index, endpoints, semaphore)
                    for episode, index in batch
                ))
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps({
                    "completed_new": min(start + len(batch), len(pending)),
                    "pending_new": len(pending), "elapsed_s": time.perf_counter() - started,
                }), flush=True)
    outcomes = read_outcomes(rows_path)
    relevant = [item for item in outcomes if item.episode_id in {episode.episode_id for episode in episodes}]
    summary = summarize_outcomes_with_episodes(relevant, episodes)
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "config_sha256": config_hash,
        "collection_seconds_this_invocation": time.perf_counter() - started,
        "episodes": len(episodes), "incidents": len(relevant),
        "files": {
            "config.json": {"bytes": config_path.stat().st_size, "sha256": sha256_file(config_path)},
            "outcomes.jsonl": {"bytes": rows_path.stat().st_size, "sha256": sha256_file(rows_path)},
        },
        "summary": summary,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def replay_command(args: argparse.Namespace) -> int:
    episodes = read_episode_bank(args.bank.resolve(), split=args.split)
    outcomes = read_outcomes(args.outcomes.resolve())
    grouped = group_outcomes(outcomes)
    episodes = [item for item in episodes if item.episode_id in grouped]
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    policies = {
        "A8-long-heuristic": a8_heuristic_policy,
        "weak-only": weak_only_policy,
        "strong-only": strong_only_policy,
        "oracle": oracle_policy,
    }
    for name, policy in policies.items():
        rows = []
        for episode in episodes:
            env = ModelBackedLongHorizonEnv(episode, grouped[episode.episode_id])
            observation, _ = env.reset()
            total_reward = 0.0
            while not env.terminated:
                action = int(policy(env, observation, env.action_mask()))
                observation, reward, _, _, _ = env.step(action)
                total_reward += reward
            rows.append({**env.info(), "return": total_reward, "trajectory": env.trajectory})
        rows_by_policy[name] = rows
    payload = {
        "schema_version": "multitown-live-model-replay-result-v1",
        "split": args.split, "episodes": len(episodes),
        "systems": {name: summarize_results(rows) for name, rows in rows_by_policy.items()},
        "action_counts": {
            name: dict(sorted(Counter(
                step["action"] for row in rows for step in row["trajectory"]
            ).items())) for name, rows in rows_by_policy.items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect resumable real-model outcomes")
    collect.add_argument("--bank", type=Path, required=True)
    collect.add_argument("--split", choices=("train", "dev"), required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--max-episodes", type=int, default=12)
    collect.add_argument("--concurrency", type=int, default=3)
    collect.add_argument("--batch-incidents", type=int, default=4)
    collect.add_argument("--timeout", type=float, default=180.0)
    for role, endpoint, model in (
        ("weak", "http://127.0.0.1:8001/v1", "qwen3.5-4b"),
        ("strong", "http://127.0.0.1:8002/v1", "qwen3.5-35b-a3b"),
        ("reviewer", "http://127.0.0.1:8001/v1", "qwen3.5-4b"),
    ):
        collect.add_argument(f"--{role}-endpoint", default=endpoint)
        collect.add_argument(f"--{role}-model", default=model)
        collect.add_argument(f"--{role}-model-file", type=Path)
    collect.set_defaults(func=lambda args: asyncio.run(collect_command(args)))
    replay = subparsers.add_parser("replay", help="evaluate fixed policies on measured outcomes")
    replay.add_argument("--bank", type=Path, required=True)
    replay.add_argument("--split", choices=("train", "dev"), required=True)
    replay.add_argument("--outcomes", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    replay.set_defaults(func=replay_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
