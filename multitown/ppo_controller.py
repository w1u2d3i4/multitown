"""Train and evaluate an online PPO controller in the MultiTown long-horizon POMDP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .long_horizon_env import (
    ACTION_COUNT,
    ACTION_NAMES,
    MultiTownLongHorizonEnv,
    a8_heuristic_policy,
    oracle_policy,
    random_policy_factory,
    read_episode_bank,
    run_policy,
    strong_only_policy,
    summarize_results,
)
POLICY_VERSION = "multitown-a10-ppo-controller-v1"
EnvFactory = Callable[[Any], MultiTownLongHorizonEnv]


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 120
    episodes_per_update: int = 48
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 512
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    dev_interval: int = 10


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int, action_count: int):
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.action_count = action_count
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        with torch.no_grad():
            # Avoid a degenerate initial policy that terminates before observing.
            self.actor.bias[ACTION_NAMES.index("stop")] = -1.5
            self.actor.bias[ACTION_NAMES.index("human")] = -0.5

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


def bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, seed: int, iterations: int = 10_000,
) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired arrays must be non-empty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 512):
        width = min(512, iterations - start)
        indices = rng.integers(0, len(left), size=(width, len(left)))
        differences[start : start + width] = (left[indices] - right[indices]).mean(axis=1)
    return {
        "difference": float((left - right).mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "iterations": iterations,
    }


def exact_mcnemar_p(left: list[bool], right: list[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar/binomial test with an overflow-safe fraction."""

    if len(left) != len(right):
        raise ValueError("paired outcomes must have equal length")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
        exact_numerator = 1
        exact_denominator_power = 0
        log_p_value = 0.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
        denominator = 1 << discordant
        numerator = min(2 * tail, denominator)
        p_value = float(Fraction(numerator, denominator))
        if numerator == denominator:
            exact_numerator = 1
            exact_denominator_power = 0
            log_p_value = 0.0
        else:
            trailing_zero_bits = (numerator & -numerator).bit_length() - 1
            exact_numerator = numerator >> trailing_zero_bits
            exact_denominator_power = discordant - trailing_zero_bits
            log_p_value = math.log(numerator) - discordant * math.log(2.0)
    return {
        "statistic_version": "exact-binomial-two-sided-stable-fraction-v1",
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
        "log_p_value_two_sided": log_p_value,
        "p_value_exact": {
            "odd_numerator": str(exact_numerator),
            "denominator_power_of_two": exact_denominator_power,
        },
        "two_sided_definition": "double-smaller-binomial-tail-capped-at-one",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip())
    return revision, dirty


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    masked = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked)


def model_policy(model: ActorCritic, device: torch.device, *, deterministic: bool):
    def policy(_: MultiTownLongHorizonEnv, observation: np.ndarray, mask: np.ndarray) -> int:
        obs = torch.from_numpy(observation).to(device).unsqueeze(0)
        valid = torch.from_numpy(mask).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs)
            distribution = _masked_distribution(logits, valid)
            if deterministic:
                action = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).argmax(-1)
            else:
                action = distribution.sample()
        return int(action.item())

    return policy


def _episode_rollout(
    model: ActorCritic,
    episode: Any,
    device: torch.device,
    env_factory: EnvFactory = MultiTownLongHorizonEnv,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = env_factory(episode)
    observation, _ = env.reset()
    transitions: list[dict[str, Any]] = []
    total_return = 0.0
    while not env.terminated:
        mask = env.action_mask()
        obs_tensor = torch.from_numpy(observation).to(device).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).to(device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_tensor)
            distribution = _masked_distribution(logits, mask_tensor)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)
        next_observation, reward, done, _, _ = env.step(int(action.item()))
        transitions.append({
            "observation": observation,
            "mask": mask,
            "action": int(action.item()),
            "old_log_probability": float(log_probability.item()),
            "old_value": float(value.item()),
            "reward": float(reward),
            "done": done,
        })
        total_return += reward
        observation = next_observation
    return transitions, {**env.info(), "return": total_return}


def _advantages(
    episodes: list[list[dict[str, Any]]], config: PPOConfig,
) -> dict[str, np.ndarray]:
    flat: dict[str, list[Any]] = {
        "observation": [], "mask": [], "action": [], "old_log_probability": [],
        "old_value": [], "advantage": [], "return": [],
    }
    for episode in episodes:
        next_advantage = 0.0
        next_value = 0.0
        advantages = [0.0] * len(episode)
        returns = [0.0] * len(episode)
        for index in range(len(episode) - 1, -1, -1):
            transition = episode[index]
            delta = transition["reward"] + config.gamma * next_value - transition["old_value"]
            next_advantage = delta + config.gamma * config.gae_lambda * next_advantage
            advantages[index] = next_advantage
            returns[index] = next_advantage + transition["old_value"]
            next_value = transition["old_value"]
        for transition, advantage, target in zip(episode, advantages, returns, strict=True):
            for key in ("observation", "mask", "action", "old_log_probability", "old_value"):
                flat[key].append(transition[key])
            flat["advantage"].append(advantage)
            flat["return"].append(target)
    result = {key: np.asarray(value) for key, value in flat.items()}
    advantage = result["advantage"].astype(np.float32)
    result["advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    return result


def _ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    config: PPOConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    tensors = {
        "observation": torch.as_tensor(batch["observation"], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(batch["mask"], dtype=torch.bool, device=device),
        "action": torch.as_tensor(batch["action"], dtype=torch.long, device=device),
        "old_log_probability": torch.as_tensor(
            batch["old_log_probability"], dtype=torch.float32, device=device,
        ),
        "old_value": torch.as_tensor(batch["old_value"], dtype=torch.float32, device=device),
        "advantage": torch.as_tensor(batch["advantage"], dtype=torch.float32, device=device),
        "return": torch.as_tensor(batch["return"], dtype=torch.float32, device=device),
    }
    count = len(tensors["action"])
    metrics: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": [],
    }
    for _ in range(config.ppo_epochs):
        order = torch.randperm(count, generator=generator, device=device)
        for start in range(0, count, config.minibatch_size):
            indices = order[start : start + config.minibatch_size]
            logits, values = model(tensors["observation"][indices])
            distribution = _masked_distribution(logits, tensors["mask"][indices])
            new_log_probability = distribution.log_prob(tensors["action"][indices])
            log_ratio = new_log_probability - tensors["old_log_probability"][indices]
            ratio = log_ratio.exp()
            advantage = tensors["advantage"][indices]
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (values - tensors["return"][indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approx_kl"].append(float(((ratio - 1.0) - log_ratio).mean().item()))
                metrics["clip_fraction"].append(
                    float(((ratio - 1.0).abs() > config.clip_ratio).float().mean().item())
                )
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def evaluate_model(
    model: ActorCritic,
    episodes: list[Any],
    device: torch.device,
    *,
    include_trajectories: bool,
    env_factory: EnvFactory = MultiTownLongHorizonEnv,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = model_policy(model, device, deterministic=True)
    rows = [run_policy_with_env(episode, policy, env_factory) for episode in episodes]
    summary = summarize_results(rows)
    actions = Counter(
        step["action"] for row in rows for step in row["trajectory"]
    )
    summary["action_counts"] = dict(sorted(actions.items()))
    summary["routing_by_family"] = summarize_routing(episodes, rows)
    if not include_trajectories:
        rows = [{key: value for key, value in row.items() if key != "trajectory"} for row in rows]
    return summary, rows


def run_policy_with_env(
    episode: Any, policy: Any, env_factory: EnvFactory = MultiTownLongHorizonEnv,
) -> dict[str, Any]:
    env = env_factory(episode)
    observation, _ = env.reset()
    total_reward = 0.0
    while not env.terminated:
        action = int(policy(env, observation, env.action_mask()))
        observation, reward, _, _, _ = env.step(action)
        total_reward += reward
    return {**env.info(), "return": total_reward, "trajectory": env.trajectory}


def summarize_routing(episodes: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize controller actions by observable incident family."""
    episode_index = {str(episode.episode_id): episode for episode in episodes}
    result: dict[str, Any] = {
        str(family): {"incidents": 0, "action_counts": Counter()}
        for family in range(4)
    }
    for episode in episodes:
        for incident in episode.incidents:
            result[str(incident.family)]["incidents"] += 1
    for row in rows:
        episode = episode_index[str(row["episode_id"])]
        for step in row["trajectory"]:
            incident_index = int(step["incident_index"])
            if incident_index >= len(episode.incidents):
                continue
            family = str(episode.incidents[incident_index].family)
            result[family]["action_counts"][str(step["action"])] += 1
    normalized: dict[str, Any] = {}
    for family, values in result.items():
        incidents = int(values["incidents"])
        counts = dict(sorted(values["action_counts"].items()))
        normalized[family] = {
            "incidents": incidents,
            "action_counts": counts,
            "delegate_per_incident": counts.get("delegate", 0) / incidents,
            "escalate_per_incident": counts.get("escalate", 0) / incidents,
        }
    return normalized


def _save_checkpoint(
    path: Path, model: ActorCritic, config: PPOConfig, *, seed: int, update: int,
    policy_version: str = POLICY_VERSION,
) -> None:
    torch.save({
        "policy_version": policy_version,
        "observation_size": model.observation_size,
        "action_count": model.action_count,
        "hidden_size": model.hidden_size,
        "seed": seed,
        "update": update,
        "ppo_config": asdict(config),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }, path)


def load_checkpoint(
    path: Path, device: torch.device, *, expected_policy_version: str = POLICY_VERSION,
) -> tuple[ActorCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("policy_version") != expected_policy_version:
        raise ValueError(f"unsupported policy: {checkpoint.get('policy_version')}")
    model = ActorCritic(
        int(checkpoint["observation_size"]),
        int(checkpoint["hidden_size"]),
        int(checkpoint["action_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _selection_score(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["episode_success_rate"]),
        float(summary["subgoal_completion_rate"]),
        -float(summary["tokens_per_success"]),
        -float(summary["human_rate"]),
    )


def _plot_training(seed_logs: dict[int, list[dict[str, Any]]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for seed, rows in seed_logs.items():
        updates = [row["update"] for row in rows]
        axes[0, 0].plot(updates, [row["rollout_episode_success_rate"] for row in rows], label=str(seed))
        axes[0, 1].plot(updates, [row["rollout_mean_return"] for row in rows], label=str(seed))
        axes[1, 0].plot(updates, [row["rollout_tokens_per_episode"] for row in rows], label=str(seed))
        axes[1, 1].plot(updates, [row["entropy"] for row in rows], label=str(seed))
        dev_rows = [row for row in rows if "dev_episode_success_rate" in row]
        axes[0, 0].scatter(
            [row["update"] for row in dev_rows],
            [row["dev_episode_success_rate"] for row in dev_rows], s=16,
        )
    axes[0, 0].set_title("Episode success (rollout; dots=dev)")
    axes[0, 1].set_title("Mean return")
    axes[1, 0].set_title("Tokens per episode")
    axes[1, 1].set_title("Policy entropy")
    for axis in axes.flat:
        axis.set_xlabel("PPO update")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(title="seed", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    revision, dirty = _git_state(root)
    bank_root = args.bank_root.resolve()
    manifest_path = bank_root / "manifest.json"
    bank_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = bank_root / "train.jsonl"
    dev_path = bank_root / "dev.jsonl"
    train_episodes = read_episode_bank(train_path, split="train")
    dev_episodes = read_episode_bank(dev_path, split="dev")
    config = PPOConfig(
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        dev_interval=args.dev_interval,
    )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    started = time.perf_counter()
    all_logs: dict[int, list[dict[str, Any]]] = {}
    seed_results: list[dict[str, Any]] = []

    for seed in args.seeds:
        _set_seed(seed)
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir()
        model = ActorCritic(
            MultiTownLongHorizonEnv.observation_size, config.hidden_size, ACTION_COUNT,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        sample_rng = random.Random(seed)
        tensor_generator = torch.Generator(device=device)
        tensor_generator.manual_seed(seed)
        logs: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        best_path = seed_dir / "best.pt"

        for update in range(1, config.updates + 1):
            rollout_transitions: list[list[dict[str, Any]]] = []
            rollout_rows: list[dict[str, Any]] = []
            for _ in range(config.episodes_per_update):
                episode = train_episodes[sample_rng.randrange(len(train_episodes))]
                transitions, metrics = _episode_rollout(model, episode, device)
                rollout_transitions.append(transitions)
                rollout_rows.append(metrics)
            batch = _advantages(rollout_transitions, config)
            update_metrics = _ppo_update(
                model, optimizer, batch, config, device, tensor_generator,
            )
            rollout_summary = summarize_results([
                {**row, "trajectory": []} for row in rollout_rows
            ])
            log_row = {
                "update": update,
                "environment_steps": int(sum(len(item) for item in rollout_transitions)),
                "rollout_episode_success_rate": rollout_summary["episode_success_rate"],
                "rollout_subgoal_completion_rate": rollout_summary["subgoal_completion_rate"],
                "rollout_mean_return": rollout_summary["mean_return"],
                "rollout_tokens_per_episode": rollout_summary["tokens_per_episode"],
                **update_metrics,
            }
            if update % config.dev_interval == 0 or update == config.updates:
                model.eval()
                dev_summary, _ = evaluate_model(
                    model, dev_episodes, device, include_trajectories=False,
                )
                model.train()
                log_row.update({f"dev_{key}": value for key, value in dev_summary.items()})
                if best is None or _selection_score(dev_summary) > _selection_score(best["summary"]):
                    _save_checkpoint(best_path, model, config, seed=seed, update=update)
                    best = {"update": update, "summary": dev_summary}
            logs.append(log_row)
        if best is None:
            raise RuntimeError("no development checkpoint selected")
        _write_jsonl(seed_dir / "training-metrics.jsonl", logs)
        _write_json(seed_dir / "best-dev.json", best)
        all_logs[seed] = logs
        seed_results.append({
            "seed": seed,
            "checkpoint": str(best_path.relative_to(output)),
            "checkpoint_sha256": _sha256(best_path),
            "best_update": best["update"],
            "dev": best["summary"],
        })

    selected = max(seed_results, key=lambda item: _selection_score(item["dev"]))
    selected_checkpoint = output / selected["checkpoint"]
    _plot_training(all_logs, output / "training-curves.png")
    elapsed = time.perf_counter() - started
    freeze = {
        "schema_version": "multitown-a10-ppo-pretest-freeze-v1",
        "policy_version": POLICY_VERSION,
        "source_revision": revision,
        "source_dirty_at_start": dirty,
        "algorithm": "online on-policy PPO with masked discrete controller actions",
        "policy_scope": "central organization controller; fixed workers/tools/world",
        "device": str(device),
        "torch_version": torch.__version__,
        "training_seconds": elapsed,
        "ppo_config": asdict(config),
        "seeds": args.seeds,
        "action_contract": list(ACTION_NAMES),
        "reward_contract": {
            "environment": "LongHorizonReward in multitown.long_horizon_env",
            "main_success": "deterministic episode/subgoal state",
            "llm_as_judge": False,
        },
        "split_protocol": bank_manifest["split_protocol"],
        "input_hashes": {
            "bank_manifest": _sha256(manifest_path),
            "train": _sha256(train_path),
            "dev": _sha256(dev_path),
            "expected_test": bank_manifest["files"]["test.jsonl"]["sha256"],
            "environment_source": _sha256(Path(__file__).with_name("long_horizon_env.py")),
            "trainer_source": _sha256(Path(__file__)),
        },
        "test_episodes_accessed": 0,
        "seed_results": seed_results,
        "selected": selected,
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
    }
    _write_json(output / "pretest-freeze.json", freeze)
    print(json.dumps({
        "output": str(output), "selected": selected, "training_seconds": elapsed,
    }, ensure_ascii=False, indent=2))
    return 0


def _evaluate_baseline(name: str, episodes: list[Any], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = {
        "A8-long-heuristic": a8_heuristic_policy,
        "strong-only": strong_only_policy,
        "oracle": oracle_policy,
        "random": random_policy_factory(seed),
    }[name]
    rows = [run_policy(episode, policy) for episode in episodes]
    summary = summarize_results(rows)
    summary["action_counts"] = dict(sorted(Counter(
        step["action"] for row in rows for step in row["trajectory"]
    ).items()))
    summary["routing_by_family"] = summarize_routing(episodes, rows)
    return summary, rows


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    left_index = {str(row["episode_id"]): row for row in left}
    right_index = {str(row["episode_id"]): row for row in right}
    if set(left_index) != set(right_index):
        raise ValueError("paired episode IDs differ")
    ids = sorted(left_index)
    left_success = np.asarray([float(left_index[key]["episode_success"]) for key in ids])
    right_success = np.asarray([float(right_index[key]["episode_success"]) for key in ids])
    left_tokens = np.asarray([float(left_index[key]["tokens_used"]) for key in ids])
    right_tokens = np.asarray([float(right_index[key]["tokens_used"]) for key in ids])
    left_return = np.asarray([float(left_index[key]["return"]) for key in ids])
    right_return = np.asarray([float(right_index[key]["return"]) for key in ids])
    success = bootstrap_difference(left_success, right_success, seed=seed)
    tokens = bootstrap_difference(left_tokens, right_tokens, seed=seed + 1)
    returns = bootstrap_difference(left_return, right_return, seed=seed + 2)
    result = {
        "episode_success": success,
        "tokens_per_episode": tokens,
        "return": returns,
        "mcnemar_exact": exact_mcnemar_p(
            [bool(value) for value in left_success], [bool(value) for value in right_success],
        ),
        "token_reduction_fraction": 1.0 - float(left_tokens.mean() / right_tokens.mean()),
        "success_noninferior_at_minus_1pp": success["ci95_low"] >= -0.01,
    }
    if all("energy_used_j" in row for row in (*left_index.values(), *right_index.values())):
        left_energy = np.asarray([float(left_index[key]["energy_used_j"]) for key in ids])
        right_energy = np.asarray([float(right_index[key]["energy_used_j"]) for key in ids])
        result["energy_per_episode_j"] = bootstrap_difference(
            left_energy, right_energy, seed=seed + 3,
        )
        result["energy_reduction_fraction"] = 1.0 - float(
            left_energy.mean() / right_energy.mean()
        ) if right_energy.mean() > 0 else None
    return result


def evaluate(args: argparse.Namespace) -> int:
    experiment = args.experiment_dir.resolve()
    freeze_path = experiment / "pretest-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("test_episodes_accessed") != 0:
        raise ValueError("pretest freeze does not certify zero prior test access")
    test_path = args.test_bank.resolve()
    if _sha256(test_path) != freeze["input_hashes"]["expected_test"]:
        raise ValueError("held-out test hash does not match pretest freeze")
    output = experiment / "held-out-evaluation"
    if output.exists():
        raise FileExistsError("held-out evaluation already exists; refusing a second test read")
    output.mkdir()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = experiment / freeze["selected"]["checkpoint"]
    if _sha256(checkpoint_path) != freeze["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    episodes = read_episode_bank(test_path, split="test")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    ppo_summary, ppo_rows = evaluate_model(
        model, episodes, device, include_trajectories=True,
    )
    systems: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        "A10-PPO": (ppo_summary, ppo_rows),
    }
    for index, name in enumerate(("A8-long-heuristic", "strong-only", "random", "oracle")):
        systems[name] = _evaluate_baseline(name, episodes, args.seed + index)
    for name, (_, rows) in systems.items():
        _write_jsonl(output / f"{name}.jsonl", rows)
    comparisons = {
        name: _paired(ppo_rows, rows, seed=args.seed + 100 * index)
        for index, (name, (_, rows)) in enumerate(systems.items()) if name != "A10-PPO"
    }
    overall = {name: summary for name, (summary, _) in systems.items()}
    versus_a8 = comparisons["A8-long-heuristic"]
    versus_strong = comparisons["strong-only"]
    result = {
        "schema_version": "multitown-a10-ppo-held-out-result-v1",
        "policy_version": POLICY_VERSION,
        "evaluation_scope": "central-controller online PPO in deterministic long-horizon POMDP",
        "test_split": "OOD family x failure-mode combinations",
        "test_episodes": len(episodes),
        "selected_training_seed": checkpoint["seed"],
        "selected_update": checkpoint["update"],
        "overall": overall,
        "paired_comparisons": comparisons,
        "claim_gates": {
            "beats_a8_success_ci": versus_a8["episode_success"]["ci95_low"] > 0,
            "a8_noninferior_and_20pct_token_reduction": (
                versus_a8["success_noninferior_at_minus_1pp"]
                and versus_a8["token_reduction_fraction"] >= 0.20
            ),
            "beats_equal_budget_strong_success_ci": versus_strong["episode_success"]["ci95_low"] > 0,
            "may_claim_llm_weight_rl": False,
        },
        "limitations": [
            "PPO trains a lightweight central controller, not Qwen model weights.",
            "Workers, reviewer outcomes and tool failures are simulator components.",
            "This establishes Agentic RL mechanics in MultiTown, not real-world generalization.",
        ],
    }
    _write_json(output / "result.json", result)
    manifest = {
        "schema_version": "multitown-a10-ppo-evaluation-manifest-v1",
        "pretest_freeze_sha256": _sha256(freeze_path),
        "test_bank_sha256": _sha256(test_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train on train and select on dev only")
    train_parser.add_argument("--bank-root", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--seeds", type=int, nargs="+", default=[20260812, 20260813, 20260814])
    train_parser.add_argument("--updates", type=int, default=120)
    train_parser.add_argument("--episodes-per-update", type=int, default=48)
    train_parser.add_argument("--hidden-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--dev-interval", type=int, default=10)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(func=train)
    eval_parser = subparsers.add_parser("evaluate", help="evaluate a frozen policy once on held-out test")
    eval_parser.add_argument("--experiment-dir", type=Path, required=True)
    eval_parser.add_argument("--test-bank", type=Path, required=True)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--seed", type=int, default=20260812)
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
