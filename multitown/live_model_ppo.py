"""Train a development-only PPO controller on cached real-model outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .live_model_env import (
    ModelBackedLongHorizonEnv,
    group_outcomes,
    read_outcomes,
)
from .long_horizon_env import (
    ACTION_COUNT,
    MultiTownLongHorizonEnv,
    a8_heuristic_policy,
    oracle_policy,
    random_policy_factory,
    read_episode_bank,
    strong_only_policy,
    summarize_results,
    weak_only_policy,
)
from .ppo_controller import (
    ActorCritic,
    PPOConfig,
    _advantages,
    _episode_rollout,
    _paired,
    _plot_training,
    _ppo_update,
    _save_checkpoint,
    _set_seed,
    evaluate_model,
    run_policy_with_env,
)


POLICY_VERSION = "multitown-a11-live-model-ppo-controller-v1"


def live_selection_score(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["episode_success_rate"]),
        float(summary["subgoal_completion_rate"]),
        -float(summary["energy_per_episode_j"]),
        -float(summary["tokens_per_success"]),
        -float(summary["human_rate"]),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip())
    return revision, dirty


def load_live_split(
    bank: Path, outcomes_path: Path, split: str,
    *, energy_profile_j: dict[str, float] | None = None, energy_penalty_per_j: float = 0.0,
):
    manifest_path = outcomes_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing collection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {}).get("outcomes.jsonl", {}).get("sha256")
    actual = sha256_file(outcomes_path)
    if expected != actual:
        raise ValueError(f"outcome hash does not match collection manifest: {outcomes_path}")
    episodes = read_episode_bank(bank, split=split)
    grouped = group_outcomes(read_outcomes(outcomes_path))
    selected = [episode for episode in episodes if episode.episode_id in grouped]
    if not selected:
        raise ValueError(f"no {split} episodes have live outcomes")

    def factory(episode):
        return ModelBackedLongHorizonEnv(
            episode, grouped[episode.episode_id], energy_profile_j=energy_profile_j,
            energy_penalty_per_j=energy_penalty_per_j,
        )

    # Eagerly verify completeness and incident hashes before training.
    for episode in selected:
        factory(episode)
    return selected, factory


def train_family_router(episodes: list[Any], factory):
    role_correct = {family: {"weak": 0, "strong": 0} for family in range(4)}
    role_count = {family: 0 for family in range(4)}
    for episode in episodes:
        env = factory(episode)
        for index, incident in enumerate(env.episode.incidents):
            measured = env.measured_outcomes[index]
            role_count[incident.family] += 1
            role_correct[incident.family]["weak"] += int(
                measured.weak.valid and measured.weak.parsed == incident.correct_action
            )
            role_correct[incident.family]["strong"] += int(
                measured.strong.valid and measured.strong.parsed == incident.correct_action
            )
    missing = [family for family, count in role_count.items() if count == 0]
    if missing:
        raise ValueError(f"train-family router is missing families: {missing}")
    selected = {
        family: max(
            ("weak", "strong"),
            key=lambda role: (
                role_correct[family][role] / role_count[family],
                role == "weak",
            ),
        )
        for family in range(4)
    }
    return selected, {
        str(family): {
            "selected": selected[family], "incidents": role_count[family],
            "weak_accuracy": role_correct[family]["weak"] / role_count[family],
            "strong_accuracy": role_correct[family]["strong"] / role_count[family],
        }
        for family in range(4)
    }


def family_router_policy(selected: dict[int, str]):
    def policy(env, observation, mask):
        del observation
        role = selected[env.incident.family] if env.incident is not None else "weak"
        return (weak_only_policy if role == "weak" else strong_only_policy)(env, None, mask)

    return policy


def evaluate_fixed(
    name: str, episodes: list[Any], factory, seed: int,
    *, family_selection: dict[int, str] | None = None,
):
    policy = {
        "A8-live-heuristic": a8_heuristic_policy,
        "weak-only": weak_only_policy,
        "strong-only": strong_only_policy,
        "random": random_policy_factory(seed),
        "oracle": oracle_policy,
    }.get(name)
    if name == "train-family-router":
        if family_selection is None:
            raise ValueError("train-family-router requires frozen family selection")
        policy = family_router_policy(family_selection)
    if policy is None:
        raise ValueError(name)
    rows = [run_policy_with_env(episode, policy, factory) for episode in episodes]
    summary = summarize_results(rows)
    summary["action_counts"] = dict(sorted(Counter(
        step["action"] for row in rows for step in row["trajectory"]
    ).items()))
    return summary, rows


def train(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(root)
    energy_profile_j = {
        "weak": args.weak_energy_j, "strong": args.strong_energy_j,
        "reviewer": args.reviewer_energy_j,
    }
    train_episodes, train_factory = load_live_split(
        args.train_bank.resolve(), args.train_outcomes.resolve(), "train",
        energy_profile_j=energy_profile_j, energy_penalty_per_j=args.energy_penalty_per_j,
    )
    dev_episodes, dev_factory = load_live_split(
        args.dev_bank.resolve(), args.dev_outcomes.resolve(), "dev",
        energy_profile_j=energy_profile_j, energy_penalty_per_j=args.energy_penalty_per_j,
    )
    family_selection, family_router_fit = train_family_router(train_episodes, train_factory)
    config = PPOConfig(
        updates=args.updates, episodes_per_update=args.episodes_per_update,
        hidden_size=args.hidden_size, learning_rate=args.learning_rate,
        dev_interval=args.dev_interval,
    )
    device = torch.device(args.device)
    started = time.perf_counter()
    seed_results: list[dict[str, Any]] = []
    all_logs: dict[int, list[dict[str, Any]]] = {}

    for seed in args.seeds:
        _set_seed(seed)
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir()
        model = ActorCritic(
            MultiTownLongHorizonEnv.observation_size, config.hidden_size, ACTION_COUNT,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        sample_rng = random.Random(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        logs: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        best_path = seed_dir / "best.pt"
        for update in range(1, config.updates + 1):
            transition_batches = []
            rollout_rows = []
            for _ in range(config.episodes_per_update):
                episode = train_episodes[sample_rng.randrange(len(train_episodes))]
                transitions, metrics = _episode_rollout(
                    model, episode, device, env_factory=train_factory,
                )
                transition_batches.append(transitions)
                rollout_rows.append(metrics)
            batch = _advantages(transition_batches, config)
            update_metrics = _ppo_update(model, optimizer, batch, config, device, generator)
            rollout = summarize_results([{**row, "trajectory": []} for row in rollout_rows])
            log = {
                "update": update,
                "environment_steps": sum(len(item) for item in transition_batches),
                "rollout_episode_success_rate": rollout["episode_success_rate"],
                "rollout_subgoal_completion_rate": rollout["subgoal_completion_rate"],
                "rollout_mean_return": rollout["mean_return"],
                "rollout_tokens_per_episode": rollout["tokens_per_episode"],
                **update_metrics,
            }
            if update % config.dev_interval == 0 or update == config.updates:
                model.eval()
                dev_summary, _ = evaluate_model(
                    model, dev_episodes, device, include_trajectories=False,
                    env_factory=dev_factory,
                )
                model.train()
                log.update({f"dev_{key}": value for key, value in dev_summary.items()})
                if best is None or live_selection_score(dev_summary) > live_selection_score(best["summary"]):
                    _save_checkpoint(
                        best_path, model, config, seed=seed, update=update,
                        policy_version=POLICY_VERSION,
                    )
                    best = {"update": update, "summary": dev_summary}
            logs.append(log)
        if best is None:
            raise RuntimeError("no development checkpoint selected")
        write_jsonl(seed_dir / "training-metrics.jsonl", logs)
        write_json(seed_dir / "best-dev.json", best)
        all_logs[seed] = logs
        seed_results.append({
            "seed": seed, "checkpoint": str(best_path.relative_to(output)),
            "checkpoint_sha256": sha256_file(best_path),
            "best_update": best["update"], "dev": best["summary"],
        })

    selected = max(seed_results, key=lambda item: live_selection_score(item["dev"]))
    selected_path = output / selected["checkpoint"]
    checkpoint = torch.load(selected_path, map_location=device, weights_only=False)
    if checkpoint.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"unexpected selected policy version: {checkpoint.get('policy_version')}")
    model = ActorCritic(
        int(checkpoint["observation_size"]), int(checkpoint["hidden_size"]),
        int(checkpoint["action_count"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ppo_summary, ppo_rows = evaluate_model(
        model, dev_episodes, device, include_trajectories=True, env_factory=dev_factory,
    )
    systems: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        "A11-live-PPO": (ppo_summary, ppo_rows),
    }
    for index, name in enumerate((
        "A8-live-heuristic", "weak-only", "strong-only", "train-family-router", "random", "oracle",
    )):
        systems[name] = evaluate_fixed(
            name, dev_episodes, dev_factory, args.seeds[0] + index,
            family_selection=family_selection,
        )
    evaluation_dir = output / "dev-evaluation"
    evaluation_dir.mkdir()
    for name, (_, rows) in systems.items():
        write_jsonl(evaluation_dir / f"{name}.jsonl", rows)
    comparisons = {
        name: _paired(ppo_rows, rows, seed=args.seeds[0] + index * 100)
        for index, (name, (_, rows)) in enumerate(systems.items()) if name != "A11-live-PPO"
    }
    _plot_training(all_logs, output / "training-curves.png")
    result = {
        "schema_version": "multitown-a11-live-model-dev-result-v1",
        "policy_version": POLICY_VERSION,
        "evaluation_status": "development-only; no held-out test was read",
        "source_revision": revision, "source_dirty_at_start": dirty,
        "algorithm": "online PPO over a replayable environment backed by cached real Qwen calls",
        "policy_scope": "central controller only; Qwen weights fixed",
        "training_seconds": time.perf_counter() - started,
        "train_episodes": len(train_episodes), "dev_episodes": len(dev_episodes),
        "ppo_config": asdict(config), "seed_results": seed_results,
        "energy_cost_contract": {
            "profile_j_per_call": energy_profile_j,
            "reward_penalty_per_j": args.energy_penalty_per_j,
            "source": args.energy_profile_source,
        },
        "train_family_router_fit": family_router_fit,
        "selected": selected,
        "systems": {name: summary for name, (summary, _) in systems.items()},
        "paired_comparisons": comparisons,
        "input_sha256": {
            "train_bank": sha256_file(args.train_bank.resolve()),
            "dev_bank": sha256_file(args.dev_bank.resolve()),
            "train_outcomes": sha256_file(args.train_outcomes.resolve()),
            "dev_outcomes": sha256_file(args.dev_outcomes.resolve()),
            "train_outcomes_manifest": sha256_file(args.train_outcomes.resolve().with_name("manifest.json")),
            "dev_outcomes_manifest": sha256_file(args.dev_outcomes.resolve().with_name("manifest.json")),
        },
        "claim_gates": {
            "uses_real_model_outputs": True,
            "may_claim_model_backed_replay_rl": True,
            "may_claim_held_out_improvement": False,
            "may_claim_llm_weight_rl": False,
        },
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--dev-bank", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--dev-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260812, 20260813, 20260814])
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dev-interval", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--weak-energy-j", type=float, default=0.0)
    parser.add_argument("--strong-energy-j", type=float, default=0.0)
    parser.add_argument("--reviewer-energy-j", type=float, default=0.0)
    parser.add_argument("--energy-penalty-per-j", type=float, default=0.0)
    parser.add_argument("--energy-profile-source", default="none")
    return parser


def main() -> None:
    raise SystemExit(train(build_parser().parse_args()))


if __name__ == "__main__":
    main()
