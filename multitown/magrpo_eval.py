"""Evaluate official MAGRPO/GRPO writing policies on a frozen local slice."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from multitown.agentic_data import DEFAULT_DATA_ROOT
from multitown.magrpo_prompts import budgeted_formatters
from multitown.magrpo_runner import (
    DEFAULT_MODEL,
    DEFAULT_WRITING_ROOT,
    ResourceMonitor,
    _dataset_paths,
    _git_provenance,
    _grpo_reward_function,
    _load_parquet_rows,
    _load_writing_module,
    _record_generation_metrics,
    _seed_everything,
    _sha256,
    _tree_digest,
    _utc_now,
    _write_resource_plot,
)

EVAL_SCHEMA = "multitown-magrpo-evaluation-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("magrpo", "grpo"), required=True)
    parser.add_argument("--dataset", choices=("tldr", "arxiv"), default="tldr")
    parser.add_argument(
        "--prompt-profile", choices=("official", "budgeted"), default="official"
    )
    parser.add_argument(
        "--grpo-split-policy",
        choices=("official-fallback", "strict-delimiter"),
        default="official-fallback",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models", type=Path, nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--writing-root", type=Path, default=DEFAULT_WRITING_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260830])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    expected_models = 2 if args.method == "magrpo" and len(args.models) > 1 else 1
    if args.method == "grpo" and len(args.models) != 1:
        raise ValueError("GRPO evaluation requires exactly one model")
    if args.method == "grpo" and args.prompt_profile != "official":
        raise ValueError("the budgeted prompt profile is a two-agent MAGRPO profile")
    if args.method == "magrpo" and args.grpo_split_policy != "official-fallback":
        raise ValueError("GRPO split policy does not apply to MAGRPO")
    if args.method == "magrpo" and len(args.models) not in {1, 2}:
        raise ValueError("MAGRPO evaluation requires one shared or two agent models")
    if len(args.models) != expected_models:
        raise ValueError("unexpected model count")
    if args.eval_samples < 1 or args.max_new_tokens < 1:
        raise ValueError("evaluation samples and output tokens must be positive")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("evaluation seeds must be non-empty and unique")
    for model in args.models:
        if not model.is_dir():
            raise FileNotFoundError(model)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")


def _make_trainer(args: argparse.Namespace, rows: list[dict[str, Any]]) -> Any:
    import torch
    from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
    from transformers import AutoTokenizer

    writing = _load_writing_module(args.writing_root, args.method)
    reward_module = importlib.import_module(f"rewards.{args.dataset}_rewards")
    reward_module.VERBOSE = False
    num_agents = 2 if args.method == "magrpo" else 1
    tokenizers = []
    tokenizer_sources = args.models if len(args.models) > 1 else [args.models[0]]
    for source in tokenizer_sources:
        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizers.append(tokenizer)

    trainer_args = MAGRPOConfig(
        num_turns=1,
        num_train_epochs=1,
        agent_learning_rate=5e-6,
        logging_steps=1,
        num_generations=2,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=None,
        num_agents=num_agents,
        parallel_training="none",
        agent_devices=["cuda:0"],
        early_termination_threshold=None,
        rollout_buffer_size=1,
        train_batch_size=1,
        advantage_normalization=True,
        eval_interval=0,
        eval_num_samples=args.eval_samples,
        eval_batch_size=1,
        reference_kl_enabled=False,
    )
    if args.method == "magrpo":
        reward_function = writing.make_reward_function(args.dataset)
        formatters = (
            writing.get_formatters(args.dataset)
            if args.prompt_profile == "official"
            else budgeted_formatters(args.dataset)
        )
    else:
        reward_function = _grpo_reward_function(
            writing, args.dataset, args.grpo_split_policy
        )
        formatters = writing.get_formatter(args.dataset)

    common = {
        "num_agents": num_agents,
        "model_config": {
            "torch_dtype": torch.bfloat16,
            "model_kwargs": {"attn_implementation": "sdpa"},
        },
        "eval_dataset": rows,
        "dataset_type": args.dataset,
        "reward_func": reward_function,
        "formatters": formatters,
        "args": trainer_args,
        "wandb_config": None,
    }
    if len(args.models) == 1:
        common.update({"agent_model": str(args.models[0]), "tokenizer": tokenizers[0]})
    else:
        common.update(
            {
                "agents": [str(model) for model in args.models],
                "tokenizer": tokenizers,
            }
        )
    trainer = MAGRPOTrainer(**common)
    trainer.verbose = False
    return trainer


def _aggregate(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [value for record in seed_records for value in record["rewards"]]
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / max(1, len(rewards) - 1)
    return {
        "observations": len(rewards),
        "reward_mean": mean,
        "reward_std": variance**0.5,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "per_seed_reward_mean": {
            str(record["seed"]): sum(record["rewards"]) / len(record["rewards"])
            for record in seed_records
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(args.output / "monitor.jsonl", args.monitor_interval)
    started = _utc_now()
    monotonic_started = time.monotonic()
    result: dict[str, Any] = {
        "schema": EVAL_SCHEMA,
        "status": "running",
        "started_at": started,
        "method": args.method,
        "dataset": args.dataset,
    }
    monitor.start()
    try:
        import numpy as np
        import torch
        import transformers

        _, eval_path = _dataset_paths(args.data_root, args.dataset)
        rows = _load_parquet_rows(eval_path, args.eval_start, args.eval_samples)
        trainer = _make_trainer(args, rows)
        generation = _record_generation_metrics(trainer)
        seed_records = []
        for seed in args.seeds:
            reward_start = len(generation["reward_values"])
            _seed_everything(seed)
            metrics = trainer.evaluate(num_eval_samples=args.eval_samples)
            rewards = generation["reward_values"][reward_start:]
            if len(rewards) != args.eval_samples:
                raise RuntimeError(
                    f"expected {args.eval_samples} rewards, observed {len(rewards)}"
                )
            seed_records.append({"seed": seed, "metrics": metrics, "rewards": rewards})
        generation.pop("reward_values")
        result.update(
            {
                "status": "complete",
                "finished_at": _utc_now(),
                "wall_seconds": time.monotonic() - monotonic_started,
                "protocol": {
                    "eval_start": args.eval_start,
                    "eval_samples": args.eval_samples,
                    "seeds": args.seeds,
                    "max_new_tokens_per_agent": args.max_new_tokens,
                    "num_agents": 2 if args.method == "magrpo" else 1,
                    "maximum_total_output_tokens_per_task": (
                        args.max_new_tokens * (2 if args.method == "magrpo" else 1)
                    ),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "prompt_profile": args.prompt_profile,
                    "grpo_split_policy": args.grpo_split_policy,
                },
                "data": {
                    "path_recorded_as": eval_path.name,
                    "sha256": _sha256(eval_path),
                },
                "models": [
                    {
                        "path_recorded_as": model.name,
                        "tree_sha256": _tree_digest(model),
                    }
                    for model in args.models
                ],
                "upstream": {
                    "comlrl": _git_provenance(
                        Path(importlib.import_module("comlrl").__file__).parent.parent
                    ),
                    "collaborative_writing": _git_provenance(args.writing_root),
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "numpy": np.__version__,
                    "cuda_device": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                "generation": generation,
                "seed_records": seed_records,
                "aggregate": _aggregate(seed_records),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "wall_seconds": time.monotonic() - monotonic_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        monitor.stop()
        result["resources"] = monitor.summary()
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        _write_resource_plot(monitor.samples, args.output / "resource_curve.png")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
