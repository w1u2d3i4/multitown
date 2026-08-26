"""Run an audited MAGRPO training job against local frozen writing data.

This adapter intentionally calls the official CoMLRL trainer and official
collaborative-writing prompts/rewards.  Checkpoints and run telemetry are
written to an external artifact directory, never into the source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from multitown.agentic_data import DEFAULT_DATA_ROOT

RUN_SCHEMA = "multitown-magrpo-run-v1"
DEFAULT_MODEL = Path(os.environ.get("MULTITOWN_AGENTIC_MODEL", "models/Qwen3-0.6B"))
DEFAULT_WRITING_ROOT = Path(
    os.environ.get("MULTITOWN_MAGRPO_WRITING_ROOT", "third_party/LLM_Collab_Writing")
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = (
        item
        for item in root.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(root).parts
    )
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "path_recorded_as": root.name,
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain=v1", "--untracked-files=no")),
    }


def _load_writing_module(writing_root: Path) -> ModuleType:
    entrypoint = writing_root / "train_magrpo.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"official writing entrypoint not found: {entrypoint}")
    sys.path.insert(0, str(writing_root))
    spec = importlib.util.spec_from_file_location(
        "multitown_upstream_magrpo_writing", entrypoint
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import official writing code: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    if dataset == "tldr":
        root = data_root / "magrpo/trl-lib__tldr/data"
        return (
            root / "train-00000-of-00001.parquet",
            root / "test-00000-of-00001.parquet",
        )
    if dataset == "arxiv":
        root = data_root / "magrpo/OpenMLRL__arXiv_abstract/data"
        return (
            root / "train-00000-of-00001.parquet",
            root / "val-00000-of-00001.parquet",
        )
    raise ValueError(f"unsupported MAGRPO writing dataset: {dataset}")


def _load_parquet_rows(path: Path, start: int, count: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("MAGRPO runner requires the reproduction extra") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pq.read_table(path).slice(start, count)
    if table.num_rows != count:
        raise ValueError(
            f"requested {count} rows at offset {start}, received {table.num_rows}: {path}"
        )
    return table.to_pylist()


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _gpu_sample() -> dict[str, Any]:
    query = (
        "name,driver_version,power.draw,temperature.gpu,utilization.gpu,"
        "clocks.current.sm"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        return {
            "gpu_name": values[0],
            "driver_version": values[1],
            "gpu_power_w": float(values[2]),
            "gpu_temperature_c": float(values[3]),
            "gpu_utilization_pct": float(values[4]),
            "gpu_sm_clock_mhz": float(values[5]),
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {}


class ResourceMonitor:
    def __init__(self, path: Path, interval_seconds: float = 1.0) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2))

    def _run(self) -> None:
        import psutil

        process = psutil.Process()
        with self.path.open("w", encoding="utf-8") as handle:
            while not self._stop.is_set():
                memory = psutil.virtual_memory()
                sample = {
                    "elapsed_seconds": time.monotonic() - self._started,
                    "process_rss_bytes": process.memory_info().rss,
                    "system_memory_used_bytes": memory.used,
                    "system_memory_available_bytes": memory.available,
                    **_gpu_sample(),
                }
                self.samples.append(sample)
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                handle.flush()
                self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0}

        def maximum(key: str) -> float | None:
            values = [float(row[key]) for row in self.samples if key in row]
            return max(values) if values else None

        powers = [
            float(row["gpu_power_w"]) for row in self.samples if "gpu_power_w" in row
        ]
        energy_wh = 0.0
        for previous, current in zip(self.samples, self.samples[1:]):
            if "gpu_power_w" not in previous or "gpu_power_w" not in current:
                continue
            seconds = current["elapsed_seconds"] - previous["elapsed_seconds"]
            energy_wh += (
                ((previous["gpu_power_w"] + current["gpu_power_w"]) / 2)
                * seconds
                / 3600
            )
        return {
            "sample_count": len(self.samples),
            "peak_process_rss_bytes": maximum("process_rss_bytes"),
            "peak_system_memory_used_bytes": maximum("system_memory_used_bytes"),
            "peak_gpu_power_w": maximum("gpu_power_w"),
            "mean_gpu_power_w": sum(powers) / len(powers) if powers else None,
            "estimated_gpu_energy_wh": energy_wh if powers else None,
            "peak_gpu_temperature_c": maximum("gpu_temperature_c"),
            "peak_gpu_utilization_pct": maximum("gpu_utilization_pct"),
        }


def _write_resource_plot(samples: Sequence[dict[str, Any]], output: Path) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed = [row["elapsed_seconds"] for row in samples]
    rss_gib = [row["process_rss_bytes"] / 2**30 for row in samples]
    power = [row.get("gpu_power_w", float("nan")) for row in samples]
    utilization = [row.get("gpu_utilization_pct", float("nan")) for row in samples]

    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(elapsed, rss_gib, color="#4c78a8")
    axes[0].set_ylabel("RSS (GiB)")
    axes[1].plot(elapsed, power, color="#f58518")
    axes[1].set_ylabel("GPU power (W)")
    axes[2].plot(elapsed, utilization, color="#54a24b")
    axes[2].set_ylabel("GPU util. (%)")
    axes[2].set_xlabel("Elapsed seconds")
    figure.suptitle("MAGRPO run resources")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _attach_optimizer_instrumentation(trainer: Any) -> list[dict[str, Any]]:
    instrumentation: list[dict[str, Any]] = [
        {"optimizer_steps": 0, "max_gradient_l2": 0.0} for _ in trainer.optimizers
    ]
    for agent_index, optimizer in enumerate(trainer.optimizers):
        original_step = optimizer.step

        def instrumented_step(
            *arguments: Any,
            _index: int = agent_index,
            _step: Any = original_step,
            **keywords: Any,
        ) -> Any:
            squared_norm = 0.0
            for group in trainer.optimizers[_index].param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        squared_norm += (
                            float(parameter.grad.detach().float().norm()) ** 2
                        )
            metrics = instrumentation[_index]
            metrics["optimizer_steps"] += 1
            metrics["max_gradient_l2"] = max(
                metrics["max_gradient_l2"], squared_norm**0.5
            )
            return _step(*arguments, **keywords)

        optimizer.step = instrumented_step
    return instrumentation


def _record_generation_metrics(trainer: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "generation_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reward_values": [],
    }
    original_generate = trainer._generate_completions
    original_rewards = trainer._compute_rewards

    def generate(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        value = original_generate(*arguments, **keywords)
        metrics["generation_calls"] += 1
        prompt_tokens = int(value["prompt_attention_mask"].sum().item())
        sequences = sum(len(batch) for batch in value["completion_input_ids"])
        metrics["prompt_tokens"] += prompt_tokens * sequences
        metrics["completion_tokens"] += sum(value["response_lens"])
        return value

    def rewards(*arguments: Any, **keywords: Any) -> list[float]:
        values = original_rewards(*arguments, **keywords)
        metrics["reward_values"].extend(float(value) for value in values)
        return values

    trainer._generate_completions = generate
    trainer._compute_rewards = rewards
    return metrics


def _safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tldr", "arxiv"), default="tldr")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--writing-root", type=Path, default=DEFAULT_WRITING_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for field in ("train_samples", "eval_samples", "epochs"):
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.generations < 2:
        raise ValueError("--generations must be at least two for group advantages")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(args.output / "monitor.jsonl", args.monitor_interval)
    started = _utc_now()
    monotonic_started = time.monotonic()
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "started_at": started,
        "method": "official-magrpo",
        "dataset": args.dataset,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"data_root", "model", "writing_root", "output"}
        },
    }
    monitor.start()
    try:
        import numpy as np
        import torch
        import transformers
        from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
        from transformers import AutoTokenizer

        train_path, eval_path = _dataset_paths(args.data_root, args.dataset)
        train_rows = _load_parquet_rows(
            train_path, args.train_start, args.train_samples
        )
        eval_rows = _load_parquet_rows(eval_path, args.eval_start, args.eval_samples)
        writing = _load_writing_module(args.writing_root)
        reward_module = importlib.import_module(f"rewards.{args.dataset}_rewards")
        reward_module.VERBOSE = False

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        trainer_args = MAGRPOConfig(
            num_turns=1,
            num_train_epochs=args.epochs,
            agent_learning_rate=args.learning_rate,
            logging_steps=1,
            num_generations=args.generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=None,
            num_agents=2,
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
        _seed_everything(args.seed)
        trainer = MAGRPOTrainer(
            agent_model=str(args.model),
            num_agents=2,
            tokenizer=tokenizer,
            model_config={
                "torch_dtype": torch.bfloat16,
                "model_kwargs": {"attn_implementation": "sdpa"},
            },
            train_dataset=train_rows,
            eval_dataset=eval_rows,
            dataset_type=args.dataset,
            reward_func=writing.make_reward_function(args.dataset),
            formatters=writing.get_formatters(args.dataset),
            args=trainer_args,
            wandb_config=None,
        )
        trainer.verbose = False
        optimizer_metrics = _attach_optimizer_instrumentation(trainer)
        generation_metrics = _record_generation_metrics(trainer)

        _seed_everything(args.seed + 1)
        before_eval = trainer.evaluate(num_eval_samples=args.eval_samples)
        _seed_everything(args.seed)
        trainer.train()
        _seed_everything(args.seed + 1)
        after_eval = trainer.evaluate(num_eval_samples=args.eval_samples)

        if args.save_model:
            trainer.save_model(args.output / "checkpoint")

        rewards = generation_metrics.pop("reward_values")
        result.update(
            {
                "status": "complete",
                "finished_at": _utc_now(),
                "wall_seconds": time.monotonic() - monotonic_started,
                "data": {
                    "train_path_recorded_as": train_path.name,
                    "train_sha256": _sha256(train_path),
                    "train_slice": [
                        args.train_start,
                        args.train_start + args.train_samples,
                    ],
                    "eval_path_recorded_as": eval_path.name,
                    "eval_sha256": _sha256(eval_path),
                    "eval_slice": [
                        args.eval_start,
                        args.eval_start + args.eval_samples,
                    ],
                },
                "model": {
                    "path_recorded_as": args.model.name,
                    "tree_sha256": _tree_digest(args.model),
                    "parameter_count_per_agent": sum(
                        parameter.numel()
                        for parameter in trainer.agents[0].parameters()
                    ),
                    "dtype": str(next(trainer.agents[0].parameters()).dtype),
                },
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
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                "training": {
                    "env_steps": trainer.env_step,
                    "optimizers": optimizer_metrics,
                    "nonzero_gradient_confirmed": all(
                        item["max_gradient_l2"] > 0 for item in optimizer_metrics
                    ),
                    "observed_reward_count": len(rewards),
                    "observed_reward_mean": _safe_mean(rewards),
                    "observed_reward_min": min(rewards) if rewards else None,
                    "observed_reward_max": max(rewards) if rewards else None,
                },
                "generation": generation_metrics,
                "evaluation": {"before": before_eval, "after": after_eval},
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
