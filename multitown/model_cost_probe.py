"""Measure local model request latency, tokens, power, and residency cost."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from .live_model_env import EndpointSpec, call_role
from .long_horizon_env import read_episode_bank


SCHEMA_VERSION = "multitown-model-cost-probe-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrate_power(
    samples: list[dict[str, float]], start_s: float, end_s: float, baseline_w: float,
) -> dict[str, float]:
    """Piecewise-constant integration over a clipped time interval."""
    if end_s <= start_s:
        raise ValueError("end_s must be after start_s")
    ordered = sorted(samples, key=lambda item: item["time_s"])
    if not ordered:
        raise ValueError("power samples cannot be empty")
    total_j = 0.0
    incremental_j = 0.0
    for index, sample in enumerate(ordered):
        left = max(start_s, sample["time_s"])
        next_time = ordered[index + 1]["time_s"] if index + 1 < len(ordered) else end_s
        right = min(end_s, next_time)
        if right <= left:
            continue
        duration = right - left
        power = float(sample["power_w"])
        total_j += power * duration
        incremental_j += max(0.0, power - baseline_w) * duration
    return {"total_j": total_j, "incremental_j": incremental_j}


class PowerSampler:
    def __init__(self, interval_ms: int):
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "PowerSampler":
        self.process = subprocess.Popen(
            [
                "nvidia-smi", "--query-gpu=power.draw,utilization.gpu",
                "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        def read() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                parts = [item.strip() for item in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    self.samples.append({
                        "time_s": time.perf_counter(), "power_w": float(parts[0]),
                        "utilization_percent": float(parts[1]),
                    })
                except ValueError:
                    continue

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=3)


async def probe_role(
    client: httpx.AsyncClient,
    endpoint: EndpointSpec,
    incidents: list[Any],
    *,
    seed: int,
    baseline_seconds: float,
    post_seconds: float,
    sample_interval_ms: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, float]]]:
    with PowerSampler(sample_interval_ms) as sampler:
        await asyncio.sleep(baseline_seconds)
        started = time.perf_counter()
        calls = []
        for index, incident in enumerate(incidents):
            calls.append(await call_role(client, endpoint, incident, seed=seed + index))
        ended = time.perf_counter()
        await asyncio.sleep(post_seconds)
    pre = [sample["power_w"] for sample in sampler.samples if sample["time_s"] < started]
    if not pre:
        raise RuntimeError("no idle baseline power samples were collected")
    baseline_w = statistics.mean(pre)
    energy = integrate_power(sampler.samples, started, ended, baseline_w)
    workload_samples = [
        sample for sample in sampler.samples if started <= sample["time_s"] <= ended
    ]
    if not workload_samples:
        raise RuntimeError("no workload power samples were collected")
    count = len(calls)
    summary = {
        "role": endpoint.role, "model": endpoint.model, "requests": count,
        "valid_rate": sum(call.valid for call in calls) / count,
        "errors": sum(call.error is not None for call in calls),
        "prompt_tokens": sum(call.prompt_tokens for call in calls),
        "completion_tokens": sum(call.completion_tokens for call in calls),
        "total_tokens": sum(call.total_tokens for call in calls),
        "mean_tokens_per_request": statistics.mean(call.total_tokens for call in calls),
        "mean_latency_s": statistics.mean(call.latency_s for call in calls),
        "p95_latency_s": sorted(call.latency_s for call in calls)[min(count - 1, int(count * 0.95))],
        "wall_seconds": ended - started, "idle_baseline_power_w": baseline_w,
        "mean_workload_power_w": energy["total_j"] / (ended - started),
        "total_energy_j": energy["total_j"], "incremental_energy_j": energy["incremental_j"],
        "incremental_j_per_request": energy["incremental_j"] / count,
        "incremental_j_per_token": energy["incremental_j"] / max(1, sum(call.total_tokens for call in calls)),
        "mean_gpu_utilization_percent": statistics.mean(
            sample["utilization_percent"] for sample in workload_samples
        ),
        "power_samples": len(sampler.samples),
    }
    return summary, [asdict(call) for call in calls], sampler.samples


async def run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    episodes = read_episode_bank(args.bank.resolve(), split=args.split)
    incidents = [incident for episode in episodes for incident in episode.incidents][: args.requests]
    if len(incidents) != args.requests:
        raise ValueError("bank does not contain enough incidents")
    endpoints = [
        EndpointSpec("weak", args.weak_endpoint, args.weak_model),
        EndpointSpec("strong", args.strong_endpoint, args.strong_model),
    ]
    model_files = {
        "weak": args.weak_model_file.resolve(), "strong": args.strong_model_file.resolve(),
    }
    results = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), trust_env=False) as client:
        for index, endpoint in enumerate(endpoints):
            summary, calls, samples = await probe_role(
                client, endpoint, incidents, seed=args.seed + index * 10_000,
                baseline_seconds=args.baseline_seconds, post_seconds=args.post_seconds,
                sample_interval_ms=args.sample_interval_ms,
            )
            for suffix, rows in (("calls", calls), ("power", samples)):
                with (output / f"{endpoint.role}-{suffix}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            summary["model_file"] = {
                "path": str(model_files[endpoint.role]),
                "bytes": model_files[endpoint.role].stat().st_size,
                "sha256": sha256_file(model_files[endpoint.role]),
            }
            results[endpoint.role] = summary
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            await asyncio.sleep(args.cooldown_seconds)
    weak = results["weak"]
    strong = results["strong"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "development diagnostic; both model servers resident",
        "bank_sha256": sha256_file(args.bank.resolve()), "split": args.split,
        "requests_per_model": args.requests, "sample_interval_ms": args.sample_interval_ms,
        "order": [endpoint.role for endpoint in endpoints], "systems": results,
        "strong_over_weak": {
            "model_file_bytes_ratio": strong["model_file"]["bytes"] / weak["model_file"]["bytes"],
            "mean_latency_ratio": strong["mean_latency_s"] / weak["mean_latency_s"],
            "incremental_j_per_request_ratio": (
                strong["incremental_j_per_request"] / weak["incremental_j_per_request"]
                if weak["incremental_j_per_request"] > 0 else None
            ),
        },
        "limitations": [
            "Both model servers remained resident; idle residency is removed only through a local power baseline.",
            "nvidia-smi samples total GPU board power, not per-process power.",
            "The weak-then-strong order is not counterbalanced in this diagnostic."
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "multitown-model-cost-probe-manifest-v1",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--weak-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--weak-model", default="qwen3.5-4b")
    parser.add_argument("--weak-model-file", type=Path, required=True)
    parser.add_argument("--strong-endpoint", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--strong-model", default="qwen3.5-35b-a3b")
    parser.add_argument("--strong-model-file", type=Path, required=True)
    parser.add_argument("--baseline-seconds", type=float, default=3.0)
    parser.add_argument("--post-seconds", type=float, default=1.0)
    parser.add_argument("--cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
