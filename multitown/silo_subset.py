"""Run and normalize a monitored 18-cell Silo-Bench subset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.server
import json
import os
import platform
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np

from .advanced_runner import monitor_dual_system
from .masbench_routing import git_state, load_jsonl, utc_now, write_json
from .provenance import build_manifest

plt.switch_backend("Agg")

TASK_STEMS = ("I-01_n2", "I-01_n5", "II-11_n2", "II-11_n5", "III-21_n2", "III-21_n5")
PROTOCOLS = ("msg", "broadcast", "sfs")
PROTOCOL_LABELS = {"msg": "P2P", "broadcast": "Broadcast", "sfs": "SFS"}
LEVELS = ("I", "II", "III")
AGENT_COUNTS = (2, 5)


def capped_chat_payload(
    payload: dict[str, Any], *, max_tokens: int, temperature: float, seed: int
) -> dict[str, Any]:
    value = dict(payload)
    value["max_tokens"] = max_tokens
    value["temperature"] = temperature
    value["top_p"] = 1.0
    value["seed"] = seed
    return value


class CappingProxy:
    """Minimal non-streaming OpenAI proxy that bounds otherwise unlimited Silo output."""

    def __init__(
        self,
        target_base: str,
        target_api_key: str,
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        self.target_origin = target_base.rstrip("/").removesuffix("/v1")
        self.target_api_key = target_api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.server: http.server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> str:
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _forward(self, method: str) -> None:
                body = (
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    if method == "POST"
                    else b""
                )
                if method == "POST" and self.path.endswith("/chat/completions"):
                    payload = json.loads(body.decode("utf-8"))
                    body = json.dumps(
                        capped_chat_payload(
                            payload,
                            max_tokens=owner.max_tokens,
                            temperature=owner.temperature,
                            seed=owner.seed,
                        )
                    ).encode("utf-8")
                try:
                    response = httpx.request(
                        method,
                        owner.target_origin + self.path,
                        content=body or None,
                        headers={
                            "Authorization": f"Bearer {owner.target_api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=600.0,
                        trust_env=False,
                    )
                    self.send_response(response.status_code)
                    self.send_header(
                        "Content-Type",
                        response.headers.get("content-type", "application/json"),
                    )
                    self.send_header("Content-Length", str(len(response.content)))
                    self.end_headers()
                    self.wfile.write(response.content)
                except Exception as exc:  # noqa: BLE001 - proxy maps upstream failure.
                    content = json.dumps(
                        {"error": {"message": f"proxy {type(exc).__name__}: {exc}"}}
                    ).encode("utf-8")
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)

            def do_POST(self) -> None:
                self._forward("POST")

            def do_GET(self) -> None:
                self._forward("GET")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="silo-capping-proxy", daemon=True
        )
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_manifest(upstream_root: Path) -> list[dict[str, Any]]:
    values = []
    for stem in TASK_STEMS:
        path = upstream_root / "benchmarks" / f"{stem}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        values.append({"stem": stem, "path": str(path), "sha256": file_sha256(path)})
    return values


def normalize_case(case_dir: Path) -> dict[str, Any]:
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    results = json.loads((case_dir / "results.json").read_text(encoding="utf-8"))
    task_id = str(metadata["task"]["case_id"])
    level = task_id.split("-", 1)[0]
    execution = metadata["execution"]
    metrics = results["metrics"]
    return {
        "case_dir": str(case_dir),
        "case_id": metadata["case_id"],
        "task_id": task_id,
        "case_name": metadata["task"]["case_name"],
        "level": level,
        "agent_count": int(metadata["config"]["agent_count"]),
        "protocol": str(metadata["config"]["protocol"]),
        "model": metadata["config"]["model"],
        "status": execution["status"],
        "rounds": int(execution["current_round"]),
        "all_submitted": bool(execution.get("all_submitted", False)),
        "full_success": bool(results["success"]),
        "agent_success_rate": float(metrics["S_success_rate"]),
        "partial_correctness": float(metrics["P_partial_correctness"]),
        "communication_density": float(metrics["D_communication_density"]),
        "input_tokens": int(execution.get("total_input_tokens", 0)),
        "output_tokens": int(execution.get("total_output_tokens", 0)),
        "total_tokens": int(execution.get("total_input_tokens", 0))
        + int(execution.get("total_output_tokens", 0)),
        "submissions": len(results.get("submissions", [])),
    }


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "cases": count,
        "full_successes": sum(row["full_success"] for row in rows),
        "full_success_rate": sum(row["full_success"] for row in rows) / count
        if count
        else 0.0,
        "agent_success_rate_mean": float(
            np.mean([row["agent_success_rate"] for row in rows])
        )
        if rows
        else 0.0,
        "partial_correctness_mean": float(
            np.mean([row["partial_correctness"] for row in rows])
        )
        if rows
        else 0.0,
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "tokens_per_case": float(np.mean([row["total_tokens"] for row in rows]))
        if rows
        else 0.0,
        "rounds_mean": float(np.mean([row["rounds"] for row in rows])) if rows else 0.0,
        "all_submitted_rate": sum(row["all_submitted"] for row in rows) / count
        if count
        else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_level: dict[str, Any] = {}
    by_protocol: dict[str, Any] = {}
    by_agents: dict[str, Any] = {}
    for level in LEVELS:
        by_level[level] = grouped_summary(
            [row for row in rows if row["level"] == level]
        )
    for protocol in PROTOCOLS:
        by_protocol[protocol] = grouped_summary(
            [row for row in rows if row["protocol"] == protocol]
        )
    for count in AGENT_COUNTS:
        by_agents[str(count)] = grouped_summary(
            [row for row in rows if row["agent_count"] == count]
        )
    return {
        "schema_version": "multitown-silo-subset-v1",
        "evidence_level": "subset_reproduced",
        "adaptation_scope": "18-cell representative subset of upstream tasks; not the upstream 54-setting full matrix",
        "expected_cases": len(TASK_STEMS) * len(PROTOCOLS),
        "overall": grouped_summary(rows),
        "by_level": by_level,
        "by_protocol": by_protocol,
        "by_agent_count": by_agents,
    }


def plot_outcomes(rows: list[dict[str, Any]], output: Path) -> None:
    categories = [f"{level}-{agents}" for level in LEVELS for agents in AGENT_COUNTS]
    x = np.arange(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, protocol in enumerate(PROTOCOLS):
        values = []
        for level in LEVELS:
            for agents in AGENT_COUNTS:
                cell = [
                    row
                    for row in rows
                    if row["level"] == level
                    and row["agent_count"] == agents
                    and row["protocol"] == protocol
                ]
                values.append(100 * cell[0]["agent_success_rate"] if cell else 0.0)
        ax.bar(x + (index - 1) * width, values, width, label=PROTOCOL_LABELS[protocol])
    ax.set_xticks(x, categories)
    ax.set_xlabel("Level-agent count")
    ax.set_ylabel("Agent success rate (%)")
    ax.set_title("Silo-Bench representative subset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_tokens(rows: list[dict[str, Any]], output: Path) -> None:
    categories = [f"{level}-{agents}" for level in LEVELS for agents in AGENT_COUNTS]
    x = np.arange(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, protocol in enumerate(PROTOCOLS):
        values = []
        for level in LEVELS:
            for agents in AGENT_COUNTS:
                cell = [
                    row
                    for row in rows
                    if row["level"] == level
                    and row["agent_count"] == agents
                    and row["protocol"] == protocol
                ]
                values.append(cell[0]["total_tokens"] if cell else 0)
        ax.bar(x + (index - 1) * width, values, width, label=PROTOCOL_LABELS[protocol])
    ax.set_xticks(x, categories)
    ax.set_xlabel("Level-agent count")
    ax.set_ylabel("Total input + output tokens")
    ax.set_title("Silo-Bench communication cost")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_system(metrics_path: Path, output: Path) -> None:
    rows = load_jsonl(metrics_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    specs = (
        ("gpu_util_percent", "GPU utilization (%)"),
        ("gpu_power_w", "GPU power (W)"),
        ("cpu_percent", "CPU utilization (%)"),
        ("ram_used_gb", "RAM used (GiB)"),
    )
    elapsed = [row["elapsed_s"] / 60 for row in rows]
    for ax, (field, label) in zip(axes.flat, specs, strict=True):
        values = [row.get(field) for row in rows]
        ax.plot(elapsed, values, linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Elapsed minutes")
    axes[1, 1].set_xlabel("Elapsed minutes")
    fig.suptitle("Silo-Bench monitored system telemetry")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown(summary: dict[str, Any], config: dict[str, Any]) -> str:
    lines = [
        "# Silo-Bench representative subset",
        "",
        "Evidence level: **subset reproduced**. This is an 18-cell representative subset,",
        "not the upstream full matrix.",
        "",
        f"Model: `{config['model']}`; max rounds: {config['max_rounds']}; upstream commit: `{config['upstream_revision']}`.",
        "",
        "| Protocol | Full case success | Mean agent success | Tokens/case | Mean rounds |",
        "|---|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        value = summary["by_protocol"][protocol]
        lines.append(
            f"| {PROTOCOL_LABELS[protocol]} | {value['full_successes']}/{value['cases']} | "
            f"{100 * value['agent_success_rate_mean']:.1f}% | {value['tokens_per_case']:.1f} | {value['rounds_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Plots: `outcomes.png`, `tokens.png`, and `system-curves.png`.",
            "All upstream case directories remain in the ignored raw bundle and are content-addressed",
            "by `artifact-manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    upstream_root = args.upstream_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "workspace"
    source_revision, source_dirty = git_state(project_root)
    upstream_revision, upstream_dirty = git_state(upstream_root)
    proxy = CappingProxy(
        args.api_base,
        args.api_key,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    proxy_base = proxy.start()
    config = {
        "schema_version": "multitown-silo-subset-config-v1",
        "created_at_utc": utc_now(),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "upstream_revision": upstream_revision,
        "upstream_dirty": upstream_dirty,
        "upstream_root": str(upstream_root),
        "silo_python": str(args.silo_python.absolute()),
        "python": platform.python_version(),
        "task_stems": list(TASK_STEMS),
        "tasks": task_manifest(upstream_root),
        "protocols": list(PROTOCOLS),
        "agent_counts": list(AGENT_COUNTS),
        "expected_cases": len(TASK_STEMS) * len(PROTOCOLS),
        "model": args.model,
        "api_base": args.api_base,
        "silo_proxy_base": proxy_base,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "workers": 1,
        "monitor_interval_s": args.monitor_interval,
    }
    write_json(output / "config.json", config)
    command = [
        str(args.silo_python.absolute()),
        "-m",
        "src.batch_run",
        "--task-dir",
        str(upstream_root / "benchmarks"),
        "--task-ids",
        *TASK_STEMS,
        "--protocols",
        *PROTOCOLS,
        "--models",
        args.model,
        "--api-base",
        proxy_base,
        "--api-key",
        "multitown-proxy",
        "--max-rounds",
        str(args.max_rounds),
        "--workspace",
        str(workspace),
        "--workers",
        "1",
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    started = time.perf_counter()
    stop_event = asyncio.Event()
    monitor = asyncio.create_task(
        monitor_dual_system(
            run_dir=output,
            endpoints={"weak": args.api_base},
            api_key=args.api_key,
            started=started,
            interval=args.monitor_interval,
            stop_event=stop_event,
        )
    )
    process: asyncio.subprocess.Process | None = None
    return_code: int | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=upstream_root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        with (output / "upstream.log").open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            async for raw in process.stdout:
                line = raw.decode("utf-8", errors="replace")
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        return_code = await process.wait()
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        stop_event.set()
        await monitor
        proxy.stop()
    if return_code != 0:
        raise RuntimeError(f"upstream Silo-Bench exited with status {return_code}")

    rows = [
        normalize_case(case_dir)
        for case_dir in sorted(workspace.iterdir())
        if (case_dir / "results.json").is_file()
    ]
    summary = summarize(rows)
    summary.update(
        {
            "created_at_utc": utc_now(),
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "upstream_revision": upstream_revision,
            "wall_seconds": time.perf_counter() - started,
            "found_cases": len(rows),
        }
    )
    if len(rows) != config["expected_cases"]:
        raise RuntimeError(
            f"expected {config['expected_cases']} results, found {len(rows)}"
        )
    with (output / "normalized_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(
            rows,
            key=lambda value: (value["level"], value["agent_count"], value["protocol"]),
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "summary.json", summary)
    plot_outcomes(rows, output / "outcomes.png")
    plot_tokens(rows, output / "tokens.png")
    plot_system(output / "system_metrics.jsonl", output / "system-curves.png")
    (output / "RESULTS.md").write_text(markdown(summary, config), encoding="utf-8")
    manifest = build_manifest(project_root, [output])
    write_json(output / "artifact-manifest.json", manifest)
    if args.record_dir:
        record = args.record_dir.resolve()
        record.mkdir(parents=True, exist_ok=True)
        for name in (
            "config.json",
            "summary.json",
            "normalized_cases.jsonl",
            "RESULTS.md",
            "artifact-manifest.json",
            "outcomes.png",
            "tokens.png",
            "system-curves.png",
        ):
            shutil.copy2(output / name, record / name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root", type=Path, default=Path("third_party/Silo-Bench")
    )
    parser.add_argument("--silo-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen-mm-backup")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    args = parser.parse_args()
    if (
        args.max_rounds <= 0
        or args.max_output_tokens <= 0
        or args.monitor_interval <= 0
    ):
        raise SystemExit(
            "max-rounds, max-output-tokens and monitor-interval must be positive"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
