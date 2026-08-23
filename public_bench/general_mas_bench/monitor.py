from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import psutil


def _gpu() -> dict[str, float | None]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if result.returncode or not result.stdout.strip():
        return {
            "gpu_util_percent": None,
            "gpu_power_w": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
        }
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    keys = (
        "gpu_util_percent", "gpu_power_w", "gpu_memory_used_mb",
        "gpu_memory_total_mb", "gpu_temperature_c",
    )
    def numeric(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {key: numeric(value) for key, value in zip(keys, values, strict=True)}


class SystemMonitor:
    def __init__(self, output: Path, interval_s: float = 5.0):
        self.output = output
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "SystemMonitor":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(10, self.interval_s * 2))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            row = {
                "elapsed_s": time.perf_counter() - self.started,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_bytes": psutil.virtual_memory().used,
                **_gpu(),
            }
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            self.stop_event.wait(self.interval_s)
