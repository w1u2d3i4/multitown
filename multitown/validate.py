"""Validate duration, raw-record conservation, GPU coverage, and artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from .report import discover_architectures
from .runner import atomic_json, utc_now


ADVANCED_REQUIRED = (
    "config.json", "scenario_bank.json", "requests.jsonl", "decisions.jsonl",
    "system_metrics.jsonl", "status.json", "summary.json", "summary.csv",
    "decision_curves.png", "system_curves.png", "model_server_curves.png",
    "organization_curves.png", "confidence_calibration.png",
    "confidence_calibration.csv",
)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{number}: {exc}") from exc


def validate_architecture(run_dir: Path) -> dict[str, Any]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    decisions = list(rows(run_dir / "decisions.jsonl"))
    requests = list(rows(run_dir / "requests.jsonl"))
    metrics = list(rows(run_dir / "system_metrics.jsonl"))
    elapsed = [float(item["elapsed_s"]) for item in metrics]
    metric_gaps = [right - left for left, right in zip(elapsed, elapsed[1:])]
    expected_requests = sum(int(item["request_count"]) for item in decisions)
    decision_requests = [item for item in requests if item.get("expects_decision")]
    strict_count = sum(bool(item.get("strict_json_compliant")) for item in decision_requests)
    duration = float(config["duration_seconds"])
    interval = float(config["monitor_interval"])
    missing = [name for name in ADVANCED_REQUIRED if not (run_dir / name).exists()]
    checks = {
        "status_complete": status.get("state") == "complete",
        "continuous_duration_met": float(status.get("elapsed_s", 0)) >= duration,
        "decision_count_matches_status": len(decisions) == int(status.get("decisions", -1)),
        "request_count_conserved": len(requests) == expected_requests,
        "nonempty_decisions": len(decisions) > 0,
        "zero_request_errors": not any(item.get("error") for item in requests),
        "zero_invalid_decisions": not any(not item.get("valid") for item in decisions),
        "monitor_starts_near_zero": bool(elapsed) and elapsed[0] <= interval * 2,
        "monitor_reaches_deadline": bool(elapsed) and elapsed[-1] >= duration - interval * 2,
        "monitor_gap_within_two_intervals": bool(elapsed) and (not metric_gaps or max(metric_gaps) <= interval * 2),
        "gpu_telemetry_complete": bool(metrics) and not any(
            item.get("gpu_util_percent") is None or item.get("gpu_power_w") is None
            for item in metrics
        ),
        "gpu_watchdog_never_triggered": not (run_dir / "gpu-health-error.json").exists(),
        "required_artifacts_present": not missing,
    }
    return {
        "architecture": run_dir.name,
        "passed": all(checks.values()),
        "checks": checks,
        "duration_seconds_planned": duration,
        "elapsed_seconds": float(status["elapsed_s"]),
        "decisions": len(decisions),
        "requests": len(requests),
        "expected_requests": expected_requests,
        "system_metric_samples": len(metrics),
        "system_metric_first_s": elapsed[0] if elapsed else None,
        "system_metric_last_s": elapsed[-1] if elapsed else None,
        "system_metric_max_gap_s": max(metric_gaps) if metric_gaps else None,
        "request_errors": sum(bool(item.get("error")) for item in requests),
        "invalid_decisions": sum(not item.get("valid") for item in decisions),
        "strict_json_decision_requests": strict_count,
        "decision_requests": len(decision_requests),
        "strict_json_request_rate": strict_count / len(decision_requests) if decision_requests else None,
        "missing_artifacts": missing,
    }


def validate(root: str | Path, expected: tuple[str, ...] = ("A3", "A4", "A5")) -> dict[str, Any]:
    root = Path(root).resolve()
    expected_set = set(expected)
    architectures = [name for name in discover_architectures(root) if name in expected_set]
    if architectures != list(expected):
        raise RuntimeError(f"Expected {'/'.join(expected)} under {root}; found {architectures}")
    results = [validate_architecture(root / architecture) for architecture in architectures]
    payload = {
        "validated_at_utc": utc_now(),
        "root": str(root),
        "passed": all(item["passed"] for item in results),
        "architectures": results,
    }
    atomic_json(root / "validation.json", payload)
    lines = [
        f"# {'-'.join(expected)} formal-run validation", "",
        f"Overall: **{'PASS' if payload['passed'] else 'FAIL'}**", "",
        "| Architecture | Result | Planned (s) | Elapsed (s) | Decisions | Requests | GPU samples | Max gap (s) | Errors | Invalid | Strict JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['architecture']} | {'PASS' if item['passed'] else 'FAIL'} | "
            f"{item['duration_seconds_planned']:.0f} | {item['elapsed_seconds']:.3f} | "
            f"{item['decisions']:,} | {item['requests']:,} | {item['system_metric_samples']:,} | "
            f"{item['system_metric_max_gap_s']:.3f} | {item['request_errors']} | "
            f"{item['invalid_decisions']} | {item['strict_json_request_rate']:.3%} |"
        )
    lines.extend(["", "All individual checks are stored in `validation.json`.", ""])
    (root / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--architectures", nargs="+", default=["A3", "A4", "A5"])
    args = parser.parse_args()
    expected = tuple(args.architectures)
    if any(not item.startswith("A") or not item[1:].isdigit() for item in expected):
        parser.error("--architectures must contain names such as A3 or A6")
    result = validate(args.path, expected)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
