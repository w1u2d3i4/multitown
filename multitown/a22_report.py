"""Validate and plot the frozen A22 adaptive-development result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_MANIFEST_SHA256 = (
    "fd1898483012c4595b8536b2f7af146a03f347be083543f0b21a39c706ae0015"
)
EXPECTED_RESULT_SHA256 = (
    "d8193ee28c2c81c87858408870c8af5d91b2f47ae49ca683d05512439154a719"
)
EXPECTED_SELECTION_SHA256 = (
    "e39aff85db69ad446ad7b60f3cb745d5ed02a7461368d72ea86022cb61da9ead"
)
# Keep the frozen formal labels local: report validation is intentionally
# independent of the PyTorch training stack.
MECHANISM_NAMES = (
    "reference",
    "lagrangian",
    "shield",
    "lagrangian-plus-shield",
)
OUTER_FOLDS = tuple(range(5))
TRAINING_SEEDS = (20260812, 20260813, 20260814)
UPDATES_PER_FIT = 120
EXPECTED_RAW_MANIFEST_FILES = 248


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_manifest_files(raw: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != EXPECTED_RAW_MANIFEST_FILES:
        raise RuntimeError("A22 raw manifest must contain exactly 248 files")
    expected = set()
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise RuntimeError(f"unsafe A22 manifest path: {name!r}")
        expected.add(name)
    actual = {
        path.relative_to(raw).as_posix()
        for path in raw.rglob("*") if path.is_file()
    }
    for marker in ("RUNNING.json", "INVALIDATED.json"):
        if marker in actual:
            raise RuntimeError(f"A22 raw state marker is present: {marker}")
    expected_with_manifest = expected | {"artifact-manifest.json"}
    if actual != expected_with_manifest:
        missing = sorted(expected_with_manifest - actual)
        extra = sorted(actual - expected_with_manifest)
        raise RuntimeError(
            f"A22 raw path-set mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    failures = []
    for name, metadata in files.items():
        path = raw / name
        if (
            not path.is_file() or path.stat().st_size != int(metadata["bytes"])
            or _sha256(path) != metadata["sha256"]
        ):
            failures.append(name)
    if failures:
        raise RuntimeError(f"A22 manifest validation failed: {failures[:5]}")


def validate_raw(raw: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = raw.resolve()
    manifest_path = raw / "artifact-manifest.json"
    result_path = raw / "result.json"
    selection_path = raw / "all-selections-frozen.json"
    if (
        _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256
        or _sha256(result_path) != EXPECTED_RESULT_SHA256
        or _sha256(selection_path) != EXPECTED_SELECTION_SHA256
    ):
        raise RuntimeError("A22 formal result pin mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_files(raw, manifest)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result["adaptive_development_gate_passed"] is not False
        or result["fits"] != 60 or result["calibration_rows"] != 36_000
        or result["outer_rows"] != 9_000
    ):
        raise RuntimeError("A22 frozen result scope mismatch")
    return manifest, result


def collect_training_logs(raw: Path) -> list[dict[str, Any]]:
    raw = raw.resolve()
    expected = {
        raw / "fits" / f"outer-fold-{fold}" / f"seed-{seed}" / mechanism
        / "training-metrics.jsonl"
        for fold in OUTER_FOLDS for seed in TRAINING_SEEDS
        for mechanism in MECHANISM_NAMES
    }
    actual = set(raw.glob("fits/outer-fold-*/seed-*/*/training-metrics.jsonl"))
    if actual != expected:
        raise RuntimeError("A22 report requires the exact 5 x 3 x 4 fit-log product")
    rows = []
    for path in sorted(expected):
        fit_rows = _read_jsonl(path)
        mechanism = path.parent.name
        seed = int(path.parent.parent.name.removeprefix("seed-"))
        fold = int(path.parent.parent.parent.name.removeprefix("outer-fold-"))
        if (
            len(fit_rows) != UPDATES_PER_FIT
            or {int(row["update"]) for row in fit_rows}
            != set(range(1, UPDATES_PER_FIT + 1))
            or any(
                row["mechanism"] != mechanism
                or int(row["training_seed"]) != seed
                or int(row["outer_fold"]) != fold
                for row in fit_rows
            )
        ):
            raise RuntimeError(f"A22 fit log is incomplete: {path}")
        rows.extend(fit_rows)
    return rows


def training_series(
    rows: Sequence[Mapping[str, Any]], metric: str, *, divisor: str | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for mechanism in MECHANISM_NAMES:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        updates = sorted({int(row["update"]) for row in mechanism_rows})
        if not updates:
            raise ValueError(f"missing mechanism rows: {mechanism}")
        values = []
        for update in updates:
            selected = [row for row in mechanism_rows if int(row["update"]) == update]
            sample = np.asarray([
                float(row[metric]) / (float(row[divisor]) if divisor else 1.0)
                for row in selected
            ], dtype=float)
            values.append(sample)
        matrix = np.stack(values)
        result[mechanism] = {
            "update": np.asarray(updates, dtype=int),
            "mean": matrix.mean(axis=1),
            "q10": np.quantile(matrix, 0.10, axis=1),
            "q90": np.quantile(matrix, 0.90, axis=1),
        }
    return result


def plot_training(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    specifications = (
        ("rollout_episode_success_rate", None, "Rollout success", "%", 100.0),
        ("rollout_tokens_per_episode", None, "Tokens per episode", "tokens", 1.0),
        ("rollout_unsafe_rate", None, "Unsafe episodes", "%", 100.0),
        (
            "rollout_wrong_per_fixed_mean_incident", None,
            "Wrong executions / fixed-mean incident", "%", 100.0,
        ),
        ("lambda_unsafe_after", None, "Unsafe multiplier", "lambda", 1.0),
        (
            "shield_interventions", "rollout_episodes",
            "Shield interventions / episode", "interventions", 1.0,
        ),
    )
    colors = {
        "reference": "#4c78a8", "lagrangian": "#f58518",
        "shield": "#54a24b", "lagrangian-plus-shield": "#b279a2",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True)
    for axis, (metric, divisor, title, unit, scale) in zip(
        axes.flat, specifications, strict=True,
    ):
        series = training_series(rows, metric, divisor=divisor)
        for mechanism in MECHANISM_NAMES:
            item = series[mechanism]
            axis.plot(
                item["update"], item["mean"] * scale,
                label=mechanism, color=colors[mechanism], linewidth=1.5,
            )
            axis.fill_between(
                item["update"], item["q10"] * scale, item["q90"] * scale,
                color=colors[mechanism], alpha=0.12, linewidth=0,
            )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("PPO update")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False,
    )
    fig.suptitle("A22 training diagnostics (mean and 10–90% across 15 fits)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _evaluation_panels(result: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    statistics = result["statistics"]["overall_fixed_seed_mean"]
    primary = statistics["primary_episode_success"]
    secondary = statistics["secondary"]
    return (
        (
            "Autonomous success", primary["a8_mean"] * 100,
            primary["a22_mean"] * 100, primary, "percentage points", 100.0,
        ),
        (
            "Unsafe episodes", secondary["wrong_execution"]["a8_mean"] * 100,
            secondary["wrong_execution"]["a22_mean"] * 100,
            secondary["wrong_execution"], "percentage points", 100.0,
        ),
        (
            "Wrong executions / incident",
            secondary["wrong_executions_per_incident"]["a8_ratio"] * 100,
            secondary["wrong_executions_per_incident"]["a22_ratio"] * 100,
            secondary["wrong_executions_per_incident"], "percentage points", 100.0,
        ),
        (
            "Tokens per episode", secondary["tokens_used"]["a8_mean"],
            secondary["tokens_used"]["a22_mean"], secondary["tokens_used"],
            "tokens", 1.0,
        ),
    )


def plot_evaluation(result: Mapping[str, Any], output: Path) -> None:
    panels = _evaluation_panels(result)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    for axis, (title, a8, a22, interval, unit, scale) in zip(
        axes.flat, panels, strict=True,
    ):
        bars = axis.bar(["A8", "A22"], [a8, a22], color=["#9da3a6", "#4c78a8"])
        axis.bar_label(bars, fmt="%.2f", padding=3)
        difference = float(interval["point"]) * scale
        low = float(interval["ci95_low"]) * scale
        high = float(interval["ci95_high"]) * scale
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.text(
            0.5, 0.02, f"A22-A8 {difference:+.2f}  (95% CI {low:+.2f}, {high:+.2f})",
            transform=axis.transAxes, ha="center", va="bottom", fontsize=9,
        )
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.22)
    fig.suptitle("A22 selected outer meta-policy: retained negative development result")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate(raw: Path, output: Path) -> dict[str, Any]:
    manifest, result = validate_raw(raw)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"A22 report output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        training_path = staging / "training-diagnostics.png"
        evaluation_path = staging / "outer-comparison.png"
        plot_training(collect_training_logs(raw), training_path)
        plot_evaluation(result, evaluation_path)
        summary = {
            "schema_version": "multitown-a22-formal-report-summary-v1",
            "raw_artifact_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "raw_result_sha256": EXPECTED_RESULT_SHA256,
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "raw_manifest_files_verified": len(manifest["files"]),
            "adaptive_development_gate_passed": False,
            "report_generator": {
                "source_sha256": _sha256(Path(__file__)),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "plots": {
                training_path.name: _sha256(training_path),
                evaluation_path.name: _sha256(evaluation_path),
            },
        }
        (staging / "report-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        staging.replace(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.raw_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
