"""Freeze a balanced 1,200-scenario MultiTown train/dev/test bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .masbench_routing import git_state, utc_now, write_json
from .scenarios import Scenario, build_scenario_bank


SCHEMA_VERSION = "multitown-scenario-bank-v2"
DEFAULT_BASE_SEED = 20260807
DEFAULT_SPLIT_SEED = 20260810
DEFAULT_COUNT = 1200
DEFAULT_SPLIT_COUNTS = {"train": 140, "dev": 30, "test": 30}


def canonical_scenario(scenario: Scenario) -> bytes:
    return json.dumps(
        scenario.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def scenario_sha256(scenario: Scenario) -> str:
    return hashlib.sha256(canonical_scenario(scenario)).hexdigest()


def split_scenarios(
    scenarios: list[Scenario],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    counts = dict(split_counts or DEFAULT_SPLIT_COUNTS)
    if set(counts) != {"train", "dev", "test"} or any(value <= 0 for value in counts.values()):
        raise ValueError("split_counts must contain positive train/dev/test counts")
    grouped: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        grouped[scenario.family].append(scenario)
    assignment: dict[str, str] = {}
    required = sum(counts.values())
    for family, rows in sorted(grouped.items()):
        if len(rows) != required:
            raise ValueError(f"family {family!r} has {len(rows)} scenarios, expected {required}")
        ranked = sorted(
            rows,
            key=lambda item: hashlib.sha256(
                f"{split_seed}:{family}:{item.scenario_id}:{scenario_sha256(item)}".encode("utf-8")
            ).hexdigest(),
        )
        cursor = 0
        for split in ("train", "dev", "test"):
            for scenario in ranked[cursor : cursor + counts[split]]:
                if scenario.scenario_id in assignment:
                    raise RuntimeError(f"duplicate scenario ID: {scenario.scenario_id}")
                assignment[scenario.scenario_id] = split
            cursor += counts[split]
    return assignment


def bank_rows(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    count: int = DEFAULT_COUNT,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> list[dict[str, Any]]:
    scenarios = build_scenario_bank(base_seed, count)
    assignment = split_scenarios(scenarios, split_seed=split_seed)
    rows = []
    for scenario in scenarios:
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "split": assignment[scenario.scenario_id],
            "scenario_sha256": scenario_sha256(scenario),
            **scenario.to_dict(),
        })
    return sorted(rows, key=lambda row: row["scenario_id"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path, *, base_seed: int, count: int, split_seed: int) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = bank_rows(base_seed=base_seed, count=count, split_seed=split_seed)
    bank_path = output / "scenario-bank.jsonl"
    with bank_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    split_ids = {
        split: sorted(row["scenario_id"] for row in rows if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    split_payload = {
        "schema_version": "multitown-scenario-splits-v1",
        "base_seed": base_seed,
        "split_seed": split_seed,
        "split_rule": "SHA256(split_seed:family:scenario_id:canonical_scenario_sha256), ranked within family",
        "counts": {split: len(ids) for split, ids in split_ids.items()},
        "family_counts": {
            split: dict(sorted(Counter(row["family"] for row in rows if row["split"] == split).items()))
            for split in split_ids
        },
        "scenario_ids": split_ids,
    }
    write_json(output / "splits.json", split_payload)
    project_root = Path(__file__).resolve().parents[1]
    revision, dirty = git_state(project_root)
    manifest = {
        "schema_version": "multitown-scenario-freeze-manifest-v1",
        "created_at_utc": utc_now(),
        "source_revision": revision,
        "source_dirty": dirty,
        "generator": "multitown.scenarios.build_scenario_bank",
        "base_seed": base_seed,
        "split_seed": split_seed,
        "scenario_count": len(rows),
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "splits": split_payload["counts"],
        "scenario_bank_sha256": _sha256(bank_path),
        "splits_sha256": _sha256(output / "splits.json"),
    }
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# MultiTown 1,200-scenario bank\n\n"
        "Deterministic A7/A8 bank generated from executable Python oracles. Each of six\n"
        "families contains 200 independent scenarios and is split into 140 train, 30 dev\n"
        "and 30 test rows by stable SHA-256 rank. Training code must load IDs from\n"
        "`splits.json`; test outcomes must never be used for fitting, preprocessing or\n"
        "threshold selection. `manifest.json` locks source, data and split hashes.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/multitown-v0.2-1200"))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args()
    freeze(args.output, base_seed=args.base_seed, count=args.count, split_seed=args.split_seed)


if __name__ == "__main__":
    main()
