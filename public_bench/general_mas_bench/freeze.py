from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .common import canonical_sha256, read_json, sha256_file, write_json
from .upstream_audit import MARBLE_FILES, audit


EXCLUDED_TEAMBENCH_TASK = "GH120_redis-py_3863"


def resolve_test_task(
    tasks_root: Path,
    task_id: str,
    seed: int = 0,
    has_generator: bool | None = None,
) -> dict[str, Any]:
    exact = tasks_root / task_id
    seeded = tasks_root / f"{task_id}_seed{seed}"
    if exact.is_dir() and (exact / "task.yaml").is_file():
        source = exact
        generated = (
            has_generator
            if has_generator is not None
            else not (exact / "spec.md").is_file() or not (exact / "brief.md").is_file()
        )
    elif seeded.is_dir() and (seeded / "task.yaml").is_file():
        source = seeded
        generated = False
    else:
        raise FileNotFoundError(f"cannot resolve TeamBench task {task_id}")
    grader = "grade.sh" if (source / "grade.sh").is_file() else "workspace/check_solution.py"
    if not (source / grader).is_file():
        raise FileNotFoundError(f"no deterministic grader for {task_id}")
    return {
        "task_id": task_id,
        "source_task": source.name,
        "seed": seed,
        "generated_at_runtime": generated,
        "grader": grader,
    }


def round_robin_stratified(
    rows: list[dict[str, Any]], count: int, seed: str
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "Other")].append(row)
    for category, values in by_category.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{category}:{row['task_id']}".encode("utf-8")
            ).hexdigest()
        )
    chosen: list[dict[str, Any]] = []
    categories = sorted(by_category)
    while len(chosen) < count:
        progressed = False
        for category in categories:
            if by_category[category]:
                chosen.append(by_category[category].pop(0))
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            raise ValueError(f"requested {count} rows from only {len(chosen)} candidates")
    return chosen


def _metadata_index(dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in dataset}


def freeze(project_root: Path, output_root: Path, dev_count: int = 30) -> dict[str, Any]:
    audit_result = audit(project_root)
    if not audit_result["ok"]:
        raise RuntimeError(f"upstream audit failed: {audit_result['issues']}")

    team = project_root / "third_party" / "TeamBench"
    tasks_root = team / "tasks"
    dataset = read_json(team / "shared" / "teambench_dataset.json")
    metadata = _metadata_index(dataset)
    public = read_json(team / "leaderboard" / "data" / "leaderboard_90_tasks.json")["tasks"]
    public_ids = {str(row["task_id"]) for row in public}

    test_rows = []
    for row in public:
        task_id = str(row["task_id"])
        if task_id == EXCLUDED_TEAMBENCH_TASK:
            continue
        resolved = resolve_test_task(
            tasks_root,
            task_id,
            has_generator=bool(metadata.get(task_id, {}).get("has_generator", False)),
        )
        test_rows.append({**row, **resolved, "split": "test"})

    candidates: list[dict[str, Any]] = []
    for source in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        required = ("task.yaml", "spec.md", "brief.md", "grade.sh")
        if not all((source / name).is_file() for name in required):
            continue
        task_yaml = yaml.safe_load((source / "task.yaml").read_text(encoding="utf-8")) or {}
        base_id = str(task_yaml.get("task_id") or source.name)
        if base_id in public_ids or source.name in public_ids:
            continue
        if any(source.name.startswith(f"{task_id}_seed") for task_id in public_ids):
            continue
        meta = metadata.get(source.name) or metadata.get(base_id) or {}
        candidates.append({
            "task_id": source.name,
            "source_task": source.name,
            "seed": 0,
            "generated_at_runtime": bool(meta.get("has_generator", False)),
            "grader": "grade.sh",
            "category": str(meta.get("category") or task_yaml.get("category") or "Other"),
            "difficulty": str(meta.get("difficulty") or task_yaml.get("difficulty") or "unknown"),
        })

    dev_rows = [
        {**row, "split": "dev"}
        for row in round_robin_stratified(candidates, dev_count, "general-mas-dev-v1")
    ]
    split_rows = dev_rows + test_rows
    if len({row["source_task"] for row in split_rows}) != len(split_rows):
        raise ValueError("dev/test source task overlap")

    split_payload = {
        "schema_version": "general-mas-teambench-split-v1",
        "selection": {
            "dev": "SHA256-ranked round-robin over non-leaderboard categories",
            "test": "official TeamBench-90 minus upstream re-curation exclusion",
            "excluded_test": EXCLUDED_TEAMBENCH_TASK,
        },
        "counts": {"dev": len(dev_rows), "test": len(test_rows)},
        "rows": split_rows,
    }
    split_payload["split_sha256"] = canonical_sha256(split_payload)

    team_output = output_root / "teambench-v1"
    write_json(team_output / "split.json", split_payload)
    team_manifest = {
        "schema_version": "general-mas-teambench-manifest-v1",
        "upstream_revision": audit_result["observed"]["TeamBench"]["revision"],
        "source_dataset_sha256": sha256_file(team / "shared" / "teambench_dataset.json"),
        "official_test_sha256": sha256_file(
            team / "leaderboard" / "data" / "leaderboard_90_tasks.json"
        ),
        "split_sha256": sha256_file(team_output / "split.json"),
        "counts": split_payload["counts"],
    }
    write_json(team_output / "manifest.json", team_manifest)

    marble = project_root / "third_party" / "MARBLE"
    marble_manifest = {
        "schema_version": "general-mas-marble-manifest-v1",
        "upstream_revision": audit_result["observed"]["MultiAgentBench_MARBLE"]["revision"],
        "domains": {
            domain: {
                "path": rel,
                "rows": audit_result["observed"]["MultiAgentBench_MARBLE"]["datasets"][domain]["rows"],
                "sha256": sha256_file(marble / rel),
            }
            for domain, rel in MARBLE_FILES.items()
        },
    }
    marble_manifest["manifest_sha256"] = canonical_sha256(marble_manifest)
    write_json(output_root / "marble-v1" / "manifest.json", marble_manifest)
    return {"teambench": team_manifest, "marble": marble_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks"))
    parser.add_argument("--dev-count", type=int, default=30)
    args = parser.parse_args()
    result = freeze(args.project_root.resolve(), args.output_root.resolve(), args.dev_count)
    print(f"TeamBench split: {result['teambench']['counts']}")
    print(f"MARBLE domains: {len(result['marble']['domains'])}")


if __name__ == "__main__":
    main()
