from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from .common import read_json, read_jsonl, sha256_file, write_json


MARBLE_FILES = {
    "bargaining": "multiagentbench/bargaining/bargaining_main.jsonl",
    "coding": "multiagentbench/coding/coding_main.jsonl",
    "database": "multiagentbench/database/database_main.jsonl",
    "minecraft": "multiagentbench/minecraft/minecraft_main.jsonl",
    "research": "multiagentbench/research/research_main.jsonl",
}

UPSTREAM_DIRS = {
    "TeamBench": "TeamBench",
    "MultiAgentBench_MARBLE": "MARBLE",
}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def audit(
    project_root: Path, selected_upstreams: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    lock = read_json(project_root / "upstream-lock.json")
    issues: list[str] = []
    observed: dict[str, Any] = {}
    selected = tuple(selected_upstreams or UPSTREAM_DIRS)
    unknown = set(selected) - set(UPSTREAM_DIRS)
    if unknown:
        raise ValueError(f"unknown upstreams: {sorted(unknown)}")

    for name in selected:
        rel = UPSTREAM_DIRS[name]
        repo = project_root / "third_party" / rel
        expected = lock["upstreams"][name]
        if not (repo / ".git").is_dir():
            issues.append(f"{name}: missing checkout {repo}")
            continue
        revision = git(repo, "rev-parse", "HEAD")
        dirty = bool(git(repo, "status", "--porcelain"))
        license_sha = sha256_file(repo / "LICENSE")
        observed[name] = {
            "path": str(repo),
            "revision": revision,
            "dirty": dirty,
            "license_sha256": license_sha,
        }
        if revision != expected["revision"]:
            issues.append(f"{name}: revision mismatch")
        if dirty:
            issues.append(f"{name}: checkout is dirty")
        if license_sha != expected["license_sha256"]:
            issues.append(f"{name}: license hash mismatch")

    team = project_root / "third_party" / "TeamBench"
    if "TeamBench" in observed:
        expected = lock["upstreams"]["TeamBench"]
        for field, rel in (
            ("dataset", "shared/teambench_dataset.json"),
            ("official_test", "leaderboard/data/leaderboard_90_tasks.json"),
        ):
            path = team / rel
            digest = sha256_file(path)
            rows = read_json(path)
            count = len(rows if isinstance(rows, list) else rows["tasks"])
            observed["TeamBench"][field] = {"sha256": digest, "rows": count}
            if digest != expected[field]["sha256"] or count != expected[field]["rows"]:
                issues.append(f"TeamBench: {field} data mismatch")

    marble = project_root / "third_party" / "MARBLE"
    if "MultiAgentBench_MARBLE" in observed:
        expected = lock["upstreams"]["MultiAgentBench_MARBLE"]["datasets"]
        observed["MultiAgentBench_MARBLE"]["datasets"] = {}
        for domain, rel in MARBLE_FILES.items():
            path = marble / rel
            digest = sha256_file(path)
            count = len(read_jsonl(path))
            observed["MultiAgentBench_MARBLE"]["datasets"][domain] = {
                "sha256": digest,
                "rows": count,
            }
            if digest != expected[domain]["sha256"] or count != expected[domain]["rows"]:
                issues.append(f"MARBLE: {domain} data mismatch")

    return {
        "schema_version": "general-mas-upstream-audit-v1",
        "selected_upstreams": list(selected),
        "ok": not issues,
        "issues": issues,
        "observed": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--upstream", action="append", choices=tuple(UPSTREAM_DIRS),
        help="Audit only the selected upstream; repeat to select more than one.",
    )
    args = parser.parse_args()
    selected = tuple(args.upstream) if args.upstream else None
    result = audit(args.project_root.resolve(), selected)
    if args.output:
        write_json(args.output, result)
    print("OK" if result["ok"] else "FAILED")
    for issue in result["issues"]:
        print(f"- {issue}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
