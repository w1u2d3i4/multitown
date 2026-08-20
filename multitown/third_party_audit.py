"""Validate pinned third-party source checkouts without modifying them."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


LOCK_SCHEMA = "multitown-third-party-lock-v1"


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def audit(lock_path: Path, project_root: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError(f"unsupported lock schema: {lock.get('schema_version')!r}")

    checks: list[dict[str, Any]] = []
    for source in lock["sources"]:
        if source["kind"] != "git":
            checks.append(
                {
                    "name": source["name"],
                    "kind": source["kind"],
                    "status": "locked_remote",
                    "expected_revision": source["revision"],
                }
            )
            continue

        checkout = project_root / source["local_path"]
        if not checkout.is_dir():
            checks.append(
                {
                    "name": source["name"],
                    "kind": "git",
                    "status": "missing",
                    "expected_revision": source["revision"],
                }
            )
            continue

        actual_revision = _git(checkout, "rev-parse", "HEAD")
        actual_url = _git(checkout, "remote", "get-url", "origin")
        clean = not bool(_git(checkout, "status", "--porcelain"))
        matches = actual_revision == source["revision"] and actual_url == source["url"] and clean
        checks.append(
            {
                "name": source["name"],
                "kind": "git",
                "status": "pass" if matches else "mismatch",
                "expected_revision": source["revision"],
                "actual_revision": actual_revision,
                "expected_url": source["url"],
                "actual_url": actual_url,
                "clean": clean,
            }
        )

    failures = [item for item in checks if item["status"] in {"missing", "mismatch"}]
    return {
        "schema_version": "multitown-third-party-audit-v1",
        "lock_path": str(lock_path.relative_to(project_root)),
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failure_count": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=Path("records/third-party-lock.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    lock_path = args.lock if args.lock.is_absolute() else root / args.lock
    report = audit(lock_path, root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
