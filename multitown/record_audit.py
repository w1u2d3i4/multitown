"""Verify frozen MultiTown artifact manifests without rewriting formal records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(project_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Return a machine-readable, non-mutating verification result."""

    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    verified_bytes = 0
    verified_files = 0

    if manifest.get("schema_version") != "multitown-artifact-manifest-v1":
        issues.append({
            "kind": "schema_version",
            "actual": manifest.get("schema_version"),
        })

    files = manifest.get("files")
    if not isinstance(files, list):
        files = []
        issues.append({"kind": "files_not_a_list"})

    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative:
            issues.append({"kind": "invalid_path", "value": relative})
            continue
        if relative in seen:
            issues.append({"kind": "duplicate_path", "path": relative})
            continue
        seen.add(relative)

        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError:
            issues.append({"kind": "path_escape", "path": relative})
            continue
        if not path.is_file():
            issues.append({"kind": "missing", "path": relative})
            continue

        actual_bytes = path.stat().st_size
        expected_bytes = item.get("bytes")
        if actual_bytes != expected_bytes:
            issues.append({
                "kind": "size_mismatch",
                "path": relative,
                "expected": expected_bytes,
                "actual": actual_bytes,
            })
            continue

        actual_hash = sha256_file(path)
        expected_hash = item.get("sha256")
        if actual_hash != expected_hash:
            issues.append({
                "kind": "sha256_mismatch",
                "path": relative,
                "expected": expected_hash,
                "actual": actual_hash,
            })
            continue
        verified_bytes += actual_bytes
        verified_files += 1

    return {
        "schema_version": "multitown-formal-record-audit-v1",
        "manifest": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "source_revision": manifest.get("source_revision"),
        "files_declared": len(files),
        "files_verified": verified_files,
        "bytes_verified": verified_bytes,
        "passed": not issues,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    results = [verify_manifest(root, root / value) for value in args.manifest]
    payload = {
        "schema_version": "multitown-formal-record-audit-bundle-v1",
        "passed": all(result["passed"] for result in results),
        "manifests": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
