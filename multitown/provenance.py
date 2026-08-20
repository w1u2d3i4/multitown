"""Create content-addressed manifests for formal benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=project_root, text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _artifact_summary(root: Path) -> dict[str, Any]:
    summary_paths = sorted(root.glob("A*/summary.json"))
    if root.name.startswith("A") and (root / "summary.json").exists():
        summary_paths = [root / "summary.json"]
    summaries = {}
    for path in summary_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        summaries[str(value.get("architecture", path.parent.name))] = {
            key: value.get(key)
            for key in (
                "decisions", "correct", "accuracy", "request_errors", "wall_hours",
                "total_tokens", "tokens_per_decision", "latency_mean_s", "latency_p95_s",
                "gpu_energy_kwh",
            )
        }
    return summaries


def build_manifest(project_root: Path, artifact_roots: list[Path]) -> dict[str, Any]:
    project_root = project_root.resolve()
    files: list[dict[str, Any]] = []
    roots_payload = []
    for root in artifact_roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        root_files = sorted(path for path in root.rglob("*") if path.is_file())
        roots_payload.append({
            "path": str(root.relative_to(project_root)),
            "file_count": len(root_files),
            "byte_count": sum(path.stat().st_size for path in root_files),
            "summaries": _artifact_summary(root),
        })
        for path in root_files:
            files.append({
                "path": str(path.relative_to(project_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    status = _git(project_root, "status", "--porcelain")
    return {
        "schema_version": "multitown-artifact-manifest-v1",
        "source_revision": _git(project_root, "rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "artifact_roots": roots_payload,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--artifact-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project_root = Path(args.project_root)
    payload = build_manifest(project_root, [Path(value) for value in args.artifact_root])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "source_revision": payload["source_revision"],
        "source_dirty": payload["source_dirty"],
        "files": len(payload["files"]),
        "bytes": sum(item["bytes"] for item in payload["files"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
