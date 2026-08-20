"""Fail-closed private artifact I/O for frozen A15 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .stateful_model_protocol import (
    private_trace_manifest,
    run_scripted_model_actions,
    validate_private_trace_artifact,
)
from .stateful_ops import StatefulScenario, build_scenario


PRIVATE_TRACE_ARTIFACT_VERSION = "multitown-stateful-private-trace-artifact-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\n?\Z")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(payload: str, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            payload, object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _git_state() -> tuple[str, bool]:
    module_path = Path(__file__).resolve()
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=module_path.parent,
        text=True, capture_output=True, check=False,
    )
    if top_level.returncode or not top_level.stdout.strip():
        raise RuntimeError("private trace export requires its source Git checkout")
    project_root = Path(top_level.stdout.strip()).resolve()
    package_root = module_path.parent
    try:
        package_relative = package_root.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError("executed package is outside its reported Git checkout") from exc
    listed = subprocess.run(
        [
            "git", "ls-tree", "-r", "--name-only", "HEAD", "--",
            str(package_relative),
        ],
        cwd=project_root, text=True, capture_output=True, check=False,
    )
    if listed.returncode:
        raise RuntimeError("cannot enumerate Python sources at the reported revision")
    relative_sources = sorted(
        line for line in listed.stdout.splitlines() if line.endswith(".py")
    )
    if not relative_sources or str(module_path.relative_to(project_root)) not in relative_sources:
        raise RuntimeError("executed stateful source is not tracked by its Git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative_sources],
        cwd=project_root, text=True, capture_output=True, check=False,
    )
    if tracked.returncode:
        raise RuntimeError("executed stateful source is not tracked by its Git checkout")
    _assert_sources_match_head(project_root, relative_sources)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        text=True, capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root,
        text=True, capture_output=True, check=False,
    )
    if revision.returncode or status.returncode or not revision.stdout.strip():
        raise RuntimeError("private trace export requires a readable Git source state")
    return revision.stdout.strip(), bool(status.stdout.strip())


def _assert_sources_match_head(
    project_root: Path, relative_sources: list[str],
) -> None:
    """Compare disk bytes with HEAD, bypassing mutable index stat flags."""

    for relative in relative_sources:
        disk_path = project_root / relative
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=project_root,
            capture_output=True, check=False,
        )
        if blob.returncode or not disk_path.is_file():
            raise RuntimeError("cannot read a tracked Python source at HEAD")
        if disk_path.read_bytes() != blob.stdout:
            raise RuntimeError(
                f"executed Python source differs from HEAD: {relative}"
            )


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _atomic_new_file(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publishes the fully fsynced inode only if the destination
        # does not exist; unlike an exists()+replace() pair it cannot overwrite
        # a concurrently created artifact.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_private_trace_artifact(
    path: Path, scenario: StatefulScenario, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically write a clean-revision private trace and SHA-256 sidecar."""

    path = Path(path)
    sidecar = _sidecar_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError("private trace artifact or SHA-256 sidecar already exists")
    revision, dirty = _git_state()
    if dirty:
        raise RuntimeError("private trace export requires a clean source revision")
    manifest = private_trace_manifest(
        scenario, rows, source_revision=revision,
    )
    envelope = {
        "schema_version": PRIVATE_TRACE_ARTIFACT_VERSION,
        "source_dirty_at_export": False,
        "trace_manifest": manifest,
        "trace_manifest_sha256": _digest_bytes(_canonical(manifest).encode()),
        "public_trace": rows,
    }
    payload = (_canonical(envelope) + "\n").encode()
    payload_hash = _digest_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_new_file(path, payload)
    try:
        _atomic_new_file(sidecar, (payload_hash + "\n").encode("ascii"))
    except Exception:
        # An envelope without its sidecar is deliberately unusable. Remove only
        # the new file created by this operation so a clean retry is possible.
        path.unlink(missing_ok=True)
        raise
    return envelope


def load_private_trace_artifact(
    path: Path, scenario: StatefulScenario, *, expected_source_revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load raw JSON, require its sidecar, and validate provenance and replay."""

    if (
        not isinstance(expected_source_revision, str)
        or not expected_source_revision.strip()
    ):
        raise ValueError("expected source revision is required")
    path = Path(path)
    payload = path.read_bytes()
    sidecar = _sidecar_path(path)
    expected_hash = sidecar.read_text(encoding="ascii")
    if not _SHA256_PATTERN.fullmatch(expected_hash):
        raise ValueError("invalid private trace artifact SHA-256 sidecar")
    if _digest_bytes(payload) != expected_hash.strip():
        raise ValueError("private trace artifact SHA-256 mismatch")
    value = _strict_json(payload.decode("utf-8"), label="private trace artifact")
    required = {
        "schema_version", "source_dirty_at_export", "trace_manifest",
        "trace_manifest_sha256", "public_trace",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("invalid private trace artifact envelope schema")
    if value.get("schema_version") != PRIVATE_TRACE_ARTIFACT_VERSION:
        raise ValueError("unsupported private trace artifact version")
    if value.get("source_dirty_at_export") is not False:
        raise ValueError("private trace artifact source was dirty at export")
    manifest = value.get("trace_manifest")
    rows = value.get("public_trace")
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise ValueError("private trace artifact payload types are invalid")
    manifest_hash = _digest_bytes(_canonical(manifest).encode())
    if value.get("trace_manifest_sha256") != manifest_hash:
        raise ValueError("private trace artifact manifest hash mismatch")
    validate_private_trace_artifact(
        scenario, rows, manifest,
        expected_source_revision=expected_source_revision,
    )
    return rows, manifest


def _load_actions(path: Path) -> list[str]:
    value = _strict_json(path.read_text(encoding="utf-8"), label="action script")
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("action script must be a non-empty JSON array of strings")
    return value


def export_main() -> None:
    parser = argparse.ArgumentParser(description="Export one private A15 trace artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--dynamics-branch")
    args = parser.parse_args()
    scenario = build_scenario(
        args.family, variant_id=args.variant, world_seed=args.world_seed,
        dynamics_branch=args.dynamics_branch,
    )
    rows, terminal = run_scripted_model_actions(
        scenario, _load_actions(args.actions),
    )
    envelope = export_private_trace_artifact(args.output, scenario, rows)
    print(json.dumps({
        "artifact": str(args.output.resolve()),
        "schema_version": envelope["schema_version"],
        "row_count": len(rows),
        "terminal_success": terminal["success"],
        "source_revision": envelope["trace_manifest"]["source_revision"],
    }, ensure_ascii=False, indent=2))


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate one private A15 trace artifact")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--dynamics-branch")
    args = parser.parse_args()
    scenario = build_scenario(
        args.family, variant_id=args.variant, world_seed=args.world_seed,
        dynamics_branch=args.dynamics_branch,
    )
    rows, _ = load_private_trace_artifact(
        args.artifact, scenario,
        expected_source_revision=args.expected_source_revision,
    )
    print(json.dumps({
        "artifact": str(args.artifact.resolve()),
        "valid": True,
        "row_count": len(rows),
    }, ensure_ascii=False, indent=2))
