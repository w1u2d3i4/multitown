from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .common import canonical_sha256, sha256_file, write_json


def tree_sha256(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return canonical_sha256(rows)


def _copy_static(source: Path, run_dir: Path) -> None:
    workspace = run_dir / "workspace"
    source_workspace = source / "workspace"
    if source_workspace.is_dir():
        for item in source_workspace.iterdir():
            if item.name == "check_solution.py":
                continue
            target = workspace / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


def stage_task(row: dict[str, Any], team_root: Path, run_dir: Path) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"non-empty run directory: {run_dir}")
    workspace = run_dir / "workspace"
    reports = run_dir / "reports"
    messages = run_dir / "messages"
    submission = run_dir / "submission"
    task_view = run_dir / "task"
    for path in (workspace, reports, messages, submission, task_view):
        path.mkdir(parents=True, exist_ok=True)

    source = team_root / "tasks" / str(row["source_task"])
    if not source.is_dir():
        raise FileNotFoundError(source)

    generated = bool(row.get("generated_at_runtime"))
    if generated:
        from generators.registry import get_generator

        generator = get_generator(str(row["task_id"]))
        value = generator.generate(seed=int(row.get("seed", 0)))
        generator.write_to_disk(
            value,
            workspace_dir=str(workspace),
            reports_dir=str(reports),
            task_dir=str(task_view),
        )
    else:
        _copy_static(source, run_dir)
        for name in ("spec.md", "brief.md", "task.yaml"):
            if (source / name).is_file():
                shutil.copy2(source / name, task_view / name)

    for required in ("spec.md", "brief.md"):
        if not (task_view / required).is_file():
            raise FileNotFoundError(f"staged task missing {required}: {row['task_id']}")

    meta = {
        "schema_version": "general-mas-staged-task-v1",
        "task_id": row["task_id"],
        "source_task": row["source_task"],
        "seed": int(row.get("seed", 0)),
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
        "generated_at_runtime": generated,
        "grader": row["grader"],
        "initial_workspace_sha256": tree_sha256(workspace),
    }
    write_json(run_dir / "run_meta.json", meta)
    return meta


def run_setup_in_sandbox(
    *,
    row: dict[str, Any],
    team_root: Path,
    run_dir: Path,
    image: str,
) -> None:
    source = team_root / "tasks" / str(row["source_task"])
    if row.get("generated_at_runtime") or not (source / "setup.sh").is_file():
        return
    args = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,nosuid,size=1g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={team_root.resolve()},dst=/upstream,readonly",
        "--mount", f"type=bind,src={(run_dir / 'workspace').resolve()},dst=/workspace",
        "--mount", f"type=bind,src={(run_dir / 'reports').resolve()},dst=/reports",
        image, "bash", f"/upstream/tasks/{row['source_task']}/setup.sh",
        "/workspace", "/reports", "general-mas", str(row.get("seed", 0)),
    ]
    result = subprocess.run(args, text=True, capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(f"setup failed: {result.stderr[-2000:]}")


def grade_in_sandbox(
    *,
    row: dict[str, Any],
    team_root: Path,
    run_dir: Path,
    image: str,
) -> dict[str, Any]:
    cidfile = run_dir / "grader.cid"
    cidfile.unlink(missing_ok=True)
    common = [
        "docker", "run", "--rm", "--cidfile", str(cidfile),
        "--network", "none", "--read-only",
        "--pids-limit", "512", "--memory", "8g", "--cpus", "12",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,nosuid,size=2g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={team_root.resolve()},dst=/upstream,readonly",
        "--mount", f"type=bind,src={(run_dir / 'workspace').resolve()},dst=/workspace",
        "--mount", f"type=bind,src={(run_dir / 'reports').resolve()},dst=/reports",
        "--mount", f"type=bind,src={(run_dir / 'submission').resolve()},dst=/submission,readonly",
    ]
    # Some trusted upstream graders run npm to exercise the submitted program.
    # Keep the grader network-disabled, but let npm resolve those dependencies
    # from a host-side cache populated before the frozen run. The cache is not
    # mounted into any agent role, so it cannot become a network side channel.
    default_npm_cache = team_root.parent.parent / ".cache" / "npm-grader"
    npm_cache = Path(os.environ.get("GENERAL_MAS_NPM_CACHE", default_npm_cache))
    if npm_cache.is_dir():
        common.extend([
            "--env", "npm_config_offline=true",
            "--env", "npm_config_cache=/npm-cache",
            "--env", "npm_config_audit=false",
            "--env", "npm_config_fund=false",
            "--mount", f"type=bind,src={npm_cache.resolve()},dst=/npm-cache",
        ])
    source_task = str(row["source_task"])
    if row["grader"] == "workspace/check_solution.py":
        pristine = team_root / "tasks" / source_task / "workspace" / "check_solution.py"
        placeholder = run_dir / "workspace" / "check_solution.py"
        placeholder.touch()
        args = common + [
            "--mount", f"type=bind,src={pristine.resolve()},dst=/workspace/check_solution.py,readonly",
            image, "python", "/workspace/check_solution.py",
        ]
    else:
        expected = run_dir / "reports" / "expected.json"
        command = [
            "bash", f"/upstream/tasks/{source_task}/grade.sh",
            "/workspace", "/reports", "/submission", f"/upstream/tasks/{source_task}",
        ]
        if expected.is_file():
            command.append("/reports/expected.json")
        args = common + [image, *command]
    grader_timeout_s = 240
    try:
        result = subprocess.run(
            args, text=True, capture_output=True, timeout=grader_timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        if cidfile.is_file():
            container_id = cidfile.read_text(encoding="utf-8").strip()
            if container_id:
                subprocess.run(
                    ["docker", "rm", "--force", container_id],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
        # A submitted program can make an upstream shell grader hang (for
        # example, when the grader invokes a generated service without its own
        # timeout).  This is a deterministic task failure, not an LLM/provider
        # invocation error.  Fail closed so one bad candidate cannot invalidate
        # or stall an otherwise comparable benchmark sweep.
        return {
            "pass": False,
            "primary": {"success": 0},
            "secondary": {"partial_score": 0.0},
            "failure_modes": ["grader_timeout"],
            "grader_exit_code": 124,
            "grader_timeout_s": grader_timeout_s,
        }
    finally:
        if row["grader"] == "workspace/check_solution.py":
            placeholder.unlink(missing_ok=True)
        cidfile.unlink(missing_ok=True)
    score_path = run_dir / "reports" / "score.json"
    if not score_path.is_file():
        return {
            "pass": False,
            "primary": {"success": 0},
            "secondary": {"partial_score": 0.0},
            "failure_modes": ["grader_no_score"],
            "grader_exit_code": result.returncode,
            "grader_stdout": result.stdout[-2000:],
            "grader_stderr": result.stderr[-2000:],
        }
    return json.loads(score_path.read_text(encoding="utf-8"))
