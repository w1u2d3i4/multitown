from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from harness.agent_interface import Tool, ToolResult


def _subprocess_text(value: str | bytes | None) -> str:
    """Normalize subprocess output, including TimeoutExpired's bytes payloads."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _force_remove_container(name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception:
        # Cleanup must never replace the actual tool result. A surviving
        # container is still externally visible to the run-level audit.
        pass


def _safe_resolve(path: str, roots: dict[str, Path], base: Path) -> Path:
    for prefix, root in sorted(roots.items(), key=lambda item: len(item[0]), reverse=True):
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            suffix = path[len(prefix):].lstrip("/")
            candidate = (root / suffix).resolve()
            break
    else:
        candidate = (base / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    allowed = {root.resolve() for root in roots.values()}
    if not any(candidate == root or candidate.is_relative_to(root) for root in allowed):
        raise PermissionError(f"path outside allowed roots: {path}")
    return candidate


class SafeReadTool(Tool):
    name = "read"

    def __init__(self, roots: dict[str, Path], base: Path):
        self.roots = roots
        self.base = base

    def execute(self, path: str = "", **_: object) -> ToolResult:
        if not path:
            return ToolResult(stderr="path is required", exit_code=1)
        try:
            target = _safe_resolve(path, self.roots, self.base)
            if target.is_dir():
                entries = sorted(item.name + ("/" if item.is_dir() else "") for item in target.iterdir())
                return ToolResult(stdout="\n".join(entries))
            return ToolResult(stdout=target.read_text(encoding="utf-8"))
        except Exception as exc:
            return ToolResult(stderr=f"{type(exc).__name__}: {exc}", exit_code=1)


class SafeWriteTool(Tool):
    name = "write"

    def __init__(self, roots: dict[str, Path], base: Path):
        self.roots = roots
        self.base = base

    def execute(self, path: str = "", content: str = "", **_: object) -> ToolResult:
        if not path:
            return ToolResult(stderr="path is required", exit_code=1)
        try:
            target = _safe_resolve(path, self.roots, self.base)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(stdout=f"wrote {len(content.encode('utf-8'))} bytes to {path}")
        except Exception as exc:
            return ToolResult(stderr=f"{type(exc).__name__}: {exc}", exit_code=1)


class SafeMessageTool(Tool):
    name = "send_message"

    def __init__(self, messages_dir: Path, sender: str):
        self.messages_dir = messages_dir
        self.sender = sender

    def execute(self, to: str = "", content: str = "", **_: object) -> ToolResult:
        if to not in {"planner", "executor", "verifier", "controller", "all"}:
            return ToolResult(stderr=f"invalid message recipient: {to}", exit_code=1)
        if not content:
            return ToolResult(stderr="content is required", exit_code=1)
        self.messages_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": self.sender,
            "type": "message",
            "to": to,
            "content": content,
        }
        with (self.messages_dir / "dialogue.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return ToolResult(stdout=f"message sent to {to}")


class DockerCommandTool(Tool):
    name = "run"

    def __init__(
        self,
        workspace: Path,
        image: str,
        read_only: bool = False,
        timeout_s: int = 120,
        mounts: dict[str, tuple[Path, bool]] | None = None,
    ):
        self.workspace = workspace.resolve()
        self.image = image
        self.read_only = read_only
        self.timeout_s = timeout_s
        self.mounts = mounts or {"/workspace": (self.workspace, self.read_only)}

    def execute(self, cmd: str = "", **_: object) -> ToolResult:
        if not cmd:
            return ToolResult(stderr="cmd is required", exit_code=1)
        container_name = f"general-mas-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        args = [
            "docker", "run", "--rm", "--name", container_name,
            "--network", "none", "--read-only",
            "--pids-limit", "256", "--memory", "6g", "--cpus", "12",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,size=1g",
            "--user", f"{os.getuid()}:{os.getgid()}",
        ]
        for destination, (source, mount_read_only) in self.mounts.items():
            mount = f"type=bind,src={source.resolve()},dst={destination}"
            if mount_read_only:
                mount += ",readonly"
            args.extend(("--mount", mount))
        args.extend([
            "--workdir", "/workspace", self.image,
            "bash", "-lc", "export HOME=/tmp/home; mkdir -p $HOME; " + cmd,
        ])
        try:
            result = subprocess.run(
                args, text=True, capture_output=True, timeout=self.timeout_s, check=False
            )
            return ToolResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                stdout=_subprocess_text(exc.stdout),
                stderr=_subprocess_text(exc.stderr)
                + f"\ncommand timed out after {self.timeout_s}s",
                exit_code=124,
            )
        except Exception as exc:
            return ToolResult(stderr=f"{type(exc).__name__}: {exc}", exit_code=1)
        finally:
            # Killing a timed-out `docker run` client does not stop its
            # container. Always issue an idempotent, name-scoped cleanup so a
            # model command cannot leak CPU/RAM or contaminate later trials.
            _force_remove_container(container_name)


def docker_available() -> tuple[bool, str]:
    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0, (result.stdout.strip() or result.stderr.strip())
