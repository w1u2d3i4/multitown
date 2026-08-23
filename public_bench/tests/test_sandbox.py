import subprocess
from pathlib import Path

from general_mas_bench.sandbox import DockerCommandTool, SafeReadTool, SafeWriteTool


def test_safe_tools_reject_sibling_prefix_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    sibling = tmp_path / "work-escape"
    workspace.mkdir()
    sibling.mkdir()
    (workspace / "ok.txt").write_text("ok")
    (sibling / "secret.txt").write_text("secret")
    reader = SafeReadTool({"/workspace": workspace}, workspace)
    writer = SafeWriteTool({"/workspace": workspace}, workspace)
    assert reader.execute(path="ok.txt").stdout == "ok"
    assert reader.execute(path="../work-escape/secret.txt").exit_code == 1
    assert writer.execute(path="../work-escape/new.txt", content="bad").exit_code == 1
    assert not (sibling / "new.txt").exists()


def test_safe_read_lists_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "data.txt").write_text("x")
    (workspace / "folder").mkdir()
    result = SafeReadTool({"/workspace": workspace}, workspace).execute(path="/workspace")
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["data.txt", "folder/"]


def test_docker_command_timeout_normalizes_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["docker"], timeout=7, output=b"partial output", stderr=b"partial error"
        )

    monkeypatch.setattr("general_mas_bench.sandbox.subprocess.run", fake_run)
    result = DockerCommandTool(workspace, "test-image", timeout_s=7).execute(cmd="sleep 8")

    assert result.exit_code == 124
    assert result.stdout == "partial output"
    assert result.stderr == "partial error\ncommand timed out after 7s"
