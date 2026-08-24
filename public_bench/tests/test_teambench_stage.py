from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

from general_mas_bench.teambench_stage import grade_in_sandbox


def test_grader_receives_offline_npm_cache(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    team_root = project / "third_party" / "TeamBench"
    task_root = team_root / "tasks" / "JS2"
    task_root.mkdir(parents=True)
    (task_root / "grade.sh").write_text("#!/bin/sh\n")

    run_dir = project / "run"
    for name in ("workspace", "reports", "submission"):
        (run_dir / name).mkdir(parents=True)
    (run_dir / "reports" / "score.json").write_text(
        '{"pass": false, "primary": {"success": 0}, '
        '"secondary": {"partial_score": 0.0}}'
    )
    (project / ".cache" / "npm-grader").mkdir(parents=True)

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("general_mas_bench.teambench_stage.subprocess.run", fake_run)
    grade_in_sandbox(
        row={"source_task": "JS2", "grader": "grade.sh"},
        team_root=team_root,
        run_dir=run_dir,
        image="runner:test",
    )

    command = calls[0]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "npm_config_offline=true" in command
    assert "npm_config_cache=/npm-cache" in command
    assert any("dst=/npm-cache" in item for item in command)


def test_grader_timeout_is_a_deterministic_failure(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    team_root = project / "third_party" / "TeamBench"
    task_root = team_root / "tasks" / "HANG"
    task_root.mkdir(parents=True)
    (task_root / "grade.sh").write_text("#!/bin/sh\n")

    run_dir = project / "run"
    for name in ("workspace", "reports", "submission"):
        (run_dir / name).mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["docker", "rm", "--force"]:
            return CompletedProcess(args, 0, "", "")
        Path(args[args.index("--cidfile") + 1]).write_text("exact-container-id")
        raise TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr("general_mas_bench.teambench_stage.subprocess.run", fake_run)
    score = grade_in_sandbox(
        row={"source_task": "HANG", "grader": "grade.sh"},
        team_root=team_root,
        run_dir=run_dir,
        image="runner:test",
    )

    assert score["pass"] is False
    assert score["secondary"]["partial_score"] == 0.0
    assert score["failure_modes"] == ["grader_timeout"]
    assert score["grader_exit_code"] == 124
    assert calls[-1] == ["docker", "rm", "--force", "exact-container-id"]
    assert not (run_dir / "grader.cid").exists()
