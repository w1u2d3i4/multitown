from pathlib import Path
from subprocess import CompletedProcess

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
