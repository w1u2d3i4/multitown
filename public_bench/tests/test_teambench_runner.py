from pathlib import Path

from harness.agent_loop import AgentTurn

from general_mas_bench.teambench_runner import (
    DEFAULT_CONTROLLER,
    _allow_initial_early_stop,
    _attestation,
    _phase_config,
    _runtime_validator,
    _select_failed_review_candidate,
)


def test_runtime_validator_uses_observable_execution_signals(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("before\n")
    from general_mas_bench.teambench_stage import tree_sha256

    initial = tree_sha256(tmp_path)
    (tmp_path / "code.py").write_text("after\n")
    turn = AgentTurn(
        turn=0,
        role="executor",
        tool_calls=[
            {"name": "write", "args": {"path": "code.py"}},
            {"name": "run", "args": {"cmd": "python code.py"}},
        ],
        tool_results=[
            {"stdout": "", "stderr": "", "exit_code": 0},
            {"stdout": "after", "stderr": "", "exit_code": 0},
        ],
    )
    value = _runtime_validator(
        turns=[turn],
        initial_hash=initial,
        workspace=tmp_path,
        category="Other",
        difficulty="medium",
        controller=DEFAULT_CONTROLLER,
    )
    assert value["workspace_changed"] is True
    assert value["successful_commands"] == 1
    assert value["hard_fail"] is False
    assert "passed" not in value


def test_high_risk_task_cannot_take_initial_early_stop() -> None:
    validator = {
        "hard_fail": False,
        "reliability_score": 1.0,
        "successful_commands": 2,
    }
    assert _allow_initial_early_stop(
        validator,
        {"category": "Other", "difficulty": "medium"},
        DEFAULT_CONTROLLER,
    )
    assert not _allow_initial_early_stop(
        validator,
        {"category": "Cross-System Integration", "difficulty": "medium"},
        DEFAULT_CONTROLLER,
    )


def test_attestation_parses_valid_json_and_rejects_invalid(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    path.write_text('{"verdict": "pass"}')
    assert _attestation(tmp_path) == {"verdict": "pass"}
    path.write_text("not-json")
    assert _attestation(tmp_path) is None


def test_failed_review_falls_back_to_initial_candidate_on_tie(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    planned = tmp_path / "planned"
    workspace = tmp_path / "workspace"
    for path, value in ((initial, "initial"), (planned, "planned"), (workspace, "current")):
        path.mkdir()
        (path / "candidate.txt").write_text(value)

    selected = _select_failed_review_candidate(
        initial={"reliability_score": 0.9},
        post={"reliability_score": 0.9},
        initial_snapshot=initial,
        planned_snapshot=planned,
        workspace=workspace,
        controller={"preserve_initial_candidate": True},
    )

    assert selected == "initial"
    assert (workspace / "candidate.txt").read_text() == "initial"


def test_selective_planner_relative_reads_use_workspace(tmp_path: Path) -> None:
    task = tmp_path / "task"
    workspace = tmp_path / "workspace"
    for name in ("reports", "messages", "submission"):
        (tmp_path / name).mkdir()
    task.mkdir()
    workspace.mkdir()
    (task / "DECISIONS.md").write_text("wrong")
    (workspace / "DECISIONS.md").write_text("right")

    config = _phase_config(
        "planner",
        task=task,
        workspace=workspace,
        reports=tmp_path / "reports",
        messages=tmp_path / "messages",
        submission=tmp_path / "submission",
        image="unused",
        planner_reads_workspace=True,
    )

    result = config.tools[0].execute(path="DECISIONS.md")
    assert result.exit_code == 0
    assert result.stdout == "right"


def test_solo_has_full_read_access_but_cannot_write_task_spec(tmp_path: Path) -> None:
    task = tmp_path / "task"
    workspace = tmp_path / "workspace"
    reports = tmp_path / "reports"
    messages = tmp_path / "messages"
    submission = tmp_path / "submission"
    for path in (task, workspace, reports, messages, submission):
        path.mkdir()
    (task / "spec.md").write_text("full requirements")
    (workspace / "code.py").write_text("before")

    config = _phase_config(
        "solo",
        task=task,
        workspace=workspace,
        reports=reports,
        messages=messages,
        submission=submission,
        image="unused",
    )

    read_tool = next(tool for tool in config.tools if tool.name == "read")
    write_tool = next(tool for tool in config.tools if tool.name == "write")
    assert read_tool.execute(path="/task/spec.md").stdout == "full requirements"
    assert write_tool.execute(path="/workspace/code.py", content="after").exit_code == 0
    assert write_tool.execute(path="/task/spec.md", content="tampered").exit_code == 1
    assert (task / "spec.md").read_text() == "full requirements"
