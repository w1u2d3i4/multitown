from pathlib import Path

from harness.agent_loop import AgentTurn

from general_mas_bench.teambench_runner import (
    DEFAULT_CONTROLLER,
    _allow_initial_early_stop,
    _attestation,
    _command_failure_evidence,
    _controller_attestation,
    _logged_command_signals,
    _phase_config,
    _replan_decision,
    _restore_candidate,
    _runtime_validator,
    _select_failed_review_candidate,
    _sequential_decision,
    _should_interrupt_for_replan,
    _snapshot_candidate,
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


def test_runtime_validator_records_repeat_timeout_and_turn_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "code.py").write_text("before\n")
    from general_mas_bench.teambench_stage import tree_sha256

    initial = tree_sha256(tmp_path)
    (tmp_path / "code.py").write_text("after\n")
    turns = [
        AgentTurn(
            turn=index,
            role="executor",
            tool_calls=[{"name": "run", "args": {"cmd": "npm install"}}],
            tool_results=[{
                "stdout": "",
                "stderr": "command timed out after 120s",
                "exit_code": 124,
            }],
        )
        for index in range(2)
    ]

    value = _runtime_validator(
        turns=turns,
        initial_hash=initial,
        workspace=tmp_path,
        category="Other",
        difficulty="medium",
        controller=DEFAULT_CONTROLLER,
        max_turns=2,
    )

    assert value["timed_out_commands"] == 2
    assert value["repeated_commands"] == 1
    assert value["max_command_repetitions"] == 2
    assert value["max_failed_command_repetitions"] == 2
    assert value["turn_budget_exhausted"] is True


def test_sequential_decision_reviews_runtime_risk_with_budget() -> None:
    validator = {
        "workspace_changed": True,
        "reliability_score": 0.8,
        "hard_fail": False,
        "successful_commands": 2,
        "failed_commands": 1,
        "timed_out_commands": 0,
        "repeated_commands": 0,
        "turns_used": 5,
        "turn_budget_exhausted": False,
    }

    review = _sequential_decision(
        validator,
        consumed_tokens=40_000,
        controller=DEFAULT_CONTROLLER,
    )
    stopped = _sequential_decision(
        validator,
        consumed_tokens=120_000,
        controller=DEFAULT_CONTROLLER,
    )

    assert review["action"] == "review"
    assert review["triggers"] == ["failed_commands"]
    assert stopped["action"] == "stop"
    assert stopped["reason"] == "token_budget_exhausted"


def test_replan_decision_ignores_ordinary_failed_test_after_success() -> None:
    validator = {
        "workspace_changed": True,
        "successful_commands": 2,
        "failed_commands": 1,
        "timed_out_commands": 0,
        "max_failed_command_repetitions": 1,
        "turns_used": 12,
    }

    decision = _replan_decision(
        validator,
        plan_delivered=True,
        plan_retry_used=False,
        consumed_tokens=40_000,
        controller=DEFAULT_CONTROLLER,
    )

    assert decision["action"] == "stop"
    assert decision["triggers"] == []


def test_replan_decision_escalates_hard_runtime_failure_with_budget() -> None:
    validator = {
        "workspace_changed": True,
        "successful_commands": 0,
        "failed_commands": 2,
        "timed_out_commands": 1,
        "max_failed_command_repetitions": 3,
        "turns_used": 3,
    }

    decision = _replan_decision(
        validator,
        plan_delivered=True,
        plan_retry_used=False,
        consumed_tokens=40_000,
        controller=DEFAULT_CONTROLLER,
    )

    assert decision["action"] == "escalate"
    assert decision["triggers"] == [
        "command_timeout",
        "repeated_command",
        "all_commands_failed",
    ]


def test_replan_ignores_unchanged_workspace_after_successful_report_check() -> None:
    validator = {
        "workspace_changed": False,
        "successful_commands": 1,
        "failed_commands": 0,
        "timed_out_commands": 0,
        "max_failed_command_repetitions": 0,
        "turns_used": 5,
    }

    decision = _replan_decision(
        validator,
        plan_delivered=True,
        consumed_tokens=30_000,
        controller=DEFAULT_CONTROLLER,
    )

    assert decision["action"] == "stop"
    assert decision["triggers"] == []


def test_replan_escalates_unchanged_workspace_without_success() -> None:
    validator = {
        "workspace_changed": False,
        "successful_commands": 0,
        "failed_commands": 0,
        "timed_out_commands": 0,
        "max_failed_command_repetitions": 0,
        "turns_used": 5,
    }

    decision = _replan_decision(
        validator,
        plan_delivered=True,
        consumed_tokens=30_000,
        controller=DEFAULT_CONTROLLER,
    )

    assert decision["action"] == "escalate"
    assert decision["triggers"] == ["workspace_unchanged_no_success"]


def test_logged_command_signals_drive_live_replan_interrupt(tmp_path: Path) -> None:
    log_dir = tmp_path / "executor"
    log_dir.mkdir()
    for index in range(3):
        (log_dir / f"turn_{index:03d}.json").write_text(
            '{"tool_calls":[{"name":"run","args":{"cmd":"npm install"}}],'
            '"tool_results":[{"stdout":"","stderr":"","exit_code":1}]}'
        )

    signals = _logged_command_signals(log_dir)

    assert signals["failed_commands"] == 3
    assert signals["max_command_repetitions"] == 3
    assert signals["max_failed_command_repetitions"] == 3
    assert _should_interrupt_for_replan(log_dir, DEFAULT_CONTROLLER)


def test_successful_validation_rerun_does_not_interrupt_replan(tmp_path: Path) -> None:
    log_dir = tmp_path / "executor"
    log_dir.mkdir()
    for index in range(2):
        (log_dir / f"turn_{index:03d}.json").write_text(
            '{"tool_calls":[{"name":"run","args":{"cmd":"pytest -q"}}],'
            '"tool_results":[{"stdout":"ok","stderr":"","exit_code":0}]}'
        )

    signals = _logged_command_signals(log_dir)

    assert signals["max_command_repetitions"] == 2
    assert signals["max_failed_command_repetitions"] == 0
    assert not _should_interrupt_for_replan(log_dir, DEFAULT_CONTROLLER)


def test_replan_failure_evidence_is_bounded_and_ignores_success() -> None:
    turns = [
        AgentTurn(
            turn=0,
            role="executor",
            tool_calls=[
                {"name": "run", "args": {"cmd": "pytest -q"}},
                {"name": "run", "args": {"cmd": "python -m compileall ."}},
            ],
            tool_results=[
                {"stdout": "failed", "stderr": "AssertionError", "exit_code": 1},
                {"stdout": "ok", "stderr": "", "exit_code": 0},
            ],
        )
    ]

    evidence = _command_failure_evidence(turns)

    assert evidence == [{
        "command": "pytest -q",
        "exit_code": 1,
        "stdout_tail": "failed",
        "stderr_tail": "AssertionError",
    }]


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


def test_candidate_snapshot_restores_workspace_and_reports(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    reports = tmp_path / "reports"
    snapshot = tmp_path / "candidate"
    workspace.mkdir()
    reports.mkdir()
    (workspace / "code.py").write_text("before\n")
    (reports / "answer.json").write_text('{"answer": "before"}\n')

    _snapshot_candidate(workspace, reports, snapshot)
    (workspace / "code.py").write_text("after\n")
    (reports / "answer.json").write_text('{"answer": "after"}\n')
    (reports / "recovery-only.txt").write_text("remove me\n")
    _restore_candidate(snapshot, workspace, reports)

    assert (workspace / "code.py").read_text() == "before\n"
    assert (reports / "answer.json").read_text() == '{"answer": "before"}\n'
    assert not (reports / "recovery-only.txt").exists()


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


def test_fixed_strategy_controller_attestation_is_explicit(tmp_path: Path) -> None:
    _controller_attestation(
        tmp_path,
        "task-1",
        "fixed_plan_execute_without_independent_review",
        source="fixed_strategy_protocol_controller",
    )

    value = _attestation(tmp_path)
    assert value is not None
    assert value["verdict"] == "pass"
    assert value["source"] == "fixed_strategy_protocol_controller"
    assert value["reason"] == "fixed_plan_execute_without_independent_review"
