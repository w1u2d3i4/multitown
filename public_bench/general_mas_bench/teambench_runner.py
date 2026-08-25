from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.agent_interface import RoleConfig
from harness.agent_loop import AgentLoop, AgentTurn

from .common import append_jsonl, read_json, write_json
from .model_adapter import OutcomeStopAdapter, RecordedAdapter
from .monitor import SystemMonitor
from .sandbox import (
    DockerCommandTool,
    SafeMessageTool,
    SafeReadTool,
    SafeWriteTool,
    docker_available,
)
from .strategy_selector import select_method
from .teambench_stage import (
    grade_in_sandbox,
    run_setup_in_sandbox,
    stage_task,
    tree_sha256,
)

DEFAULT_CONTROLLER = {
    "schema_version": "general-mas-a8-controller-v2",
    "early_stop_threshold": 0.75,
    "post_plan_verifier_skip_threshold": 0.85,
    "high_risk_categories": [
        "Adversarial", "Cross-System Integration", "Distributed Systems",
        "Data Engineering", "Long-Horizon", "Multi-language", "Operations",
        "Security",
    ],
    "high_risk_difficulties": ["expert"],
    "require_successful_command_for_early_stop": True,
    "max_remediation_loops": 1,
    "a4_max_remediation_loops": 1,
    "preserve_initial_candidate": True,
    "guided_remediation_without_reverification": True,
    "sequential_action_space": [
        "stop", "delegate", "escalate", "review", "human/abstain",
    ],
    "sequential_review_failed_commands": 1,
    "sequential_review_timeouts": 1,
    "sequential_review_repeated_commands": 2,
    "sequential_review_on_turn_exhaustion": True,
    "sequential_max_total_tokens": 120_000,
    "sequential_verifier_turns": 8,
    "sequential_remediation_turns": 6,
    "replan_action_space": [
        "stop", "delegate", "escalate", "review", "human/abstain",
    ],
    "replan_command_timeout_s": 30,
    "replan_interrupt_timeouts": 1,
    "replan_interrupt_failed_command_repetitions": 3,
    "replan_workspace_unchanged_requires_no_success": True,
    "replan_failed_without_success": 2,
    "replan_max_total_tokens": 90_000,
    "replan_planner_turns": 3,
    "replan_executor_turns": 6,
}


def _phase_config(
    role: str,
    *,
    task: Path,
    workspace: Path,
    reports: Path,
    messages: Path,
    submission: Path,
    image: str,
    planner_reads_workspace: bool = False,
    command_timeout_s: int = 120,
) -> RoleConfig:
    if role == "solo":
        command_mounts = {
            "/workspace": (workspace, False),
            "/shared/workspace": (workspace, False),
            "/reports": (reports, False),
            "/shared/reports": (reports, False),
            "/task": (task, True),
            "/submission": (submission, False),
            "/shared/submission": (submission, False),
        }
        tools = [
            DockerCommandTool(
                workspace,
                image,
                read_only=False,
                timeout_s=command_timeout_s,
                mounts=command_mounts,
            ),
            SafeReadTool(
                {
                    "/task": task,
                    "/workspace": workspace,
                    "/shared/workspace": workspace,
                    "/reports": reports,
                    "/shared/reports": reports,
                    "/submission": submission,
                    "/shared/submission": submission,
                },
                workspace,
            ),
            SafeWriteTool(
                {
                    "/workspace": workspace,
                    "/shared/workspace": workspace,
                    "/reports": reports,
                    "/shared/reports": reports,
                    "/submission": submission,
                    "/shared/submission": submission,
                },
                workspace,
            ),
        ]
        system = (
            "You are a single full-access software agent. Read the complete task "
            "specification, inspect and modify the workspace, run relevant local tests, "
            "and certify your own result. The task grader is hidden and unavailable. "
            "Use /workspace or /shared/workspace in shell commands. Only after completing "
            "and checking the task, write /submission/attestation.json with task_id, "
            "verdict and checklist, then output DONE."
        )
    elif role == "planner":
        roots = {"/task": task}
        tools = [SafeMessageTool(messages, "planner")]
        if planner_reads_workspace:
            roots.update({"/workspace": workspace, "/shared/workspace": workspace})
            tools.insert(0, SafeReadTool(roots, workspace))
        system = (
            "You are the Planner. Read the full specification, identify every explicit "
            "constraint and send a precise implementation plan to the Executor. The full "
            "specification is already in your prompt. Do not attempt to inspect paths that "
            "are not exposed by your tools. You may inspect but never modify an exposed workspace."
        )
    elif role == "executor":
        command_mounts = {
            "/workspace": (workspace, False),
            "/shared/workspace": (workspace, False),
            "/reports": (reports, False),
            "/shared/reports": (reports, False),
        }
        tools = [
            DockerCommandTool(
                workspace,
                image,
                read_only=False,
                timeout_s=command_timeout_s,
                mounts=command_mounts,
            ),
            SafeReadTool(
                {"/workspace": workspace, "/shared/workspace": workspace,
                 "/reports": reports, "/shared/reports": reports}, workspace,
            ),
            SafeWriteTool(
                {"/workspace": workspace, "/shared/workspace": workspace,
                 "/reports": reports, "/shared/reports": reports}, workspace,
            ),
            SafeMessageTool(messages, "executor"),
        ]
        system = (
            "You are the Executor. You only receive the public brief and team messages. "
            "In shell commands, the task workspace is /workspace (also /shared/workspace) "
            "and reports are /reports (also /shared/reports); host paths do not exist in the "
            "sandbox. Inspect and edit only these mounts. Run relevant local tests. Do not "
            "invent success: report failures and use TASK_COMPLETE only when finished."
        )
    elif role == "verifier":
        command_mounts = {
            "/workspace": (workspace, True),
            "/shared/workspace": (workspace, True),
            "/reports": (reports, True),
            "/shared/reports": (reports, True),
            "/task": (task, True),
            "/submission": (submission, False),
            "/shared/submission": (submission, False),
        }
        tools = [
            DockerCommandTool(
                workspace,
                image,
                read_only=True,
                timeout_s=command_timeout_s,
                mounts=command_mounts,
            ),
            SafeReadTool(
                {
                    "/task": task,
                    "/workspace": workspace,
                    "/shared/workspace": workspace,
                    "/reports": reports,
                    "/shared/reports": reports,
                },
                workspace,
            ),
            SafeWriteTool(
                {"/submission": submission, "/shared/submission": submission}, submission
            ),
            SafeMessageTool(messages, "verifier"),
        ]
        system = (
            "You are an independent Verifier. Read the full specification and inspect the "
            "read-only workspace at /workspace (also /shared/workspace). Run local checks, "
            "but never try to edit workspace files. Read-only access is intentional: your "
            "job is to assess the Executor's existing work, not implement the task. Do not "
            "treat read-only access itself as a task failure. Before ending, write only a valid "
            "/submission/attestation.json with verdict pass or fail using the write tool. "
            "Send concrete remediation feedback on failure."
        )
    else:
        raise ValueError(role)
    return RoleConfig(role=role, system_prompt=system, tools=tools)


def _run_phase(
    *,
    role: str,
    adapter: RecordedAdapter,
    prompt: str,
    config: RoleConfig,
    messages: Path,
    logs: Path,
    max_turns: int,
    stop_when: Any | None = None,
) -> list[AgentTurn]:
    if stop_when is None and role == "planner":
        stop_when = lambda: _message_exists(
            messages, sender="planner", recipient="executor"
        )
    effective_adapter = OutcomeStopAdapter(adapter, stop_when) if stop_when else adapter
    loop = AgentLoop(
        role_config=config,
        adapter=effective_adapter,
        messages_dir=str(messages),
        log_dir=str(logs / role),
        max_turns=max_turns,
        lenient_mode=True,
    )
    return loop.run(prompt)


def _message_exists(messages: Path, *, sender: str, recipient: str) -> bool:
    path = messages / "dialogue.jsonl"
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("role") == sender and row.get("to") in {recipient, "all"}:
                return True
    return False


def _message_count(messages: Path, *, sender: str, recipient: str) -> int:
    path = messages / "dialogue.jsonl"
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("role") == sender and row.get("to") in {recipient, "all"}:
                count += 1
    return count


def _logged_command_signals(log_dir: Path) -> dict[str, int]:
    """Read completed agent turns so a controller can interrupt before the next LLM call."""
    successful = 0
    failed = 0
    timed_out = 0
    command_counts: dict[str, int] = {}
    failed_command_counts: dict[str, int] = {}
    turns = 0
    for path in sorted(log_dir.glob("turn_*.json")):
        try:
            turn = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        turns += 1
        calls = list(turn.get("tool_calls") or [])
        results = list(turn.get("tool_results") or [])
        for call, result in zip(calls, results):
            if call.get("name") != "run":
                continue
            exit_code = int(result.get("exit_code", 1))
            successful += int(exit_code == 0)
            failed += int(exit_code != 0)
            timed_out += int(
                exit_code == 124
                or "timed out" in str(result.get("stderr", "")).lower()
            )
            command = " ".join(str(call.get("args", {}).get("cmd", "")).split())
            if command:
                digest = hashlib.sha256(command.encode()).hexdigest()
                command_counts[digest] = command_counts.get(digest, 0) + 1
                if exit_code != 0:
                    failed_command_counts[digest] = (
                        failed_command_counts.get(digest, 0) + 1
                    )
    return {
        "turns": turns,
        "successful_commands": successful,
        "failed_commands": failed,
        "timed_out_commands": timed_out,
        "max_command_repetitions": max(command_counts.values(), default=0),
        "max_failed_command_repetitions": max(
            failed_command_counts.values(), default=0
        ),
    }


def _should_interrupt_for_replan(log_dir: Path, controller: dict[str, Any]) -> bool:
    signals = _logged_command_signals(log_dir)
    return (
        signals["timed_out_commands"]
        >= int(controller.get("replan_interrupt_timeouts", 1))
        or signals["max_failed_command_repetitions"]
        >= int(controller.get("replan_interrupt_failed_command_repetitions", 3))
    )


def _command_failure_evidence(
    turns: list[AgentTurn], *, limit: int = 3
) -> list[dict[str, Any]]:
    """Return a bounded, observable failure summary for the recovery Planner."""
    evidence: list[dict[str, Any]] = []
    for turn in turns:
        for call, result in zip(turn.tool_calls, turn.tool_results):
            if call.get("name") != "run" or int(result.get("exit_code", 1)) == 0:
                continue
            evidence.append({
                "command": " ".join(
                    str(call.get("args", {}).get("cmd", "")).split()
                )[:512],
                "exit_code": int(result.get("exit_code", 1)),
                "stdout_tail": str(result.get("stdout", ""))[-1000:],
                "stderr_tail": str(result.get("stderr", ""))[-1000:],
            })
    return evidence[-limit:]


def _runtime_validator(
    *,
    turns: list[AgentTurn],
    initial_hash: str,
    workspace: Path,
    category: str,
    difficulty: str,
    controller: dict[str, Any],
    max_turns: int | None = None,
) -> dict[str, Any]:
    calls = [call for turn in turns for call in turn.tool_calls]
    results = [result for turn in turns for result in turn.tool_results]
    writes = sum(call.get("name") == "write" for call in calls)
    command_pairs = [
        (call, result) for call, result in zip(calls, results)
        if call.get("name") == "run"
    ]
    commands = [result for _, result in command_pairs]
    successful_commands = sum(int(result.get("exit_code", 1)) == 0 for result in commands)
    failed_commands = sum(int(result.get("exit_code", 1)) != 0 for result in commands)
    timed_out_commands = sum(
        int(result.get("exit_code", 1)) == 124
        or "timed out" in str(result.get("stderr", "")).lower()
        for result in commands
    )
    command_counts: dict[str, int] = {}
    failed_command_counts: dict[str, int] = {}
    for call, _ in command_pairs:
        command = " ".join(str(call.get("args", {}).get("cmd", "")).split())
        if command:
            digest = hashlib.sha256(command.encode()).hexdigest()
            command_counts[digest] = command_counts.get(digest, 0) + 1
    for call, result in command_pairs:
        if int(result.get("exit_code", 1)) == 0:
            continue
        command = " ".join(str(call.get("args", {}).get("cmd", "")).split())
        if command:
            digest = hashlib.sha256(command.encode()).hexdigest()
            failed_command_counts[digest] = failed_command_counts.get(digest, 0) + 1
    repeated_commands = sum(max(0, count - 1) for count in command_counts.values())
    max_command_repetitions = max(command_counts.values(), default=0)
    max_failed_command_repetitions = max(failed_command_counts.values(), default=0)
    current_hash = tree_sha256(workspace)
    changed = current_hash != initial_hash
    score = 1.0
    if not changed:
        score -= 0.55
    if writes == 0:
        score -= 0.15
    if successful_commands == 0:
        score -= 0.25
    if failed_commands:
        score -= min(0.30, 0.10 * failed_commands)
    if category in set(controller["high_risk_categories"]):
        score -= 0.10
    if difficulty in set(controller["high_risk_difficulties"]):
        score -= 0.10
    hard_fail = not changed or (failed_commands > 0 and successful_commands == 0)
    return {
        "schema_version": "general-mas-runtime-validator-v2",
        "workspace_changed": changed,
        "workspace_sha256": current_hash,
        "write_calls": writes,
        "successful_commands": successful_commands,
        "failed_commands": failed_commands,
        "timed_out_commands": timed_out_commands,
        "repeated_commands": repeated_commands,
        "max_command_repetitions": max_command_repetitions,
        "max_failed_command_repetitions": max_failed_command_repetitions,
        "turns_used": len(turns),
        "turn_budget": max_turns,
        "turn_budget_exhausted": bool(
            max_turns is not None
            and len(turns) >= max_turns
            and (not turns or not turns[-1].done)
        ),
        "category": category,
        "difficulty": difficulty,
        "reliability_score": max(0.0, min(1.0, score)),
        "hard_fail": hard_fail,
    }


def _sequential_decision(
    validator: dict[str, Any],
    *,
    consumed_tokens: int,
    controller: dict[str, Any],
) -> dict[str, Any]:
    """Choose a post-execution action from observable state only."""
    token_budget = int(controller.get("sequential_max_total_tokens", 120_000))
    action_space = list(controller.get("sequential_action_space", []))
    required_actions = {"stop", "delegate", "escalate", "review", "human/abstain"}
    if not required_actions.issubset(action_space):
        raise ValueError("sequential_action_space omits a protocol action")

    triggers: list[str] = []
    if int(validator.get("failed_commands", 0)) >= int(
        controller.get("sequential_review_failed_commands", 1)
    ):
        triggers.append("failed_commands")
    if int(validator.get("timed_out_commands", 0)) >= int(
        controller.get("sequential_review_timeouts", 1)
    ):
        triggers.append("command_timeout")
    if int(validator.get("repeated_commands", 0)) >= int(
        controller.get("sequential_review_repeated_commands", 2)
    ):
        triggers.append("repeated_commands")
    if (
        controller.get("sequential_review_on_turn_exhaustion", True)
        and validator.get("turn_budget_exhausted", False)
    ):
        triggers.append("turn_budget_exhausted")

    budget_exhausted = consumed_tokens >= token_budget
    action = "review" if triggers and not budget_exhausted else "stop"
    reason = "runtime_risk" if action == "review" else (
        "token_budget_exhausted" if budget_exhausted else "runtime_checks_clear"
    )
    return {
        "schema_version": "multitown-sequential-decision-v1",
        "policy_source": "frozen_deterministic_runtime_controller",
        "action_space": action_space,
        "state": {
            "workspace_changed": bool(validator.get("workspace_changed", False)),
            "reliability_score": float(validator.get("reliability_score", 0.0)),
            "hard_fail": bool(validator.get("hard_fail", False)),
            "successful_commands": int(validator.get("successful_commands", 0)),
            "failed_commands": int(validator.get("failed_commands", 0)),
            "timed_out_commands": int(validator.get("timed_out_commands", 0)),
            "repeated_commands": int(validator.get("repeated_commands", 0)),
            "turns_used": int(validator.get("turns_used", 0)),
            "turn_budget_exhausted": bool(
                validator.get("turn_budget_exhausted", False)
            ),
            "consumed_tokens": consumed_tokens,
            "remaining_token_budget": max(0, token_budget - consumed_tokens),
        },
        "action": action,
        "reason": reason,
        "triggers": triggers,
    }


def _validate_sequential_controller(controller: dict[str, Any]) -> None:
    action_space = controller.get("sequential_action_space")
    if not isinstance(action_space, list):
        raise ValueError("sequential_action_space must be a list")
    required_actions = {"stop", "delegate", "escalate", "review", "human/abstain"}
    if not required_actions.issubset(action_space):
        raise ValueError("sequential_action_space omits a protocol action")
    for key in (
        "sequential_review_failed_commands",
        "sequential_review_timeouts",
        "sequential_review_repeated_commands",
        "sequential_max_total_tokens",
        "sequential_verifier_turns",
        "sequential_remediation_turns",
    ):
        if int(controller.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be a positive integer")


def _replan_decision(
    validator: dict[str, Any],
    *,
    plan_delivered: bool,
    plan_retry_used: bool = False,
    consumed_tokens: int,
    controller: dict[str, Any],
) -> dict[str, Any]:
    """Escalate only on high-precision execution failures; never inspect grader output."""
    action_space = list(controller.get("replan_action_space", []))
    required_actions = {"stop", "delegate", "escalate", "review", "human/abstain"}
    if not required_actions.issubset(action_space):
        raise ValueError("replan_action_space omits a protocol action")

    triggers: list[str] = []
    if not plan_delivered:
        triggers.append("plan_missing")
    if int(validator.get("timed_out_commands", 0)) >= int(
        controller.get("replan_interrupt_timeouts", 1)
    ):
        triggers.append("command_timeout")
    if int(validator.get("max_failed_command_repetitions", 0)) >= int(
        controller.get("replan_interrupt_failed_command_repetitions", 3)
    ):
        triggers.append("repeated_command")
    workspace_unchanged = not bool(validator.get("workspace_changed", False))
    successful_commands = int(validator.get("successful_commands", 0))
    require_no_success = bool(
        controller.get("replan_workspace_unchanged_requires_no_success", True)
    )
    if workspace_unchanged and (not require_no_success or successful_commands == 0):
        triggers.append(
            "workspace_unchanged_no_success"
            if require_no_success
            else "workspace_unchanged"
        )
    if (
        successful_commands == 0
        and int(validator.get("failed_commands", 0))
        >= int(controller.get("replan_failed_without_success", 2))
    ):
        triggers.append("all_commands_failed")

    token_budget = int(controller.get("replan_max_total_tokens", 90_000))
    budget_exhausted = consumed_tokens >= token_budget
    action = "escalate" if triggers and not budget_exhausted else "stop"
    reason = (
        "runtime_failure"
        if action == "escalate"
        else ("token_budget_exhausted" if budget_exhausted else "runtime_checks_clear")
    )
    return {
        "schema_version": "multitown-replan-decision-v1",
        "policy_source": "frozen_deterministic_replan_controller",
        "action_space": action_space,
        "state": {
            "plan_delivered": plan_delivered,
            "plan_retry_used": plan_retry_used,
            "workspace_changed": bool(validator.get("workspace_changed", False)),
            "successful_commands": int(validator.get("successful_commands", 0)),
            "failed_commands": int(validator.get("failed_commands", 0)),
            "timed_out_commands": int(validator.get("timed_out_commands", 0)),
            "max_failed_command_repetitions": int(
                validator.get("max_failed_command_repetitions", 0)
            ),
            "turns_used": int(validator.get("turns_used", 0)),
            "consumed_tokens": consumed_tokens,
            "remaining_token_budget": max(0, token_budget - consumed_tokens),
        },
        "action": action,
        "reason": reason,
        "triggers": triggers,
    }


def _validate_replan_controller(controller: dict[str, Any]) -> None:
    action_space = controller.get("replan_action_space")
    if not isinstance(action_space, list):
        raise ValueError("replan_action_space must be a list")
    required_actions = {"stop", "delegate", "escalate", "review", "human/abstain"}
    if not required_actions.issubset(action_space):
        raise ValueError("replan_action_space omits a protocol action")
    for key in (
        "replan_command_timeout_s",
        "replan_interrupt_timeouts",
        "replan_interrupt_failed_command_repetitions",
        "replan_failed_without_success",
        "replan_max_total_tokens",
        "replan_planner_turns",
        "replan_executor_turns",
    ):
        if int(controller.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(
        controller.get("replan_workspace_unchanged_requires_no_success", True), bool
    ):
        raise ValueError(
            "replan_workspace_unchanged_requires_no_success must be a boolean"
        )


def _attestation(submission: Path) -> dict[str, Any] | None:
    path = submission / "attestation.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _is_high_risk(row: dict[str, Any], controller: dict[str, Any]) -> bool:
    return (
        row.get("category") in set(controller["high_risk_categories"])
        or row.get("difficulty") in set(controller["high_risk_difficulties"])
    )


def _allow_initial_early_stop(
    validator: dict[str, Any], row: dict[str, Any], controller: dict[str, Any]
) -> bool:
    require_command = bool(controller["require_successful_command_for_early_stop"])
    return (
        not _is_high_risk(row, controller)
        and not validator["hard_fail"]
        and validator["reliability_score"] >= float(controller["early_stop_threshold"])
        and (not require_command or validator["successful_commands"] > 0)
    )


def _controller_attestation(
    submission: Path,
    task_id: str,
    reason: str,
    *,
    source: str = "deterministic_a8_runtime_validator",
) -> None:
    write_json(submission / "attestation.json", {
        "task_id": task_id,
        "verdict": "pass",
        "checklist": [],
        "source": source,
        "reason": reason,
    })


def _snapshot_workspace(workspace: Path, snapshot: Path) -> None:
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace, snapshot, symlinks=True)


def _restore_workspace(snapshot: Path, workspace: Path) -> None:
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(snapshot, workspace, symlinks=True)


def _snapshot_candidate(workspace: Path, reports: Path, snapshot: Path) -> None:
    """Snapshot every grader-visible surface that a worker can modify."""
    _snapshot_workspace(workspace, snapshot / "workspace")
    _snapshot_workspace(reports, snapshot / "reports")


def _restore_candidate(snapshot: Path, workspace: Path, reports: Path) -> None:
    """Restore without leaving post-snapshot report artifacts behind."""
    _restore_workspace(snapshot / "workspace", workspace)
    _restore_workspace(snapshot / "reports", reports)


def _select_failed_review_candidate(
    *,
    initial: dict[str, Any],
    post: dict[str, Any],
    initial_snapshot: Path,
    planned_snapshot: Path,
    workspace: Path,
    controller: dict[str, Any],
) -> str:
    """Select a candidate without consulting the hidden benchmark grader."""
    preserve = bool(controller.get("preserve_initial_candidate", False))
    if preserve and float(initial["reliability_score"]) >= float(post["reliability_score"]):
        _restore_workspace(initial_snapshot, workspace)
        return "initial"
    _restore_workspace(planned_snapshot, workspace)
    return "planned"


def _prompts(task_id: str, spec: str, brief: str) -> dict[str, str]:
    return {
        "planner": (
            f"Task: {task_id}\n\nFULL SPECIFICATION:\n{spec}\n\n"
            "The specification is already included; do not search the workspace unless a "
            "workspace read tool is explicitly available. Create a complete plan and make "
            "send_message(to='executor', content=...) your first action. "
            "Include exact values, edge cases and validation commands, then output DONE."
        ),
        "executor": (
            f"Task: {task_id}\n\nPUBLIC BRIEF:\n{brief}\n\n"
            "Use /workspace or /shared/workspace in shell commands, never a host path. Read "
            "any Planner message available, inspect the workspace, implement the task, "
            "run useful local checks, notify verifier if present, then output TASK_COMPLETE."
        ),
        "initial_executor": (
            f"Task: {task_id}\n\nPUBLIC BRIEF:\n{brief}\n\n"
            "No Planner is active yet. Use /workspace or /shared/workspace in shell commands. "
            "Inspect the workspace, implement the best complete "
            "solution you can from the brief, run useful local checks, then output TASK_COMPLETE."
        ),
        "solo": (
            f"Task: {task_id}\n\nFULL SPECIFICATION:\n{spec}\n\nPUBLIC BRIEF:\n{brief}\n\n"
            "You have full specification and workspace access. Inspect the existing files, "
            "implement every requirement, and run useful local checks. Do not inspect or "
            "guess the hidden grader. When the work is complete, write "
            "/submission/attestation.json as valid JSON with task_id, verdict and checklist, "
            "then output DONE."
        ),
        "remediation": (
            f"Task: {task_id}\n\nPUBLIC BRIEF:\n{brief}\n\n"
            "New specialist feedback is available in team messages. Use /workspace or "
            "/shared/workspace in shell commands. Make only the required "
            "targeted fixes, rerun relevant checks, then output TASK_COMPLETE."
        ),
        "planner_retry": (
            f"Task: {task_id}\n\nFULL SPECIFICATION:\n{spec}\n\n"
            "Your previous attempt did not deliver a plan. Do not inspect unavailable paths. "
            "Make send_message(to='executor', content=...) your first and mandatory action. "
            "Send a concise recovery-ready plan with exact requirements and checks, then "
            "output DONE."
        ),
        "replan": (
            f"Task: {task_id}\n\nFULL SPECIFICATION:\n{spec}\n\n"
            "The first Executor attempt hit a controller-observed runtime failure. You now "
            "have read-only workspace access. Inspect only what is necessary, infer the "
            "failure from the current files and command evidence in the prompt, and make "
            "send_message(to='executor', content=...) your first mandatory action. Send one "
            "short, concrete recovery plan; do not implement or certify the task."
        ),
        "recovery": (
            f"Task: {task_id}\n\nPUBLIC BRIEF:\n{brief}\n\n"
            "A strong Planner has inspected the failed attempt and sent a recovery plan. "
            "Read the newest Planner message, make only targeted changes in /workspace, "
            "avoid repeating failed commands, run one focused local check, and output "
            "TASK_COMPLETE."
        ),
        "verifier": (
            f"Task: {task_id}\n\nFULL SPECIFICATION:\n{spec}\n\n"
            "Independently inspect the read-only /workspace against every requirement. Run "
            "local checks, but do not modify workspace files. "
            "Write /submission/attestation.json as valid JSON with task_id, verdict, and "
            "checklist. If failing, message the Executor with exact fixes. Output DONE."
        ),
    }


def _usage(adapters: list[RecordedAdapter]) -> dict[str, int]:
    values = [adapter.get_usage() for adapter in adapters]
    return {
        "input_tokens": sum(int(value.get("input_tokens", 0)) for value in values),
        "output_tokens": sum(int(value.get("output_tokens", 0)) for value in values),
        "total_tokens": sum(int(value.get("total_tokens", 0)) for value in values),
    }


def run_task(
    *,
    method: str,
    row: dict[str, Any],
    team_root: Path,
    run_dir: Path,
    request_log: Path,
    image: str,
    controller: dict[str, Any],
    strong: dict[str, Any],
    weak: dict[str, Any],
    max_tokens: int,
    temperature: float,
    policy: dict[str, Any] | None = None,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    reported_method = method
    execution_method = method
    if method == "MTSelector":
        if policy is None:
            raise ValueError("MTSelector requires a frozen policy")
        execution_method = select_method(policy, row)
    started = time.perf_counter()
    meta = stage_task(row, team_root, run_dir)
    run_setup_in_sandbox(row=row, team_root=team_root, run_dir=run_dir, image=image)
    workspace = run_dir / "workspace"
    reports = run_dir / "reports"
    messages = run_dir / "messages"
    submission = run_dir / "submission"
    task = run_dir / "task"
    logs = run_dir / "logs"
    meta["initial_workspace_sha256"] = tree_sha256(workspace)
    write_json(run_dir / "run_meta.json", meta)
    spec = (task / "spec.md").read_text(encoding="utf-8")
    brief = (task / "brief.md").read_text(encoding="utf-8")
    prompts = _prompts(str(row["task_id"]), spec, brief)

    adapters: list[RecordedAdapter] = []

    def adapter(role: str, tier: dict[str, Any]) -> RecordedAdapter:
        value = RecordedAdapter(
            role=role,
            task_id=str(row["task_id"]),
            method=method,
            request_log=request_log,
            endpoint=str(tier["endpoint"]),
            api_key=str(tier["api_key"]),
            model=str(tier["model"]),
            max_tokens=max_tokens,
            temperature=temperature,
            sampling_seed=sampling_seed,
        )
        adapters.append(value)
        return value

    role_counts = {"solo": 0, "planner": 0, "executor": 0, "verifier": 0}
    validators: list[dict[str, Any]] = []
    decision_trace: list[dict[str, Any]] = []
    route = ""
    replan_prefix_snapshot: Path | None = None

    if execution_method == "Solo":
        solo = adapter("solo", strong)
        role_counts["solo"] = 1
        _run_phase(
            role="solo", adapter=solo, prompt=prompts["solo"],
            config=_phase_config(
                "solo", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "solo", max_turns=20,
        )
        route = "single_strong_full_access"
    elif execution_method in {"PlanExecute", "MTSequential", "MTReplan"}:
        # TeamBench's official no-verifier ablation: a strong Planner transfers
        # the full-spec plan to a weak, brief-only Executor. There is no
        # independent review, so the controller supplies the protocol-required
        # attestation after execution without consulting the hidden grader.
        planner = adapter("planner", strong)
        executor = adapter("executor", weak)
        role_counts.update(planner=1, executor=1)
        _run_phase(
            role="planner", adapter=planner, prompt=prompts["planner"],
            config=_phase_config(
                "planner", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "planning", max_turns=6,
        )
        plan_delivered = _message_exists(
            messages, sender="planner", recipient="executor"
        )
        plan_retry_used = False
        if execution_method == "MTReplan" and not plan_delivered:
            plan_retry_used = True
            prior_plan_count = _message_count(
                messages, sender="planner", recipient="executor"
            )
            role_counts["planner"] += 1
            _run_phase(
                role="planner",
                adapter=planner,
                prompt=prompts["planner_retry"],
                config=_phase_config(
                    "planner",
                    task=task,
                    workspace=workspace,
                    reports=reports,
                    messages=messages,
                    submission=submission,
                    image=image,
                ),
                messages=messages,
                logs=logs / "planning_retry",
                max_turns=2,
                stop_when=lambda: _message_count(
                    messages, sender="planner", recipient="executor"
                )
                > prior_plan_count,
            )
            plan_delivered = _message_count(
                messages, sender="planner", recipient="executor"
            ) > prior_plan_count
        execution_logs = logs / "execution"
        command_timeout_s = (
            int(controller.get("replan_command_timeout_s", 30))
            if execution_method == "MTReplan"
            else 120
        )
        executor_turns = _run_phase(
            role="executor", adapter=executor, prompt=prompts["executor"],
            config=_phase_config(
                "executor", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
                command_timeout_s=command_timeout_s,
            ),
            messages=messages,
            logs=execution_logs,
            max_turns=12,
            stop_when=(
                lambda: _should_interrupt_for_replan(
                    execution_logs / "executor", controller
                )
                if execution_method == "MTReplan"
                else False
            ),
        )
        execution_validator = _runtime_validator(
            turns=executor_turns,
            initial_hash=meta["initial_workspace_sha256"],
            workspace=workspace,
            category=str(row.get("category") or "Other"),
            difficulty=str(row.get("difficulty") or "unknown"),
            controller=controller,
            max_turns=12,
        )
        validators.append(execution_validator)
        if execution_method == "PlanExecute":
            _controller_attestation(
                submission,
                str(row["task_id"]),
                "fixed_plan_execute_without_independent_review",
                source="fixed_strategy_protocol_controller",
            )
            route = "fixed_plan_execute"
        elif execution_method == "MTSequential":
            candidates = run_dir / "candidates"
            plan_execute_snapshot = candidates / "plan_execute"
            _snapshot_workspace(workspace, plan_execute_snapshot)
            decision = _sequential_decision(
                execution_validator,
                consumed_tokens=_usage(adapters)["total_tokens"],
                controller=controller,
            )
            decision_trace.append({"step": 0, "phase": "post_execution", **decision})
            if decision["action"] == "stop":
                _controller_attestation(
                    submission,
                    str(row["task_id"]),
                    f"mt_sequential_stop:{decision['reason']}",
                    source="frozen_deterministic_runtime_controller",
                )
                route = f"sequential_stop:{decision['reason']}"
            else:
                verifier = adapter("verifier", strong)
                role_counts["verifier"] = 1
                (submission / "attestation.json").unlink(missing_ok=True)
                _run_phase(
                    role="verifier", adapter=verifier, prompt=prompts["verifier"],
                    config=_phase_config(
                        "verifier", task=task, workspace=workspace, reports=reports,
                        messages=messages, submission=submission, image=image,
                    ),
                    messages=messages, logs=logs / "sequential_review",
                    max_turns=int(controller.get("sequential_verifier_turns", 8)),
                    stop_when=lambda: _attestation(submission) is not None,
                )
                att = _attestation(submission)
                has_feedback = _message_exists(
                    messages, sender="verifier", recipient="executor"
                )
                consumed = _usage(adapters)["total_tokens"]
                token_budget = int(
                    controller.get("sequential_max_total_tokens", 120_000)
                )
                should_remediate = bool(
                    att
                    and att.get("verdict") == "fail"
                    and has_feedback
                    and consumed < token_budget
                )
                followup_action = "delegate" if should_remediate else "stop"
                followup_reason = (
                    "verifier_feedback"
                    if should_remediate
                    else (
                        "verifier_passed"
                        if att and att.get("verdict") == "pass"
                        else "review_failed_or_budget_exhausted"
                    )
                )
                decision_trace.append({
                    "schema_version": "multitown-sequential-decision-v1",
                    "step": 1,
                    "phase": "post_review",
                    "policy_source": "frozen_deterministic_runtime_controller",
                    "action_space": list(
                        controller.get("sequential_action_space", [])
                    ),
                    "state": {
                        "verifier_verdict": att.get("verdict") if att else "missing",
                        "verifier_feedback": has_feedback,
                        "consumed_tokens": consumed,
                        "remaining_token_budget": max(0, token_budget - consumed),
                    },
                    "action": followup_action,
                    "reason": followup_reason,
                    "triggers": [followup_reason],
                })
                if should_remediate:
                    role_counts["executor"] += 1
                    before_remediation = tree_sha256(workspace)
                    remediation_turns = _run_phase(
                        role="executor", adapter=executor,
                        prompt=prompts["remediation"],
                        config=_phase_config(
                            "executor", task=task, workspace=workspace,
                            reports=reports, messages=messages,
                            submission=submission, image=image,
                        ),
                        messages=messages, logs=logs / "sequential_remediation",
                        max_turns=int(
                            controller.get("sequential_remediation_turns", 6)
                        ),
                    )
                    remediation_validator = _runtime_validator(
                        turns=remediation_turns,
                        initial_hash=before_remediation,
                        workspace=workspace,
                        category=str(row.get("category") or "Other"),
                        difficulty=str(row.get("difficulty") or "unknown"),
                        controller=controller,
                        max_turns=int(
                            controller.get("sequential_remediation_turns", 6)
                        ),
                    )
                    validators.append(remediation_validator)
                    if remediation_validator["workspace_changed"]:
                        selected = "remediated"
                    else:
                        _restore_workspace(plan_execute_snapshot, workspace)
                        selected = "plan_execute_rollback"
                    _controller_attestation(
                        submission,
                        str(row["task_id"]),
                        f"mt_sequential_{selected}",
                        source="frozen_deterministic_runtime_controller",
                    )
                    route = f"sequential_review:{selected}"
                elif att and att.get("verdict") == "pass":
                    route = "sequential_review:verifier_passed"
                else:
                    _restore_workspace(plan_execute_snapshot, workspace)
                    _controller_attestation(
                        submission,
                        str(row["task_id"]),
                        "mt_sequential_review_failed_plan_execute_rollback",
                        source="frozen_deterministic_runtime_controller",
                    )
                    route = "sequential_review:plan_execute_rollback"
        else:
            candidates = run_dir / "candidates"
            plan_execute_snapshot = candidates / "plan_execute"
            _snapshot_candidate(workspace, reports, plan_execute_snapshot)
            replan_prefix_snapshot = plan_execute_snapshot
            decision = _replan_decision(
                execution_validator,
                plan_delivered=plan_delivered,
                plan_retry_used=plan_retry_used,
                consumed_tokens=_usage(adapters)["total_tokens"],
                controller=controller,
            )
            decision_trace.append({"step": 0, "phase": "post_execution", **decision})
            if decision["action"] == "stop":
                _controller_attestation(
                    submission,
                    str(row["task_id"]),
                    f"mt_replan_stop:{decision['reason']}",
                    source="frozen_deterministic_replan_controller",
                )
                route = f"replan_stop:{decision['reason']}"
            else:
                prior_plan_count = _message_count(
                    messages, sender="planner", recipient="executor"
                )
                incident = json.dumps(
                    {
                        "triggers": decision["triggers"],
                        "runtime": execution_validator,
                        "failure_evidence": _command_failure_evidence(executor_turns),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                role_counts["planner"] += 1
                _run_phase(
                    role="planner",
                    adapter=planner,
                    prompt=f"{prompts['replan']}\n\nCONTROLLER OBSERVATION:\n{incident}",
                    config=_phase_config(
                        "planner",
                        task=task,
                        workspace=workspace,
                        reports=reports,
                        messages=messages,
                        submission=submission,
                        image=image,
                        planner_reads_workspace=True,
                        command_timeout_s=command_timeout_s,
                    ),
                    messages=messages,
                    logs=logs / "replanning",
                    max_turns=int(controller.get("replan_planner_turns", 3)),
                    stop_when=lambda: _message_count(
                        messages, sender="planner", recipient="executor"
                    )
                    > prior_plan_count,
                )
                recovery_plan_delivered = _message_count(
                    messages, sender="planner", recipient="executor"
                ) > prior_plan_count
                consumed = _usage(adapters)["total_tokens"]
                token_budget = int(controller.get("replan_max_total_tokens", 90_000))
                should_delegate = recovery_plan_delivered and consumed < token_budget
                decision_trace.append({
                    "schema_version": "multitown-replan-decision-v1",
                    "step": 1,
                    "phase": "post_replan",
                    "policy_source": "frozen_deterministic_replan_controller",
                    "action_space": list(controller.get("replan_action_space", [])),
                    "state": {
                        "recovery_plan_delivered": recovery_plan_delivered,
                        "consumed_tokens": consumed,
                        "remaining_token_budget": max(0, token_budget - consumed),
                    },
                    "action": "delegate" if should_delegate else "stop",
                    "reason": (
                        "recovery_plan_ready"
                        if should_delegate
                        else "replan_failed_or_budget_exhausted"
                    ),
                    "triggers": decision["triggers"],
                })
                if not should_delegate:
                    _restore_candidate(plan_execute_snapshot, workspace, reports)
                    selected = "plan_execute_rollback"
                else:
                    role_counts["executor"] += 1
                    before_recovery = tree_sha256(workspace)
                    recovery_logs = logs / "recovery"
                    recovery_turns = _run_phase(
                        role="executor",
                        adapter=executor,
                        prompt=prompts["recovery"],
                        config=_phase_config(
                            "executor",
                            task=task,
                            workspace=workspace,
                            reports=reports,
                            messages=messages,
                            submission=submission,
                            image=image,
                            command_timeout_s=command_timeout_s,
                        ),
                        messages=messages,
                        logs=recovery_logs,
                        max_turns=int(controller.get("replan_executor_turns", 6)),
                        stop_when=lambda: _should_interrupt_for_replan(
                            recovery_logs / "executor", controller
                        ),
                    )
                    recovery_validator = _runtime_validator(
                        turns=recovery_turns,
                        initial_hash=before_recovery,
                        workspace=workspace,
                        category=str(row.get("category") or "Other"),
                        difficulty=str(row.get("difficulty") or "unknown"),
                        controller=controller,
                        max_turns=int(controller.get("replan_executor_turns", 6)),
                    )
                    validators.append(recovery_validator)
                    keep_recovery = bool(
                        recovery_validator["workspace_changed"]
                        and recovery_validator["reliability_score"]
                        >= execution_validator["reliability_score"]
                    )
                    if keep_recovery:
                        selected = "recovered"
                    else:
                        _restore_candidate(plan_execute_snapshot, workspace, reports)
                        selected = "plan_execute_rollback"
                _controller_attestation(
                    submission,
                    str(row["task_id"]),
                    f"mt_replan_{selected}",
                    source="frozen_deterministic_replan_controller",
                )
                decision_trace.append({
                    "schema_version": "multitown-replan-decision-v1",
                    "step": 2,
                    "phase": "post_recovery",
                    "policy_source": "frozen_deterministic_replan_controller",
                    "action_space": list(controller.get("replan_action_space", [])),
                    "state": {"selected_candidate": selected},
                    "action": "stop",
                    "reason": selected,
                    "triggers": [],
                })
                route = f"replan:{selected}"
    elif execution_method == "ExecuteReview":
        # TeamBench's official no-planner ablation: a weak Executor works from
        # the public brief, then an independent strong Verifier reads the full
        # specification and certifies the result. No remediation loop is added.
        executor = adapter("executor", weak)
        verifier = adapter("verifier", strong)
        role_counts.update(executor=1, verifier=1)
        executor_turns = _run_phase(
            role="executor", adapter=executor, prompt=prompts["initial_executor"],
            config=_phase_config(
                "executor", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "execution", max_turns=12,
        )
        validators.append(_runtime_validator(
            turns=executor_turns,
            initial_hash=meta["initial_workspace_sha256"],
            workspace=workspace,
            category=str(row.get("category") or "Other"),
            difficulty=str(row.get("difficulty") or "unknown"),
            controller=controller,
        ))
        (submission / "attestation.json").unlink(missing_ok=True)
        _run_phase(
            role="verifier", adapter=verifier, prompt=prompts["verifier"],
            config=_phase_config(
                "verifier", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "review", max_turns=8,
            stop_when=lambda: _attestation(submission) is not None,
        )
        route = "fixed_execute_review"
    elif execution_method == "A4":
        planner = adapter("planner", strong)
        executor = adapter("executor", weak)
        verifier = adapter("verifier", strong)
        role_counts.update(planner=1, executor=1, verifier=1)
        _run_phase(
            role="planner", adapter=planner, prompt=prompts["planner"],
            config=_phase_config(
                "planner", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "planning", max_turns=6,
        )
        executor_turns = _run_phase(
            role="executor", adapter=executor, prompt=prompts["executor"],
            config=_phase_config(
                "executor", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "execution", max_turns=12,
        )
        validators.append(_runtime_validator(
            turns=executor_turns,
            initial_hash=meta["initial_workspace_sha256"],
            workspace=workspace,
            category=str(row.get("category") or "Other"),
            difficulty=str(row.get("difficulty") or "unknown"),
            controller=controller,
        ))
        (submission / "attestation.json").unlink(missing_ok=True)
        _run_phase(
            role="verifier", adapter=verifier, prompt=prompts["verifier"],
            config=_phase_config(
                "verifier", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "verification", max_turns=8,
            stop_when=lambda: _attestation(submission) is not None,
        )
        att = _attestation(submission)
        if (
            (not att or att.get("verdict") != "pass")
            and int(controller.get("a4_max_remediation_loops", 1)) > 0
        ):
            role_counts["executor"] += 1
            _run_phase(
                role="executor", adapter=executor, prompt=prompts["remediation"],
                config=_phase_config(
                    "executor", task=task, workspace=workspace, reports=reports,
                    messages=messages, submission=submission, image=image,
                ),
                messages=messages, logs=logs / "remediation", max_turns=6,
            )
            role_counts["verifier"] += 1
            (submission / "attestation.json").unlink(missing_ok=True)
            _run_phase(
                role="verifier", adapter=verifier, prompt=prompts["verifier"],
                config=_phase_config(
                    "verifier", task=task, workspace=workspace, reports=reports,
                    messages=messages, submission=submission, image=image,
                ),
                messages=messages, logs=logs / "reverification", max_turns=6,
                stop_when=lambda: _attestation(submission) is not None,
            )
            route = "fixed_full_with_remediation"
        else:
            route = "fixed_full"
    elif execution_method == "A8":
        executor = adapter("executor", weak)
        role_counts["executor"] = 1
        initial_turns = _run_phase(
            role="executor", adapter=executor, prompt=prompts["initial_executor"],
            config=_phase_config(
                "executor", task=task, workspace=workspace, reports=reports,
                messages=messages, submission=submission, image=image,
            ),
            messages=messages, logs=logs / "initial_execution", max_turns=8,
        )
        initial = _runtime_validator(
            turns=initial_turns,
            initial_hash=meta["initial_workspace_sha256"],
            workspace=workspace,
            category=str(row.get("category") or "Other"),
            difficulty=str(row.get("difficulty") or "unknown"),
            controller=controller,
        )
        validators.append(initial)
        candidates = run_dir / "candidates"
        initial_snapshot = candidates / "initial"
        _snapshot_workspace(workspace, initial_snapshot)
        early = _allow_initial_early_stop(initial, row, controller)
        if early:
            _controller_attestation(submission, str(row["task_id"]), "initial_runtime_checks_passed")
            route = "weak_early_stop"
        else:
            planner = adapter("planner", strong)
            role_counts["planner"] = 1
            _run_phase(
                role="planner", adapter=planner, prompt=prompts["planner"],
                config=_phase_config(
                    "planner", task=task, workspace=workspace, reports=reports,
                    messages=messages, submission=submission, image=image,
                    planner_reads_workspace=True,
                ),
                messages=messages, logs=logs / "selective_planning", max_turns=6,
            )
            role_counts["executor"] += 1
            before_remediation = tree_sha256(workspace)
            remediation_turns = _run_phase(
                role="executor", adapter=executor, prompt=prompts["remediation"],
                config=_phase_config(
                    "executor", task=task, workspace=workspace, reports=reports,
                    messages=messages, submission=submission, image=image,
                ),
                messages=messages, logs=logs / "planned_execution", max_turns=8,
            )
            post = _runtime_validator(
                turns=remediation_turns,
                initial_hash=before_remediation,
                workspace=workspace,
                category=str(row.get("category") or "Other"),
                difficulty=str(row.get("difficulty") or "unknown"),
                controller=controller,
            )
            validators.append(post)
            planned_snapshot = candidates / "planned"
            _snapshot_workspace(workspace, planned_snapshot)
            high_risk = _is_high_risk(row, controller)
            skip_verifier = (
                not post["hard_fail"]
                and not high_risk
                and post["reliability_score"]
                >= float(controller["post_plan_verifier_skip_threshold"])
            )
            if skip_verifier:
                _controller_attestation(submission, str(row["task_id"]), "post_plan_runtime_checks_passed")
                route = "planner_executor_stop"
            else:
                verifier = adapter("verifier", strong)
                role_counts["verifier"] = 1
                (submission / "attestation.json").unlink(missing_ok=True)
                _run_phase(
                    role="verifier", adapter=verifier, prompt=prompts["verifier"],
                    config=_phase_config(
                        "verifier", task=task, workspace=workspace, reports=reports,
                        messages=messages, submission=submission, image=image,
                    ),
                    messages=messages, logs=logs / "conditional_verification", max_turns=8,
                    stop_when=lambda: _attestation(submission) is not None,
                )
                att = _attestation(submission)
                has_feedback = _message_exists(
                    messages, sender="verifier", recipient="executor"
                )
                if (
                    att
                    and att.get("verdict") == "fail"
                    and has_feedback
                    and controller["max_remediation_loops"]
                    and controller.get("guided_remediation_without_reverification", False)
                ):
                    role_counts["executor"] += 1
                    before_guided_remediation = tree_sha256(workspace)
                    guided_turns = _run_phase(
                        role="executor", adapter=executor, prompt=prompts["remediation"],
                        config=_phase_config(
                            "executor", task=task, workspace=workspace, reports=reports,
                            messages=messages, submission=submission, image=image,
                        ),
                        messages=messages, logs=logs / "conditional_remediation", max_turns=6,
                    )
                    guided = _runtime_validator(
                        turns=guided_turns,
                        initial_hash=before_guided_remediation,
                        workspace=workspace,
                        category=str(row.get("category") or "Other"),
                        difficulty=str(row.get("difficulty") or "unknown"),
                        controller=controller,
                    )
                    validators.append(guided)
                    prior_reliability = max(
                        float(initial["reliability_score"]),
                        float(post["reliability_score"]),
                    )
                    if (
                        not guided["hard_fail"]
                        and float(guided["reliability_score"]) >= prior_reliability
                    ):
                        _controller_attestation(
                            submission,
                            str(row["task_id"]),
                            "strong_verifier_guided_remediation_runtime_checks_passed",
                        )
                        route = "strong_review_guided_remediation"
                    else:
                        selected = _select_failed_review_candidate(
                            initial=initial,
                            post=post,
                            initial_snapshot=initial_snapshot,
                            planned_snapshot=planned_snapshot,
                            workspace=workspace,
                            controller=controller,
                        )
                        route = f"strong_review_failed_fallback_{selected}"
                elif not att or att.get("verdict") != "pass":
                    selected = _select_failed_review_candidate(
                        initial=initial,
                        post=post,
                        initial_snapshot=initial_snapshot,
                        planned_snapshot=planned_snapshot,
                        workspace=workspace,
                        controller=controller,
                    )
                    route = f"strong_review_failed_fallback_{selected}"
                else:
                    route = "strong_review"
    else:
        raise ValueError(execution_method)

    if reported_method == "MTSelector":
        route = f"mt_selector:{execution_method}:{route}"

    if _attestation(submission) is None:
        write_json(submission / "attestation.json", {
            "task_id": row["task_id"], "verdict": "fail", "checklist": [],
            "source": "fail_closed_missing_attestation",
        })
    selected_workspace_sha256 = tree_sha256(workspace)
    selected_reports_sha256 = tree_sha256(reports)
    score = grade_in_sandbox(
        row=row, team_root=team_root, run_dir=run_dir, image=image
    )
    policy_latency_s = time.perf_counter() - started
    shadow_plan_execute: dict[str, Any] | None = None
    if execution_method == "MTReplan" and replan_prefix_snapshot is not None:
        # Score the exact pre-replan candidate after routing has ended. The
        # hidden result is evaluation-only and can never influence an action.
        shadow_run = run_dir / "counterfactuals" / "plan_execute_prefix"
        _snapshot_candidate(
            replan_prefix_snapshot / "workspace",
            replan_prefix_snapshot / "reports",
            shadow_run,
        )
        shadow_submission = shadow_run / "submission"
        _controller_attestation(
            shadow_submission,
            str(row["task_id"]),
            "fixed_plan_execute_without_independent_review",
            source="fixed_strategy_protocol_controller",
        )
        shadow_workspace_hash = tree_sha256(shadow_run / "workspace")
        shadow_reports_hash = tree_sha256(shadow_run / "reports")
        shadow_started = time.perf_counter()
        shadow_score = grade_in_sandbox(
            row=row, team_root=team_root, run_dir=shadow_run, image=image
        )
        shadow_plan_execute = {
            "passed": bool(shadow_score.get("pass", False)),
            "partial_score": float(
                shadow_score.get("secondary", {}).get(
                    "partial_score", 1.0 if shadow_score.get("pass") else 0.0
                )
            ),
            "failure_modes": shadow_score.get("failure_modes", []),
            "workspace_sha256": shadow_workspace_hash,
            "reports_sha256": shadow_reports_hash,
            "grader_latency_s": time.perf_counter() - shadow_started,
            "policy_input": False,
        }
    usage = _usage(adapters)
    result = {
        "schema_version": "general-mas-teambench-result-v1",
        "method": reported_method,
        "task_id": row["task_id"],
        "source_task": row["source_task"],
        "split": row["split"],
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
        "passed": bool(score.get("pass", False)),
        "partial_score": float(score.get("secondary", {}).get("partial_score", 1.0 if score.get("pass") else 0.0)),
        "failure_modes": score.get("failure_modes", []),
        "route": route,
        "role_activations": role_counts,
        "validators": validators,
        "decision_trace": decision_trace,
        **usage,
        "latency_s": policy_latency_s,
        "selected_workspace_sha256": selected_workspace_sha256,
        "selected_reports_sha256": selected_reports_sha256,
        "final_workspace_sha256": tree_sha256(workspace),
        "final_reports_sha256": tree_sha256(reports),
        "request_errors": 0,
    }
    if shadow_plan_execute is not None:
        result["shadow_plan_execute"] = shadow_plan_execute
    if reported_method == "MTSelector":
        result["selected_strategy"] = execution_method
    write_json(run_dir / "result.json", result)
    return result


def _git_state(root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False
    )
    return {
        "revision": result.stdout.strip() if result.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _docker_image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"Docker image unavailable: {image}: {result.stderr.strip()}")
    return result.stdout.strip()


def _archive_failed_attempt(task_run: Path, failed_root: Path) -> None:
    if not task_run.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = failed_root / f"{task_run.name}-{stamp}"
    counter = 1
    while target.exists():
        target = failed_root / f"{task_run.name}-{stamp}-{counter}"
        counter += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_run), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=(
            "Solo", "PlanExecute", "ExecuteReview", "A4", "A8",
            "MTSelector", "MTSequential", "MTReplan",
        ),
        required=True,
    )
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-file", type=Path, default=Path("benchmarks/teambench-v1/split.json"))
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--task", action="append")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--docker-image", default="general-mas-runner:0.1")
    parser.add_argument("--strong-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--strong-api-key", default="local")
    parser.add_argument("--strong-model", default="qwen-game")
    parser.add_argument("--weak-endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--weak-api-key", default="local")
    parser.add_argument("--weak-model", default="qwen-mm-backup")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    available, detail = docker_available()
    if not available:
        raise SystemExit(f"Docker sandbox unavailable: {detail}")

    project_root = args.project_root.resolve()
    output = args.output_dir.resolve()
    split_path = args.split_file if args.split_file.is_absolute() else project_root / args.split_file
    split = read_json(split_path.resolve())
    controller = read_json(args.controller_config.resolve()) if args.controller_config else DEFAULT_CONTROLLER
    policy = read_json(args.policy_config.resolve()) if args.policy_config else None
    if args.method == "MTSelector" and policy is None:
        parser.error("--policy-config is required for MTSelector")
    if args.method != "MTSelector" and policy is not None:
        parser.error("--policy-config is only valid for MTSelector")
    if args.method == "MTSequential":
        _validate_sequential_controller(controller)
    if args.method == "MTReplan":
        _validate_replan_controller(controller)
    rows = [row for row in split["rows"] if row["split"] == args.split]
    if args.task:
        selected = set(args.task)
        rows = [row for row in rows if row["task_id"] in selected]
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]
    if args.seed_override is not None:
        if args.seed_override < 0:
            parser.error("--seed-override must be non-negative")
        rows = [{**row, "seed": args.seed_override} for row in rows]
    if not rows:
        raise SystemExit("no tasks selected")

    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "general-mas-teambench-run-v1",
        "method": args.method,
        "split": args.split,
        "tasks": [row["task_id"] for row in rows],
        "split_sha256": split["split_sha256"],
        "source": _git_state(project_root),
        "docker_image": args.docker_image,
        "docker_image_id": _docker_image_id(args.docker_image),
        "controller": controller,
        "strong": {"endpoint": args.strong_endpoint, "model": args.strong_model},
        "weak": {"endpoint": args.weak_endpoint, "model": args.weak_model},
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "runner_source": _git_state(Path(__file__).resolve().parents[2]),
    }
    canonical_controller = json.dumps(
        controller, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    config["controller_sha256"] = hashlib.sha256(canonical_controller).hexdigest()
    if args.sampling_seed is not None:
        if args.sampling_seed < 0:
            parser.error("--sampling-seed must be non-negative")
        config["sampling_seed"] = args.sampling_seed
    if args.seed_override is not None:
        canonical_instances = json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        config["seed_override"] = args.seed_override
        config["task_instances_sha256"] = hashlib.sha256(canonical_instances).hexdigest()
    if policy is not None:
        canonical_policy = json.dumps(
            policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        config["policy"] = policy
        config["policy_sha256"] = hashlib.sha256(canonical_policy).hexdigest()
    config_path = output / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise SystemExit("existing output config differs; refuse unsafe resume")
    write_json(config_path, config)

    result_log = output / "results.jsonl"
    completed = set()
    if args.resume and result_log.exists():
        with result_log.open("r", encoding="utf-8") as handle:
            latest = {}
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    latest[value["task_id"]] = value
            completed = {
                task_id for task_id, value in latest.items()
                if int(value.get("request_errors", 0)) == 0 and "error" not in value
            }
    strong = {"endpoint": args.strong_endpoint, "api_key": args.strong_api_key, "model": args.strong_model}
    weak = {"endpoint": args.weak_endpoint, "api_key": args.weak_api_key, "model": args.weak_model}
    errors = 0
    with SystemMonitor(output / "system_metrics.jsonl", args.monitor_interval):
        for index, row in enumerate(rows, 1):
            task_id = str(row["task_id"])
            if task_id in completed:
                print(f"[{index}/{len(rows)}] {task_id}: resumed")
                continue
            task_run = output / "tasks" / task_id
            if task_run.exists():
                _archive_failed_attempt(task_run, output / "failed_attempts")
            try:
                result = run_task(
                    method=args.method,
                    row=row,
                    team_root=project_root / "third_party" / "TeamBench",
                    run_dir=task_run,
                    request_log=output / "requests.jsonl",
                    image=args.docker_image,
                    controller=controller,
                    strong=strong,
                    weak=weak,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    policy=policy,
                    sampling_seed=args.sampling_seed,
                )
            except Exception as exc:
                errors += 1
                result = {
                    "schema_version": "general-mas-teambench-result-v1",
                    "method": args.method,
                    "task_id": task_id,
                    "source_task": row["source_task"],
                    "split": args.split,
                    "passed": False,
                    "partial_score": 0.0,
                    "request_errors": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                write_json(task_run / "result.json", result)
            append_jsonl(result_log, [result])
            print(
                f"[{index}/{len(rows)}] {task_id}: pass={result.get('passed')} "
                f"partial={result.get('partial_score', 0):.3f} "
                f"tokens={result.get('total_tokens', 0)} errors={errors}"
            )
    print(f"complete: {len(rows)} tasks, invocation errors={errors}")


if __name__ == "__main__":
    main()
