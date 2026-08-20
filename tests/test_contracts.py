from __future__ import annotations

import json
import unittest

from multitown.contracts import (
    ControllerAction,
    ControllerActionKind,
    RewardComponents,
    StateFact,
    StateSnapshot,
    TaskContract,
    TrajectoryStep,
)


class ContractTests(unittest.TestCase):
    def test_trajectory_is_json_serializable_and_reward_total_is_explicit(self) -> None:
        task = TaskContract(
            task_id="task-1",
            family="test",
            instruction="choose",
            allowed_actions=("left", "right"),
            validator_id="test_oracle",
        )
        observation = StateSnapshot(
            episode_id="episode-1",
            step_index=0,
            state_version=1,
            observer_id="controller",
            facts=(StateFact("position", "center", "sim", "controller", 1, 0),),
        )
        action = ControllerAction(
            kind=ControllerActionKind.SUBMIT,
            controller_id="controller",
            task_id=task.task_id,
            selected_action="left",
            activated_agents=("worker",),
            propensity=0.75,
        )
        step = TrajectoryStep(
            trajectory_id="trajectory-1",
            episode_id="episode-1",
            architecture="A8-smoke",
            step_index=0,
            timestamp_utc="2026-08-10T00:00:00+00:00",
            task_id=task.task_id,
            observation=observation,
            controller_action=action,
            messages=(),
            tool_result={"correct": True},
            reward=RewardComponents(final_success=1, unnecessary_delegation=-0.1),
            metrics={"total_tokens": 10},
            terminated=True,
        )
        payload = step.to_dict()
        self.assertEqual(payload["controller_action"]["kind"], "submit")
        self.assertAlmostEqual(payload["reward"]["total"], 0.9)
        json.dumps(payload)

    def test_invalid_contracts_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            TaskContract("task", "test", "choose", ("x", "x"), "oracle")
        with self.assertRaises(ValueError):
            ControllerAction(
                kind=ControllerActionKind.SUBMIT,
                controller_id="controller",
                task_id="task",
            )


if __name__ == "__main__":
    unittest.main()
