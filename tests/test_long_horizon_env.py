import tempfile
import unittest
from pathlib import Path

import numpy as np

from multitown.long_horizon_env import (
    HELD_OUT_COMBINATIONS,
    MultiTownLongHorizonEnv,
    RLAction,
    a8_heuristic_policy,
    freeze_bank,
    generate_episode,
    oracle_policy,
    read_episode_bank,
    run_policy,
    strong_only_policy,
    weak_only_policy,
)


class LongHorizonEnvTests(unittest.TestCase):
    def test_generation_is_deterministic_and_test_is_combination_ood(self) -> None:
        first = generate_episode(1234, "test")
        second = generate_episode(1234, "test")
        self.assertEqual(first, second)
        self.assertTrue(20 <= first.max_steps <= 50)
        held_out = set(HELD_OUT_COMBINATIONS)
        self.assertTrue(all((item.family, item.failure_mode) in held_out for item in first.incidents))
        train = generate_episode(1234, "train")
        self.assertTrue(all((item.family, item.failure_mode) not in held_out for item in train.incidents))

    def test_observation_mask_and_invalid_action_are_auditable(self) -> None:
        env = MultiTownLongHorizonEnv(generate_episode(777, "train"))
        observation, _ = env.reset()
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertFalse(env.action_mask()[RLAction.EXECUTE])
        _, reward, terminated, _, info = env.step(RLAction.EXECUTE)
        self.assertLess(reward, 0)
        self.assertFalse(terminated)
        self.assertEqual(info["invalid_actions"], 1)

    def test_fixed_policies_terminate_within_horizon_and_budget(self) -> None:
        episode = generate_episode(991, "dev")
        for policy in (a8_heuristic_policy, weak_only_policy, strong_only_policy, oracle_policy):
            result = run_policy(episode, policy)
            self.assertLessEqual(result["steps"], episode.max_steps)
            self.assertLessEqual(result["tokens_used"], episode.token_budget)
            self.assertLessEqual(result["latency_used_s"], episode.latency_budget_s)
            self.assertEqual(result["invalid_actions"], 0)

    def test_frozen_bank_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bank"
            manifest = freeze_bank(output, train=4, dev=3, test=2)
            self.assertEqual(manifest["files"]["train.jsonl"]["episodes"], 4)
            self.assertEqual(len(read_episode_bank(output / "dev.jsonl", split="dev")), 3)

    def test_human_only_completion_is_not_autonomous_success(self) -> None:
        episode = generate_episode(123, "train")
        env = MultiTownLongHorizonEnv(episode)
        env.reset()
        while env.incident is not None:
            env.step(RLAction.HUMAN)
        env.step(RLAction.STOP)
        self.assertTrue(env.info()["assisted_episode_success"])
        self.assertFalse(env.info()["episode_success"])


if __name__ == "__main__":
    unittest.main()
