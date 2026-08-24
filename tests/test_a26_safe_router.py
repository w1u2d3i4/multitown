import unittest

from multitown.a26_safe_router import (
    RouterConfig,
    a26_episode_bank,
    bank_sha256,
    generate_a26_episode,
    learn_specialist_map,
    make_router_policy,
)
from multitown.long_horizon_env import (
    HELD_OUT_COMBINATIONS,
    RLAction,
    run_policy,
)


class A26SafeRouterTests(unittest.TestCase):
    def test_generator_is_deterministic_nonleaking_and_ood(self) -> None:
        first = generate_a26_episode(12345, "test")
        self.assertEqual(first, generate_a26_episode(12345, "test"))
        self.assertTrue(
            all(
                (incident.family, incident.failure_mode) in HELD_OUT_COMBINATIONS
                for incident in first.incidents
            )
        )
        bank = a26_episode_bank("train", 100, seed_offset=91_000_000)
        old_formula_matches = sum(
            incident.correct_action
            == (
                incident.family
                + 2 * incident.failure_mode
                + int(incident.severity >= 0.7)
            )
            % 4
            for episode in bank
            for incident in episode.incidents
        )
        incidents = sum(len(episode.incidents) for episode in bank)
        self.assertLess(old_formula_matches / incidents, 0.35)

    def test_train_only_specialist_map_and_bank_digest(self) -> None:
        train = a26_episode_bank("train", 400, seed_offset=92_000_000)
        route = learn_specialist_map(train)
        self.assertEqual(
            route,
            (
                int(RLAction.DELEGATE),
                int(RLAction.ESCALATE),
                int(RLAction.DELEGATE),
                int(RLAction.ESCALATE),
            ),
        )
        self.assertEqual(bank_sha256(train), bank_sha256(train))
        with self.assertRaises(ValueError):
            learn_specialist_map(a26_episode_bank("dev", 10, seed_offset=93_000_000))

    def test_router_terminates_without_budget_or_action_violation(self) -> None:
        config = RouterConfig(
            (
                int(RLAction.DELEGATE),
                int(RLAction.ESCALATE),
                int(RLAction.DELEGATE),
                int(RLAction.ESCALATE),
            ),
            0.8,
        )
        for episode in a26_episode_bank("dev", 40, seed_offset=94_000_000):
            result = run_policy(episode, make_router_policy(config))
            self.assertEqual(result["invalid_actions"], 0)
            self.assertEqual(result["budget_violations"], 0)
            self.assertLessEqual(result["steps"], episode.max_steps)

    def test_invalid_configuration_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            RouterConfig((1, 2, 1), 0.8)
        with self.assertRaises(ValueError):
            RouterConfig((1, 2, 1, 5), 0.8)
        with self.assertRaises(ValueError):
            RouterConfig((1, 2, 1, 2), 2.0)


if __name__ == "__main__":
    unittest.main()
