import unittest

from multitown.a26_safe_router import RouterConfig, a26_episode_bank
from multitown.a28_conservative_router import make_conservative_router_policy
from multitown.long_horizon_env import MultiTownLongHorizonEnv, RLAction, run_policy

ROUTE = (
    int(RLAction.DELEGATE),
    int(RLAction.ESCALATE),
    int(RLAction.DELEGATE),
    int(RLAction.ESCALATE),
)


class A28ConservativeRouterTests(unittest.TestCase):
    def test_disagreement_connects_before_review(self) -> None:
        episode = next(
            episode
            for episode in a26_episode_bank("dev", 300, seed_offset=95_000_000)
            if not episode.incidents[0].fail_first_actions
            and episode.incidents[0].sensor_candidate
            != (
                episode.incidents[0].weak_candidate
                if episode.incidents[0].family in {0, 2}
                else episode.incidents[0].strong_candidate
            )
        )
        env = MultiTownLongHorizonEnv(episode)
        observation, _ = env.reset()
        policy = make_conservative_router_policy(RouterConfig(ROUTE, 1.01))
        first = policy(env, observation, env.action_mask())
        self.assertEqual(first, int(RLAction.OBSERVE))
        observation, *_ = env.step(first)
        second = policy(env, observation, env.action_mask())
        self.assertIn(second, (int(RLAction.DELEGATE), int(RLAction.ESCALATE)))
        observation, *_ = env.step(second)
        third = policy(env, observation, env.action_mask())
        self.assertEqual(third, int(RLAction.CONNECT))

    def test_agreement_gate_can_execute_without_review(self) -> None:
        episode = next(
            episode
            for episode in a26_episode_bank("dev", 500, seed_offset=96_000_000)
            if not episode.incidents[0].fail_first_actions
            and episode.incidents[0].sensor_candidate
            == (
                episode.incidents[0].weak_candidate
                if episode.incidents[0].family in {0, 2}
                else episode.incidents[0].strong_candidate
            )
        )
        env = MultiTownLongHorizonEnv(episode)
        observation, _ = env.reset()
        policy = make_conservative_router_policy(RouterConfig(ROUTE, 1.01))
        for expected in (RLAction.OBSERVE, ROUTE[episode.incidents[0].family]):
            action = policy(env, observation, env.action_mask())
            self.assertEqual(action, int(expected))
            observation, *_ = env.step(action)
        self.assertEqual(
            policy(env, observation, env.action_mask()), int(RLAction.EXECUTE)
        )

    def test_controller_has_no_invalid_or_budget_actions(self) -> None:
        policy = make_conservative_router_policy(RouterConfig(ROUTE, 1.01))
        for episode in a26_episode_bank("dev", 60, seed_offset=97_000_000):
            result = run_policy(episode, policy)
            self.assertEqual(result["invalid_actions"], 0)
            self.assertEqual(result["budget_violations"], 0)
            self.assertLessEqual(result["steps"], episode.max_steps)


if __name__ == "__main__":
    unittest.main()
