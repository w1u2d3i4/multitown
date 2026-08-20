import unittest

from multitown.a8_controller import simulate_a8_cell, validate_candidate
from multitown.scenarios import build_scenario_bank


class A8ControllerTests(unittest.TestCase):
    def test_oracle_always_passes_but_validator_is_not_an_oracle(self) -> None:
        saw_wrong_action_pass = False
        for scenario in build_scenario_bank(20260807, 120):
            self.assertTrue(validate_candidate(scenario, scenario.oracle_action).hard_constraints_pass)
            for action in scenario.allowed_actions:
                if action != scenario.oracle_action and validate_candidate(scenario, action).hard_constraints_pass:
                    saw_wrong_action_pass = True
        self.assertTrue(saw_wrong_action_pass)

    def test_simulation_early_stops_only_above_threshold(self) -> None:
        scenario = build_scenario_bank(20260807, 1)[0]
        base = {
            "selected_action": scenario.oracle_action, "correct": True, "valid": True,
            "total_tokens": 100, "decision_latency_s": 1,
        }
        a2 = {**base, "candidate_actions": [scenario.oracle_action], "weak_calls": 4}
        result = simulate_a8_cell(
            scenario=scenario, a0=base, a1=base, a2=a2,
            predicted_a0_accuracy=0.9, early_stop_threshold=0.8,
        )
        self.assertTrue(result["early_stop"])
        delegated = simulate_a8_cell(
            scenario=scenario, a0=base, a1=base, a2=a2,
            predicted_a0_accuracy=0.5, early_stop_threshold=0.8,
        )
        self.assertTrue(delegated["delegated"])


if __name__ == "__main__":
    unittest.main()
