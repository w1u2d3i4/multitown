import unittest

from multitown.magrpo_report import build_report


def _run(method: str, rewards: list[list[float]], output_tokens: int = 8) -> dict:
    seeds = list(range(len(rewards)))
    return {
        "schema": "multitown-magrpo-evaluation-v1",
        "status": "complete",
        "method": method,
        "dataset": "tldr",
        "data": {"sha256": "data"},
        "upstream": {"commit": "same"},
        "protocol": {
            "eval_start": 0,
            "eval_samples": len(rewards[0]),
            "seeds": seeds,
            "maximum_total_output_tokens_per_task": 8,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "generation": {
            "prompt_tokens": 10,
            "completion_tokens": output_tokens,
        },
        "wall_seconds": 1.0,
        "resources": {"estimated_gpu_energy_wh": 0.1},
        "models": [{"tree_sha256": method}],
        "seed_records": [
            {"seed": seed, "rewards": values} for seed, values in zip(seeds, rewards)
        ],
    }


class MagrpoReportTests(unittest.TestCase):
    def test_paired_report_detects_direction(self) -> None:
        base = _run("magrpo", [[0.0, 0.0], [0.0, 0.0]])
        magrpo = _run("magrpo", [[1.0, 1.0], [1.0, 1.0]])
        grpo = _run("grpo", [[2.0, 2.0], [2.0, 2.0]])
        report = build_report(
            base, magrpo, grpo, bootstrap_seed=7, bootstrap_iterations=1000
        )
        self.assertTrue(report["claim"]["magrpo_improved_over_untrained"])
        self.assertFalse(report["claim"]["magrpo_outperformed_grpo"])
        self.assertEqual(report["comparisons"][0]["mean_reward_difference"], 1.0)
        self.assertEqual(report["comparisons"][2]["mean_reward_difference"], -1.0)

    def test_report_rejects_unequal_actual_output_tokens(self) -> None:
        base = _run("magrpo", [[0.0]], 8)
        magrpo = _run("magrpo", [[1.0]], 7)
        grpo = _run("grpo", [[2.0]], 8)
        with self.assertRaises(ValueError):
            build_report(
                base, magrpo, grpo, bootstrap_seed=7, bootstrap_iterations=1000
            )


if __name__ == "__main__":
    unittest.main()
