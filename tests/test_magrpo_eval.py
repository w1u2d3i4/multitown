import argparse
import tempfile
import unittest
from pathlib import Path

from multitown.magrpo_eval import _aggregate, _validate_args


class MagrpoEvalTests(unittest.TestCase):
    def test_reward_aggregate_keeps_seed_means(self) -> None:
        value = _aggregate(
            [
                {"seed": 1, "rewards": [0.0, 1.0]},
                {"seed": 2, "rewards": [1.0, 2.0]},
            ]
        )
        self.assertEqual(value["observations"], 4)
        self.assertEqual(value["reward_mean"], 1.0)
        self.assertEqual(value["per_seed_reward_mean"], {"1": 0.5, "2": 1.5})

    def test_magrpo_accepts_shared_or_distinct_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            output = root / "output"
            first.mkdir()
            second.mkdir()
            for models in ([first], [first, second]):
                args = argparse.Namespace(
                    method="magrpo",
                    models=models,
                    eval_samples=1,
                    max_new_tokens=1,
                    prompt_profile="official",
                    grpo_split_policy="official-fallback",
                    seeds=[1],
                    output=output,
                )
                _validate_args(args)

    def test_grpo_rejects_two_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            args = argparse.Namespace(
                method="grpo",
                models=[first, second],
                eval_samples=1,
                max_new_tokens=1,
                prompt_profile="official",
                grpo_split_policy="official-fallback",
                seeds=[1],
                output=root / "output",
            )
            with self.assertRaises(ValueError):
                _validate_args(args)


if __name__ == "__main__":
    unittest.main()
