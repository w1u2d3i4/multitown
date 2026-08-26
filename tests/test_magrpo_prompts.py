import unittest

from multitown.magrpo_prompts import budgeted_formatters


class MagrpoPromptTests(unittest.TestCase):
    def test_tldr_profiles_define_distinct_budgets(self) -> None:
        concise, detailed = budgeted_formatters("tldr")
        example = {"prompt": "A post about a difficult decision."}
        self.assertIn("12-20", concise(example))
        self.assertIn("40-60", detailed(example))
        self.assertNotEqual(concise(example), detailed(example))

    def test_arxiv_profiles_hold_role_boundary(self) -> None:
        background, method = budgeted_formatters("arxiv")
        example = {"abstract_text": "We introduce a method."}
        self.assertIn("background and motivation only", background(example))
        self.assertIn("methods and implications only", method(example))

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            budgeted_formatters("unknown")


if __name__ == "__main__":
    unittest.main()
