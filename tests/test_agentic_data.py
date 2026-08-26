import json
import tempfile
import unittest
from pathlib import Path

from multitown.agentic_data import _jsonl_metadata, _select_supergpqa


class AgenticDataTests(unittest.TestCase):
    def test_jsonl_metadata_rejects_non_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"a": 1}\n{"a": 2, "b": 3}\n')
            rows, columns = _jsonl_metadata([path])
            self.assertEqual(rows, 2)
            self.assertEqual(columns, {"a", "b"})
            path.write_text("[1, 2, 3]\n")
            with self.assertRaises(TypeError):
                _jsonl_metadata([path])

    def test_supergpqa_selection_is_stable_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            rows = []
            for index in range(80):
                rows.append(
                    {
                        "uuid": f"q-{index:03d}",
                        "discipline": "Science" if index < 60 else "Law",
                        "difficulty": "easy" if index % 2 else "hard",
                    }
                )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            first = _select_supergpqa(path, count=20)
            second = _select_supergpqa(path, count=20)
            self.assertEqual(first, second)
            self.assertEqual(first["count"], 20)
            self.assertEqual(len(set(first["ids"])), 20)
            law_ids = {f"q-{index:03d}" for index in range(60, 80)}
            self.assertLessEqual(abs(len(law_ids.intersection(first["ids"])) - 5), 1)


if __name__ == "__main__":
    unittest.main()
