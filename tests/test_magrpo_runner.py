import argparse
import tempfile
import unittest
from pathlib import Path

from multitown.magrpo_runner import (
    ResourceMonitor,
    _dataset_paths,
    _tree_digest,
    _validate_args,
)


class MagrpoRunnerTests(unittest.TestCase):
    def test_dataset_paths_are_local_and_protocol_specific(self) -> None:
        root = Path("/data")
        tldr_train, tldr_eval = _dataset_paths(root, "tldr")
        arxiv_train, arxiv_eval = _dataset_paths(root, "arxiv")
        self.assertEqual(tldr_train.name, "train-00000-of-00001.parquet")
        self.assertEqual(tldr_eval.name, "test-00000-of-00001.parquet")
        self.assertEqual(arxiv_train.name, "train-00000-of-00001.parquet")
        self.assertEqual(arxiv_eval.name, "val-00000-of-00001.parquet")
        with self.assertRaises(ValueError):
            _dataset_paths(root, "unknown")

    def test_tree_digest_is_content_and_path_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("one")
            first = _tree_digest(root)
            (root / "a").write_text("two")
            second = _tree_digest(root)
            self.assertNotEqual(first, second)
            (root / "a").rename(root / "b")
            self.assertNotEqual(_tree_digest(root), second)
            cache = root / ".cache"
            cache.mkdir()
            before_cache = _tree_digest(root)
            (cache / "transient").write_text("ignored")
            self.assertEqual(_tree_digest(root), before_cache)

    def test_nonempty_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            output = root / "output"
            output.mkdir()
            (output / "existing.json").write_text("{}")
            args = argparse.Namespace(
                train_samples=1,
                eval_samples=1,
                epochs=1,
                generations=2,
                max_new_tokens=8,
                output=output,
                model=model,
            )
            with self.assertRaises(FileExistsError):
                _validate_args(args)

    def test_resource_summary_integrates_power(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            monitor = ResourceMonitor(Path(directory) / "monitor.jsonl")
            monitor.samples = [
                {
                    "elapsed_seconds": 0.0,
                    "process_rss_bytes": 10,
                    "system_memory_used_bytes": 100,
                    "gpu_power_w": 20.0,
                },
                {
                    "elapsed_seconds": 3600.0,
                    "process_rss_bytes": 20,
                    "system_memory_used_bytes": 200,
                    "gpu_power_w": 40.0,
                },
            ]
            summary = monitor.summary()
            self.assertEqual(summary["peak_process_rss_bytes"], 20.0)
            self.assertAlmostEqual(summary["estimated_gpu_energy_wh"], 30.0)


if __name__ == "__main__":
    unittest.main()
