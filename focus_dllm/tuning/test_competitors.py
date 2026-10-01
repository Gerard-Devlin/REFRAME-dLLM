"""CPU contracts for the isolated third-party runner."""
import json
from pathlib import Path
import tempfile
import unittest

from .competitors import generation_prompt, select_samples, verified_nfe, write


class Contracts(unittest.TestCase):
    def test_manual_block_nfe_is_not_top_level_hook_count(self):
        self.assertEqual(verified_nfe("focus_v1", 26, 8, 26*32, 32), 26)
        with self.assertRaises(ValueError):
            verified_nfe("focus_v1", 26, 8, 25*32, 32)

    def test_normal_forward_nfe_consistency(self):
        self.assertEqual(verified_nfe("v1", 52, 52, 52*32, 32), 52)
        with self.assertRaises(ValueError):
            verified_nfe("v1", 52, 51, 52*32, 32)
        self.assertEqual(verified_nfe("flash_verify", 5, 9, None, 32), 9)

    def test_gold_is_not_input(self):
        sample = dict(paper_prompt="legitimate question", prompt="unused", answer="secret",
                      solution="future teacher output", test="private tests")
        self.assertEqual(generation_prompt(sample), "legitimate question")
        self.assertEqual(generation_prompt(dict(prompt="fallback", answer="secret")), "fallback")

    def test_empty_input_rejected(self):
        for sample in ({"answer": "secret"}, {"paper_prompt": ""}, {"prompt": None}):
            with self.assertRaises(ValueError):
                generation_prompt(sample)

    def test_stable_split_and_disjoint_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.json"
            write(path, [dict(id=str(i), prompt="q") for i in range(100)])
            first = select_samples(path, 32, 0)
            self.assertEqual(first, select_samples(path, 32, 0))
            validation = select_samples(path, 32, 64)
            self.assertFalse({s["id"] for s in first} & {s["id"] for s in validation})

    def test_invalid_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.json"
            write(path, [dict(id="a")])
            for limit, offset in ((0, 0), (2, 0), (1, -1), (1, 1)):
                with self.assertRaises(ValueError):
                    select_samples(path, limit, offset)

    def test_duplicate_or_missing_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.json"
            for data in ([dict(id="a"), dict(id="a")], [dict(prompt="q")]):
                write(path, data)
                with self.assertRaises(ValueError):
                    select_samples(path, len(data), 0)

    def test_atomic_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records/1.json"
            write(path, {"sample": "original"})
            write(path, {"sample": "resumed"})
            self.assertEqual(json.loads(path.read_text()), {"sample": "resumed"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
