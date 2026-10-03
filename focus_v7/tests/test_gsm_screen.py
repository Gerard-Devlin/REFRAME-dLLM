import json
from pathlib import Path
import tempfile
import unittest

from focus_dllm.common import sha256
from focus_v7.gsm_screen import saved_baselines


class SavedGsmTests(unittest.TestCase):
    def test_exact_ids_and_frozen_records_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'data.json'
            dataset.write_text('[]')
            samples = [dict(id=str(i)) for i in range(128)]
            hashes = {}
            for name in ('llada', 'fastdllm', 'ours'):
                p = root/f'gsm8k_g256_{name}'/'records.jsonl'
                p.parent.mkdir()
                p.write_text('{}\n')
                hashes[str(p)] = sha256(p)
            frozen = dict(seed=51713, datasets=dict(gsm8k=dict(
                development_ids=[str(i) for i in range(128)], sha256=sha256(dataset))),
                frozen_score_files=hashes, cells=dict(gsm8k_256=dict(correctness={}, saved={}, metrics={})))
            (root/'cpu_frozen_development128_20261003.json').write_text(json.dumps(frozen))
            self.assertFalse(saved_baselines(root, dataset, samples)['baseline_regenerated'])
            with self.assertRaises(ValueError):
                saved_baselines(root, dataset, samples[::-1])
            Path(next(iter(hashes))).write_text('mutated')
            with self.assertRaises(ValueError):
                saved_baselines(root, dataset, samples)


if __name__ == '__main__':
    unittest.main()
