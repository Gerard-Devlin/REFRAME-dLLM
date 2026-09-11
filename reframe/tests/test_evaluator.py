"""No weights/downloads: check the actual lm-eval adapter's I/O contract."""
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")
from eval_reframe import ReframeEvalHarness
from reframe_dllm.model import ReframeConfig


class Tokenizer:
    def __init__(self):
        self.chats = []

    def apply_chat_template(self, messages, **kwargs):
        self.chats.append(messages)
        return "chat"

    def __call__(self, text):
        return {"input_ids": [2, 4] if text == "chat" else [11]}

    def decode(self, ids, skip_special_tokens=False):
        return "answer" if skip_special_tokens else "answer<stop>tail"


def test_evaluator_preserves_stop_rules_and_logs_multiple_calls(tiny_model, tmp_path):
    # Avoid loading real weights; exercise the real decoder and evaluator.
    harness = object.__new__(ReframeEvalHarness)
    harness.accelerator = None
    harness.model = tiny_model
    harness.tokenizer = Tokenizer()
    harness.device = torch.device("cpu")
    harness.is_instruct = True
    harness.gen_length, harness.block_length = 4, 4
    harness.threshold, harness.factor = 1, None
    harness.mask_id, harness.show_speed = 127, False
    harness.reframe_config = ReframeConfig(kind="stale")
    harness.reframe_log = tmp_path / "metrics.jsonl"
    for doc_id in (5, 6):
        req = SimpleNamespace(args=("question", {"until": ["<stop>"]}), doc={}, doc_id=doc_id)
        assert harness.generate_until([req]) == ["answer"]
    rows = [json.loads(line) for line in harness.reframe_log.read_text().splitlines()]
    assert [row["doc_id"] for row in rows] == [5, 6]
    assert all(row["stats"]["nfe"] == 4 for row in rows)
    assert len(harness.tokenizer.chats) == 2
