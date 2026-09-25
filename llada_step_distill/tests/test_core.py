import json
from pathlib import Path

import pytest
import torch
from torch import nn

from llada_step_distill.core import (
    BLOCK_LENGTH, Decontaminator, TRAIN_QUOTAS, normalize_text, validation_quotas,
)
from llada_step_distill.data import (
    build_acceleration_state, build_retention_state, deterministic_state, proportional_quotas,
)
from llada_step_distill.decode import transfer_schedule
from llada_step_distill.evaluate import fixed_split
from llada_step_distill.model import LoRALinear, chunked_logsumexp, selected_log_probs, topk_distribution


def row(length=96):
    return {"sample_id": "abc", "prompt_ids": [10, 11], "response_ids": list(range(100, 100 + length))}


def test_exact_10m_quotas():
    assert proportional_quotas(10_000_000) == TRAIN_QUOTAS
    assert sum(TRAIN_QUOTAS.values()) == 10_000_000
    assert sum(validation_quotas().values()) == 20_000


def test_decontamination_exact_and_near_duplicate():
    question = "A farmer has five red apples and buys four more today"
    guard = Decontaminator([question])
    assert guard.contaminated(question.upper() + "!!!")
    assert guard.contaminated(question + " now")
    assert not guard.contaminated("A completely unrelated short prompt")


def test_state_has_no_future_answers():
    item = row()
    block, reveal = deterministic_state(item)
    ids, reference, absolute = build_acceleration_state(item, block, reveal)
    start = len(item["prompt_ids"]) + block * BLOCK_LENGTH
    for offset in range(BLOCK_LENGTH):
        expected = reference[block * BLOCK_LENGTH + offset] if reveal & (1 << offset) else 126336
        assert ids[start + offset] == expected
    assert all(value == 126336 for value in ids[start + BLOCK_LENGTH :])
    assert absolute == start


def test_retention_masks_only_real_response():
    item = row(40)
    ids, _, masked, p = build_retention_state(item, 3)
    assert 0 < p <= 1
    assert any(masked[:40])
    assert not any(masked[40:])
    assert len(ids) == len(item["prompt_ids"]) + 512


def test_transfer_schedule_is_exact():
    for steps in (8, 16, 32):
        schedule = transfer_schedule(32, steps)
        assert len(schedule) == steps
        assert sum(schedule) == 32


def test_fixed_gsm_split_is_stable_and_complete():
    rows = [{"id": str(i)} for i in range(1319)]
    dev, holdout = fixed_split(rows, "dev"), fixed_split(rows, "holdout")
    assert len(dev) == 256 and len(holdout) == 1063
    assert {x["id"] for x in dev}.isdisjoint(x["id"] for x in holdout)
    assert fixed_split(list(reversed(rows)), "dev") == list(reversed(dev))


def test_zero_lora_and_merge_parity():
    torch.manual_seed(1)
    base = nn.Linear(7, 5, bias=False)
    module = LoRALinear(base, rank=3, alpha=6)
    x = torch.randn(4, 7)
    reference = base(x)
    assert torch.equal(module(x), reference)
    with torch.no_grad():
        module.lora_B.normal_()
    adapted = module(x)
    merged = module.merge()
    assert torch.allclose(adapted, merged(x), atol=2e-6, rtol=2e-6)


def test_chunked_vocabulary_math_matches_dense():
    torch.manual_seed(4)
    hidden = torch.randn(3, 7)
    head = nn.Linear(7, 23, bias=False)
    dense = head(hidden).float()
    assert torch.allclose(chunked_logsumexp(hidden, head, 5), torch.logsumexp(dense, -1), atol=1e-6)
    labels = torch.tensor([1, 9, 22])
    expected = torch.log_softmax(dense, -1).gather(1, labels[:, None]).squeeze(1)
    assert torch.allclose(selected_log_probs(hidden, labels, head, 5), expected, atol=1e-6)
    ids, values, residual = topk_distribution(hidden, head, k=4, chunk_size=5)
    expected_values, expected_ids = torch.log_softmax(dense, -1).topk(4, -1)
    assert torch.equal(ids, expected_ids)
    assert torch.allclose(values, expected_values, atol=1e-6)
    assert torch.allclose(residual.exp(), 1 - expected_values.exp().sum(-1), atol=1e-6)
