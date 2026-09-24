from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from competitor_budget.decode import generate
from competitor_budget.budget import CommitGuard


class ConstantMarginalModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        probabilities = torch.tensor([0.6, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0])
        logits = probabilities.log().expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits, past_key_values=None)

    def sample_with_top_p(self, logits, top_p=0.95, temperature=0):
        probabilities = logits.softmax(-1)
        return probabilities.argmax(-1), probabilities


def test_observer_preserves_native_path_and_active_saves_model_calls():
    outputs = {}
    for policy in ("native", "observe", "budget"):
        model = ConstantMarginalModel()
        events = []
        output = generate(model, torch.tensor([[1]]), max_new_tokens=4,
                          block_size=4, small_block_size=4, mask_id=9,
                          stop_token=8, threshold=1.0, policy=policy,
                          observer=events.append)
        outputs[policy] = (output.tolist(), model.calls, events)
    assert outputs["native"][0] == outputs["observe"][0] == outputs["budget"][0]
    assert outputs["native"][1] == outputs["observe"][1] == 4
    assert outputs["budget"][1] == 3
    assert outputs["observe"][2][0]["proposed"] == [1, 2]
    assert outputs["observe"][2][0]["baseline"] == [1]


def test_non_greedy_budget_is_rejected_before_forward():
    model = ConstantMarginalModel()
    with pytest.raises(ValueError, match="greedy"):
        generate(model, torch.tensor([[1]]), max_new_tokens=4, block_size=4,
                 small_block_size=4, policy="budget", temperature=1)
    assert model.calls == 0


def test_stable_guard_uses_normal_forwards_and_resets_at_each_block():
    outcomes = {}
    for policy in ("native", "observe", "budget"):
        model = ConstantMarginalModel()
        events = []
        tokens = generate(model, torch.tensor([[1]]), max_new_tokens=8,
                          block_size=4, small_block_size=4, mask_id=9,
                          stop_token=8, threshold=1.0, policy=policy,
                          guard=CommitGuard(min_confidence=0.5, min_observations=2),
                          observer=events.append)
        outcomes[policy] = (tokens.tolist(), model.calls)
        if events:
            for block in {e["block"] for e in events}:
                first = next(e for e in events if e["block"] == block)
                assert first["extras"] == []
            assert any(e["extras"] for e in events)
    assert outcomes["native"] == outcomes["observe"]
    assert outcomes["native"][0] == outcomes["budget"][0]
    assert outcomes["budget"][1] < outcomes["native"][1]


def test_disabling_extras_retains_exact_native_call_path():
    outcomes = []
    for policy in ("native", "budget"):
        model = ConstantMarginalModel()
        tokens = generate(model, torch.tensor([[1]]), max_new_tokens=4,
                          block_size=4, small_block_size=4, mask_id=9,
                          stop_token=8, threshold=1.0, policy=policy,
                          guard=CommitGuard(min_confidence=0.99))
        outcomes.append((tokens.tolist(), model.calls))
    assert outcomes[0] == outcomes[1]
