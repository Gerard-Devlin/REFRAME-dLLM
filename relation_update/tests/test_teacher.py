from types import SimpleNamespace

import torch
from torch import nn

from relation_update.data import coarsen
from relation_update.teacher import NativeObserver, active_mask, current_candidates, shifted, transition


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 10, bias=False)
        self.calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        hidden = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        hidden = hidden + (input_ids != 9).float().sum(-1)[:, None, None] * 0.1
        return SimpleNamespace(logits=self.lm_head(hidden))

    def sample_with_top_p(self, logits, **kwargs):
        p = logits.softmax(-1)
        return p.argmax(-1), p


def test_shift_and_next_active_subblock_follow_native_indices():
    x = torch.tensor([[[0.], [1.], [2.], [3.]]])
    assert shifted(x).flatten().tolist() == [0, 0, 1, 2]
    ids = torch.tensor([[1, 9, 9, 9], [1, 2, 9, 9]])
    assert active_mask(ids, 9, 2).tolist() == [[False, True, False, False], [False, False, True, True]]


def test_current_candidate_ties_include_argmax_without_teacher_or_duplicates():
    logits = torch.zeros(2, 4, 10)
    result = current_candidates(logits, 3)
    assert torch.equal(result[..., 0], logits.argmax(-1))
    assert all(len(set(row)) == 3 for row in result.reshape(-1, 3).tolist())


def test_transition_rejects_repetition_remasking_and_nonmask_edits():
    s = torch.tensor([[1, 9, 9, 9]])
    assert transition(s, s.clone(), 9) is None
    assert transition(s, torch.tensor([[2, 9, 9, 9]]), 9) is None
    assert transition(s, torch.tensor([[9, 9, 9, 9]]), 9) is None
    assert transition(s, torch.tensor([[1, 2, 9, 9]]), 9).tolist() == [[False, True, False, False]]


@torch.no_grad()
def test_observer_records_actual_next_call_without_changing_output_or_extra_calls():
    torch.manual_seed(1)
    model = ToyTeacher()
    states = [torch.tensor([s]) for s in ([1, 9, 9, 9], [1, 0, 9, 9], [1, 0, 2, 9])]
    expected = [model(input_ids=s, update_past_key_values=False).logits for s in states]
    original_calls = model.calls
    with NativeObserver(model, block_size=4, small_block_size=2, mask_id=9, stop_id=8, top_k=3) as observer:
        actual = [model(input_ids=s, update_past_key_values=False).logits for s in states]
    assert model.calls - original_calls == len(states)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert len(observer.records) == 2
    first = observer.records[0]
    assert first["committed"].tolist() == [False, True, False, False]
    assert first["eligible"].tolist() == [False, False, True, True]
    assert torch.equal(first["candidates"], current_candidates(shifted(expected[0]), 3)[0])
    assert torch.allclose(first["teacher_probs"], coarsen(shifted(expected[1]), first["candidates"][None])[0])
    assert torch.allclose(first["base_log_probs"].exp(), coarsen(shifted(expected[0]), first["candidates"][None])[0])
    assert all(not value.requires_grad for value in first.values())


@torch.no_grad()
def test_cache_write_breaks_transition_chain_and_hook_is_removed_on_exception():
    model = ToyTeacher()
    original = model.forward
    try:
        with NativeObserver(model, block_size=4, mask_id=9, stop_id=8, top_k=2) as observer:
            model(input_ids=torch.tensor([[1, 9, 9, 9]]), update_past_key_values=False)
            model(input_ids=torch.tensor([[1, 2, 3, 4]]), update_past_key_values=True)
            model(input_ids=torch.tensor([[1, 0, 9, 9]]), update_past_key_values=False)
            assert not observer.records
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert model.forward == original
    assert not model.lm_head._forward_pre_hooks
