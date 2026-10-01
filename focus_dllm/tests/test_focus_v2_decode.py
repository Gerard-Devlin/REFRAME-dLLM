"""Check free-generation semantics, rather than only a pooled layer's logits."""
from types import SimpleNamespace

import pytest
import torch

from focus_dllm.llada_common import MASK_ID
from focus_dllm.llada_decode import generate_prefix_cache
from focus_dllm.tuning import focus_v2_decode as decoder
from focus_dllm.tuning.focus_v2 import ProxyConfig
from focus_dllm.tests.test_focus_v2 import TinyModel


def native_actions(monkeypatch):
    import focus_dllm.llada_decode as original
    actions = []
    select = original._selected_positions
    def record(confidence, threshold):
        selected = select(confidence, threshold)
        actions.append((confidence.clone(), selected.clone()))
        return selected
    monkeypatch.setattr(original, '_selected_positions', record)
    return actions


def test_active_head_matches_native_canvas_nfe_and_each_release(monkeypatch):
    torch.manual_seed(93)
    model, prompt = TinyModel().eval(), torch.tensor([[1, 2, 3]])
    observed = native_actions(monkeypatch)
    original = generate_prefix_cache(model, prompt, gen_length=8, block_length=4)
    ours = []
    select = decoder._selected_positions
    def record(confidence, threshold):
        selected = select(confidence, threshold)
        ours.append((confidence.clone(), selected.clone()))
        return selected
    monkeypatch.setattr(decoder, '_selected_positions', record)
    result, details = decoder.generate(model, prompt, gen_length=8, block_length=4, trace=True)
    assert torch.equal(original.output, result.output) and original.nfe == result.nfe
    assert len(observed) == len(ours) == len(details['actions']) == result.nfe
    assert all(torch.equal(a[1], b[1]) and torch.allclose(a[0], b[0], atol=1e-7)
               for a, b in zip(observed, ours))
    assert sum(details['nfe_by_block']) == result.nfe
    assert details['warm_calls'] == 2 and details['pooling_calls'] == []


@pytest.mark.parametrize('weighted', [False, True])
def test_free_identity_pool_preserves_native_tokens_and_budget(weighted):
    torch.manual_seed(54)
    model, prompt = TinyModel().eval(), torch.tensor([[1, 2]])
    expected, _ = decoder.generate(model, prompt, gen_length=8, block_length=4)
    result, details = decoder.generate(model, prompt, gen_length=8, block_length=4,
        config=ProxyConfig(layer=1, keep_ratio=1., block_length=4, weighted=weighted,head_min_rows=32))
    assert torch.equal(result.output, expected.output) and result.nfe == expected.nfe
    assert len(details['pooling_calls']) == details['refinement_calls']
    assert details['actions'] is None


def test_pooled_generation_never_releases_future_or_mutates_formal_prefix(monkeypatch):
    torch.manual_seed(10)
    model, prompt = TinyModel().eval(), torch.tensor([[1, 2]])
    old = decoder.FocusV2Forward
    suffix_calls = []
    class Observed(old):
        def __call__(self, ids, targets, *, past_key_values):
            before = [t.clone() for pair in past_key_values for t in pair]
            assert set(targets) == set((ids[0, :4] == MASK_ID).nonzero().flatten().tolist())
            value = super().__call__(ids, targets, past_key_values=past_key_values)
            assert all(torch.equal(t,b) for t,b in zip(
                [t for pair in past_key_values for t in pair], before))
            suffix_calls.append(ids.clone())
            return value
    monkeypatch.setattr(decoder, 'FocusV2Forward', Observed)
    result, info = decoder.generate(model, prompt, gen_length=8, block_length=4,
        config=ProxyConfig(layer=1, keep_ratio=.5, mass_implementation='repeat', block_length=4,
                           head_min_rows=32), trace=True)
    assert torch.equal(result.output[:, :2], prompt)
    assert all(all(2 + a['block'] * 4 <= p < 6 + a['block'] * 4 for p in a['positions'])
               for a in info['actions'])
    assert all((suffix[0, 4:] == MASK_ID).all() for suffix in suffix_calls if suffix.shape[1] > 4)
    assert not (result.output == MASK_ID).any()


class FixedPrediction(torch.nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.ln_f = torch.nn.Identity()
        self.prediction = prediction
    def forward(self, ids, past_key_values=None, use_cache=False):
        hidden = self.model.transformer.ln_f(ids[..., None].float())
        logits = torch.full((*hidden.shape[:2], MASK_ID+1), -50.)
        logits[..., self.prediction] = 50.
        n = ids.shape[1] + (0 if past_key_values is None else past_key_values[0][0].shape[-2])
        caches = [(torch.zeros(1, 1, n, 1), torch.zeros(1, 1, n, 1))]
        return SimpleNamespace(logits=logits, past_key_values=caches)


def test_eos_does_not_skip_remaining_native_blocks():
    prompt = torch.tensor([[2]])
    result, info = decoder.generate(FixedPrediction(126081), prompt, gen_length=8, block_length=4)
    assert info['nfe_by_block'] == [1, 1] and result.nfe == 2
    assert (result.output[0, 1:] == 126081).all()


def test_bad_mask_prediction_fails_instead_of_endless_loop():
    with pytest.raises(RuntimeError, match='no progress'):
        decoder.generate(FixedPrediction(MASK_ID), torch.tensor([[2]]), gen_length=4, block_length=4)


@pytest.mark.parametrize('kwargs', [{'gen_length':0}, {'block_length':0},
    {'gen_length':7,'block_length':4}, {'block_length':4,'config':ProxyConfig(block_length=8)}])
def test_invalid_canvas_and_pool_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        decoder.generate(TinyModel(), torch.tensor([[2]]), **kwargs)
