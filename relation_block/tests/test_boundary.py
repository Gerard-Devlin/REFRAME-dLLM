import copy
import pytest
import torch
import torch.nn.functional as F
from relation_block.boundary import masked_loss, reconstruction
from relation_block.codec import Codec
from relation_block.model import Model, training_mask
from relation_block.train import loss as clean_loss


def setup():
    torch.manual_seed(41)
    model = Model(dict(vocab_size=41, hidden_size=16, intermediate_size=24, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=1, rms_norm_eps=1e-6,
        rope_theta=1000000., tie_word_embeddings=True))
    codec = Codec(dict(block_size=4, vocab_size=41, protected=[39,40], swaps=[]), identity=True)
    raw = torch.tensor([[2,3,4,5,6,7,8,39]])
    response = torch.arange(8)[None] >= 2
    noise = torch.tensor([[.1,.9,.1,.9,.1,.9,.1,.9]])
    return model, codec, raw, response, torch.tensor([2]), noise, torch.full((1,2), .5)


def test_masked_boundaries_match_dense_shifted_ce_and_gradients():
    model, codec, raw, response, prefix, noise, probs = setup()
    reference = copy.deepcopy(model)
    value = masked_loss(model, raw, response, prefix, codec, noise, probs, 40, 4)
    value.backward()
    # Independent dense logits/shifted-label reference, including block start 4.
    selected = noise < .5
    inputs, labels = [], []
    for choose in (selected, ~selected):
        mask = response & choose
        inputs.append(torch.cat((raw.masked_fill(mask,40),raw),1))
        labels.append(raw.masked_fill(~mask,-100))
    logits, _ = reference(torch.cat(inputs), torch.arange(8).repeat(2)[None].repeat(2,1),
                           training_mask(8,4,'cpu').repeat(2,1,1,1))
    expected = F.cross_entropy(logits[:,:7].reshape(-1,41), torch.cat(labels)[:,1:].reshape(-1), reduction='sum')/response.sum()
    expected.backward()
    torch.testing.assert_close(value, expected)
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)


def test_boundary_is_masked_and_supervised_once_in_B():
    model, codec, raw, response, prefix, noise, probs = setup()
    seen = []
    handle = model.register_forward_pre_hook(lambda m,a,k: seen.append((a[0].clone(),k)),with_kwargs=True)
    masked_loss(model,raw,response,prefix,codec,noise,probs,40,4)
    ids,kw = seen.pop()
    assert ids[0,4] == 40 and ids[1,4] == raw[0,4]
    assert (kw['select'][1] == 3).sum() == 1
    assert len(kw['targets']) == response.sum()
    clean_loss(model,raw,response,prefix,codec,noise,probs,40,4)
    ids,kw = seen.pop()
    assert (ids[:,4] == raw[0,4]).all()
    assert (kw['select'][1] == 11).sum() == 1
    handle.remove()


@pytest.mark.parametrize('mode',['clean','masked'])
def test_partition_ce_sums_and_eos_boundary_overlap(mode):
    model,codec,*_ = setup()
    # EOS at block first belongs only to EOS, not double-counted in boundary.
    rows = [dict(ids=[2,3,4,5,39,6,7,8],prefix=2)]
    cfg = dict(length=8,block_size=4,pad_id=0,mask_id=40,eos_id=39)
    result = reconstruction(model.eval(),codec,rows,cfg,mode)
    for r in result.values():
        cats = r['categories']
        assert cats['all']['total'] == 6
        assert sum(cats[k]['total'] for k in ('boundary','ordinary','eos')) == 6
        assert cats['eos']['total'] == 1 and cats['boundary']['total'] == 0
        assert cats['block_first']['total'] == 1
        assert sum(cats[k]['loss_contribution'] for k in ('boundary','ordinary','eos')) == pytest.approx(r['cross_entropy'],abs=1e-6)
    assert not model._forward_pre_hooks and not model.lm_head._forward_hooks
