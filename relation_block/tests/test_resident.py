import json
import pytest
import torch
from relation_block.resident_eval import bf16_copy, merge_records, merge_reconstruction
from relation_block.model import Model


def tiny():
    return Model(dict(vocab_size=41, hidden_size=16, intermediate_size=24,
        num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
        rms_norm_eps=1e-6, rope_theta=1000000., tie_word_embeddings=True))


@pytest.mark.parametrize('world', [1, 4, 5, 6])
def test_shards_cover_each_question_once(world):
    ids = list(range(17))
    shards = [[dict(id=i) for i in ids[rank::world]] for rank in range(world)]
    assert [r['id'] for r in merge_records(shards, ids)] == ids
    with pytest.raises(ValueError):
        merge_records(shards + [[dict(id=0)]], ids)


def test_bf16_copy_is_independent_and_matches_export():
    master = tiny()
    model = bf16_copy(master)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, master.state_dict()[key].bfloat16())
        assert tensor.data_ptr() != master.state_dict()[key].data_ptr()
    assert all(p.dtype == torch.float32 and p.requires_grad for p in master.parameters())


def test_weighted_reconstruction_not_rank_mean():
    def shard(n, good, ce):
        return {str(p): dict(cross_entropy=ce, categories={k: dict(total=n, correct=good)
                for k in ('all', 'ordinary', 'boundary', 'eos')}) for p in (.25, .5, .75)}
    merged = merge_reconstruction([shard(3, 2, 1), {}, shard(1, 0, 3)])
    assert merged['0.5']['cross_entropy'] == 1.5
    assert merged['0.5']['categories']['all']['accuracy'] == .5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('boundary_probes', [False, True])
def test_real_resident_eval_preserves_master_and_rng(tmp_path, monkeypatch, boundary_probes):
    from relation_block import resident_eval as e
    import transformers
    master = tiny().cuda().train()
    original = {k: v.clone() for k, v in master.state_dict().items()}
    spec = dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[])
    (tmp_path/'codec.json').write_text(json.dumps(spec))
    (tmp_path/'gsm8k_dev_full.json').write_text(json.dumps([dict(id=i, question='q', answer='#### 1') for i in range(2)]))
    (tmp_path/'heldout.json').write_text(json.dumps([dict(ids=[2,3,4,5,6,7,8,39], prefix=2)]))
    monkeypatch.setattr(e, 'manifest', lambda _: dict(length=8, block_size=4, pad_id=0, mask_id=40, eos_id=39))
    monkeypatch.setattr(e, 'snapshot', lambda: tmp_path)
    class Tok:
        def apply_chat_template(self, *a, **kw): return [2,3]
        def decode(self, *a, **kw): return '#### 1'
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tok())
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    e.evaluate_resident(master, tmp_path, tmp_path/'eval', 'token', 0, 1, 2, '2', 4, 1,
                        boundary_probes=boundary_probes)
    assert master.training
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
    for k, v in master.state_dict().items(): assert torch.equal(v, original[k])
    result = json.loads((tmp_path/'eval/summary.json').read_text())
    assert result['results']['2']['accuracy'] == 1
    assert 'mean_generated_tokens' in result['results']['2']
    if boundary_probes:
        assert set(result['reconstruction_by_objective']) == {'clean', 'masked'}
        assert 'cross_entropy' in result['reconstruction_by_objective']['masked']['0.5']['categories']['boundary']
    # The original FP32 training model still supports backward after evaluation.
    from relation_block.train import loss
    from relation_block.codec import Codec
    ids = torch.tensor([[2,3,4,5,6,7,8,39]], device='cuda')
    response = torch.arange(8, device='cuda')[None] >= 2
    with torch.autocast('cuda', dtype=torch.bfloat16):
        value = loss(master, ids, response, torch.tensor([2], device='cuda'), Codec(spec).cuda(),
                     torch.full((1,8), .4, device='cuda'), torch.full((1,2), .5, device='cuda'), 40, 4)
    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in master.parameters())
