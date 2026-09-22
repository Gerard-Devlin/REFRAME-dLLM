import copy
import json

import pytest
import torch

from relation_block.codec import Codec
from relation_block.model import clean_mask
from relation_block.tests.test_full_training import model as tiny_model
from relation_block.train import loss
from relation_adapter.codec import coverage, make_codec, specifications
from relation_adapter.model import FrozenChart, bf16_base_copy, evaluation_copy
from relation_adapter.train import restore_checkpoint, save_checkpoint
from relation_adapter.report import report


def objective(chart, codec):
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    response = torch.tensor([[False, False, True, True, True, True, True, True]])
    return loss(chart, ids, response, torch.tensor([2]), codec,
                torch.full((1, 8), .4), torch.full((1, 2), .5), 40, 4)


def test_zero_chart_matches_original_and_base_never_updates():
    torch.manual_seed(19)
    base = tiny_model()
    assert base.lm_head.weight is base.model.embed_tokens.weight
    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    chart = FrozenChart(base, rank=8)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    positions = torch.arange(8)[None]
    mask = clean_mask(8, 4, 'cpu')
    chart.eval()
    torch.testing.assert_close(chart(ids, positions, mask)[0], base(ids, positions, mask)[0], rtol=0, atol=0)
    copy_bf16 = bf16_base_copy(base)
    exported = evaluation_copy(copy_bf16, chart)
    assert copy_bf16.lm_head.weight is copy_bf16.model.embed_tokens.weight
    torch.testing.assert_close(exported(ids, positions, mask)[0], copy_bf16(ids, positions, mask)[0], rtol=0, atol=0)
    chart.train()
    codec = Codec(dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[[3, 4, 7]]))
    optimizer = torch.optim.AdamW([p for p in chart.parameters() if p.requires_grad], lr=1e-3)
    objective(chart, codec).backward()
    trainable = [(n, p) for n, p in chart.named_parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
    assert any(p.grad.abs().sum() > 0 for _, p in trainable)
    assert all(p.grad is None for p in base.parameters())
    optimizer.step()
    assert any(p.detach().abs().sum() > 0 for n, p in trainable if n.endswith('up.weight'))
    for name, value in base.state_dict().items():
        assert torch.equal(value, frozen[name])


def test_adapter_checkpoint_resume_matches_next_update(tmp_path):
    torch.manual_seed(21)
    base = tiny_model()
    chart = FrozenChart(base, rank=4)
    codec = Codec(dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[]), identity=True)
    optimizer = torch.optim.AdamW([p for p in chart.parameters() if p.requires_grad], lr=1e-3)
    objective(chart, codec).backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    save_checkpoint(tmp_path/'step_00000001', chart, optimizer, dict(arm='token', step=1), rank=0)
    assert not (tmp_path/'step_00000001.incomplete').exists()
    objective(chart, codec).backward(); optimizer.step()
    expected = chart.trainable_state()
    restored = FrozenChart(copy.deepcopy(base), rank=4)
    optimizer2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-3)
    restore_checkpoint(tmp_path/'step_00000001', restored, optimizer2, dict(arm='token'))
    objective(restored, codec).backward(); optimizer2.step()
    for name, value in restored.trainable_state().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='Resume metadata differs'):
        restore_checkpoint(tmp_path/'step_00000001', restored, optimizer2, dict(arm='relation'))


def test_random_relation_are_invertible_with_matched_coverage(tmp_path):
    spec = dict(block_size=4, vocab_size=41, protected=[39, 40],
                swaps=[[3, 4, 7], [5, 6, 7]], shared_code=7)
    (tmp_path/'codec.json').write_text(json.dumps(spec))
    rows = [dict(ids=[2, 3, 4, 5, 5, 5, 6, 8], prefix=1)]
    specs = specifications(tmp_path, seed=7)
    assert specs['random']['swaps'] != specs['relation']['swaps']
    for arm in ('token', 'random', 'relation'):
        codec = make_codec(specs[arm], arm, 'cpu')
        fraction = coverage(codec, rows)
        assert fraction == (0 if arm == 'token' else 2/7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_bf16_autocast_chart_backward_on_gpu():
    base = tiny_model().cuda()
    chart = FrozenChart(base, rank=8)
    codec = Codec(dict(block_size=4, vocab_size=41, protected=[39, 40], swaps=[]), identity=True).cuda()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], device='cuda')
    response = torch.tensor([[False, False, True, True, True, True, True, True]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        value = loss(chart, ids, response, torch.tensor([2], device='cuda'), codec,
                     torch.full((1, 8), .4, device='cuda'),
                     torch.full((1, 2), .5, device='cuda'), 40, 4)
    value.backward()
    assert torch.isfinite(value)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in chart.parameters() if p.requires_grad)
    assert all(p.grad is None for p in base.parameters())


def test_report_requires_paired_controls(tmp_path):
    names = [f'step_{n:08d}' for n in (0, 1, 2, 5, 10)]
    original = tmp_path/'original/eval'
    original.mkdir(parents=True)
    (original/'summary.json').write_text(json.dumps(dict(ids=[1, 2])))
    base = dict(global_batch=4,world_size=2,adapter_rank=8,lr=1e-4,seed=1,
                data_hash='d',driver_hash='c',model_revision='m',
                original_tokens=10,supervised_tokens=8)
    for arm in ('token','random','relation'):
        root=tmp_path/arm
        root.mkdir()
        (root/'complete.json').write_text(json.dumps(dict(base,
            heldout_change_fraction=.1 if arm!='token' else 0)))
        for j,name in enumerate(names):
            path=root/'eval'/name
            path.mkdir(parents=True)
            ids=[1, 2] if j==4 else [1]
            (path/'summary.json').write_text(json.dumps(dict(ids=ids,
                results={r:dict(accuracy=.5) for r in ('4','8','16')})))
            for rounds in ('4','8','16'):
                (path/f'samples_{rounds}.jsonl').write_text(''.join(
                    json.dumps(dict(id=i,correct=True))+'\n' for i in ids))
    assert report(tmp_path)['complete']
    changed=tmp_path/'relation/complete.json'
    changed.write_text(json.dumps(dict(base,lr=2e-4,heldout_change_fraction=.1)))
    with pytest.raises(ValueError,match='Control differs'):
        report(tmp_path)
