import copy
import json
import pytest
import torch
from relation_block.model import add_lora, LoRA
from relation_block.adaptation import save_lora, restore_lora, update_statistics
from relation_block.resident_eval import bf16_copy
from relation_block.full_state import load_full_model
from relation_block.tests.test_full_training import model, objective


@pytest.mark.parametrize('rank',[8,32,64])
def test_frozen_backbone_and_zero_initial_export(rank):
    m = model()
    baseline = bf16_copy(m)
    add_lora(m,rank)
    assert sum(isinstance(x,LoRA) for x in m.modules()) == 7
    assert not m.lm_head.weight.requires_grad and not m.model.embed_tokens.weight.requires_grad
    assert m.lm_head.weight is m.model.embed_tokens.weight
    exported = bf16_copy(m)
    for k,v in baseline.state_dict().items(): assert torch.equal(v,exported.state_dict()[k])
    frozen = {k:p.clone() for k,p in m.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    objective(m).backward()
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    opt.step()
    for k,p in m.named_parameters():
        if k in frozen: assert torch.equal(p,frozen[k])
    assert update_statistics(m)['update_frobenius'] > 0


def test_adapter_optimizer_resume_and_merged_export(tmp_path):
    original = model()
    m = copy.deepcopy(original)
    add_lora(m,8)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    objective(m).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    expected_export = bf16_copy(m)
    checkpoint = save_lora(tmp_path,m,opt,dict(completed_steps=1,adaptation='lora',lora_rank=8),0,1)
    loaded,meta = load_full_model(checkpoint/'merged','cpu')
    assert meta['adaptation'] == 'lora'
    for k,v in expected_export.state_dict().items(): assert torch.equal(v,loaded.state_dict()[k])
    objective(m).backward(); opt.step()
    restored = copy.deepcopy(original)
    add_lora(restored,8)
    opt2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-3)
    restore_lora(checkpoint,restored,opt2,0,1,dict(adaptation='lora',lora_rank=8))
    objective(restored).backward(); opt2.step()
    for k,v in m.state_dict().items(): assert torch.equal(v,restored.state_dict()[k])
    with pytest.raises(ValueError,match='GPU count'):
        restore_lora(checkpoint,restored,opt2,0,2,{})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_checkpointed_lora_backward_and_eval():
    from relation_block.codec import Codec
    from relation_block.train import loss
    m = model().cuda()
    add_lora(m,8)
    m.gradient_checkpointing = True
    ids = torch.tensor([[2,3,4,5,6,7,8,9]],device='cuda')
    response = torch.arange(8,device='cuda')[None]>=2
    codec = Codec(dict(block_size=4,vocab_size=41,protected=[39,40],swaps=[]),identity=True).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        value = loss(m,ids,response,torch.tensor([2],device='cuda'),codec,
                     torch.full((1,8),.4,device='cuda'),torch.full((1,2),.5,device='cuda'),40,4)
    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    frozen = m.model.embed_tokens.weight.clone()
    exported = bf16_copy(m)
    assert not any(isinstance(x,LoRA) for x in exported.modules())
    assert torch.equal(m.model.embed_tokens.weight,frozen)


def test_sweep_report_rejects_unmatched_training(tmp_path):
    from relation_block.lora_report import report
    controls = dict(lr=4e-6,seed=1234,global_batch=12,micro_batch=1,world_size=4,steps=100,
        completed_steps=20,original_tokens=2000000,supervised_tokens=1800000,
        data_hash='data',evaluation_data_hash='dev',objective='same')
    for name in ('full','lora_r8','lora_r32','lora_r64'):
        train = tmp_path/name/'train'
        train.mkdir(parents=True)
        (train/'metadata.json').write_text(json.dumps(controls))
        for step in (0,5,10,20):
            dest = tmp_path/name/'eval'/f'step_{step:08d}'
            dest.mkdir(parents=True)
            (dest/'summary.json').write_text(json.dumps(dict(results={r:dict(accuracy=1.) for r in ('4','8','16')})))
            for rounds in ('4','8','16'):
                (dest/f'samples_{rounds}.jsonl').write_text(json.dumps(dict(id=1,correct=True,prediction='one'))+'\n')
    report(tmp_path)
    assert json.loads((tmp_path/'summary.json').read_text())['complete']
    (tmp_path/'lora_r64/train/metadata.json').write_text(json.dumps(dict(controls,lr=1e-3)))
    with pytest.raises(ValueError,match='Control'):
        report(tmp_path)
