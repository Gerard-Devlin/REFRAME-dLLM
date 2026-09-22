"""LoRA-only optimizer checkpoints plus independently loadable merged exports."""
import json
from pathlib import Path
import random
import torch
from .model import LoRA, adapter_state
from .common import write_json
from .full_state import barrier, local_optimizer, resolve_checkpoint


@torch.no_grad()
def merged_state(model, device=None):
    """Export ordinary BF16 model keys without modifying frozen weights/adapters."""
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRA)}
    wrapped_keys = {name+'.'+key for name, module in modules.items() for key in module.state_dict()}
    state = {k: v.detach().to(device=device or v.device, dtype=torch.bfloat16, copy=True)
             for k,v in model.state_dict().items() if k not in wrapped_keys}
    for name,module in modules.items():
        weight = module.base.weight.float() + (module.b.float() @ module.a.float()) * module.scale
        state[name+'.weight'] = weight.to(device=device or weight.device, dtype=torch.bfloat16)
        if module.base.bias is not None:
            state[name+'.bias'] = module.base.bias.detach().to(device=device or weight.device, dtype=torch.bfloat16, copy=True)
    return state


@torch.no_grad()
def update_statistics(model):
    delta2, base2, count = 0., 0., 0
    for module in model.modules():
        if isinstance(module, LoRA):
            delta = (module.b.float() @ module.a.float()) * module.scale
            delta2 += float(delta.square().sum())
            base2 += float(module.base.weight.float().square().sum())
            count += 1
    return dict(adapted_matrices=count, update_frobenius=delta2**.5,
                relative_update_frobenius=(delta2/max(base2,1e-30))**.5)


def save_lora(output, model, optimizer, metadata, rank, world):
    output = Path(output).resolve()
    target = output / f"step_{metadata['completed_steps']:08d}"
    staging = output / (target.name+'.incomplete')
    if rank == 0:
        if target.exists() or staging.exists():
            raise ValueError('Checkpoint already exists; choose a new output')
        staging.mkdir(parents=True)
    barrier()
    state = dict(optimizer=local_optimizer(optimizer).state_dict(), torch_rng=torch.get_rng_state(),
                 python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    torch.save(state, staging/f'rank_{rank}.pt')
    if rank == 0:
        torch.save(adapter_state(model), staging/'adapter.pt')
        write_json(staging/'metadata.json',dict(metadata,world_size=world,format='lora-resident-v1'))
        # Separate ordinary-model export, usable by the existing single-GPU evaluator.
        export = staging/'merged'
        export.mkdir()
        torch.save(merged_state(model,'cpu'), export/'model_bf16.pt')
        write_json(export/'config.json',model.config)
        write_json(export/'metadata.json',dict(metadata,world_size=world,format='full-v1',export_only=True))
    barrier()
    if rank == 0:
        staging.rename(target)
        write_json(output/'latest.json',dict(checkpoint=target.name))
        write_json(output/'status.json',dict(metadata,world_size=world,format='lora-resident-v1'))
    barrier()
    return target


def restore_lora(path, model, optimizer, rank, world, expected):
    path = resolve_checkpoint(path)
    meta = json.loads((path/'metadata.json').read_text())
    if meta.get('format') != 'lora-resident-v1' or meta['world_size'] != world:
        raise ValueError('Adapter checkpoint format/GPU count differs')
    for key,value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'Adapter resume metadata differs: {key}')
    state = torch.load(path/'adapter.pt',map_location='cpu',weights_only=True)
    params = {k:v for k,v in model.named_parameters() if v.requires_grad}
    if set(state) != set(params):
        raise ValueError('Adapter parameter names differ')
    with torch.no_grad():
        for k,v in params.items(): v.copy_(state[k])
    saved = torch.load(path/f'rank_{rank}.pt',map_location='cpu',weights_only=True)
    local_optimizer(optimizer).load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    random.setstate(saved['python_rng'])
    if saved['cuda_rng'] is not None: torch.cuda.set_rng_state(saved['cuda_rng'])
    return meta
