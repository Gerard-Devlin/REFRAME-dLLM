"""Isolate block-first masking, following the public 7B modeling.py convention.

Reference: Efficient-Large-Model/Fast_dLLM_v2_7B, revision
0661abf5f9f0ee338970d091052a26c8efa51974, modeling.py training branch.
This is not a claim to reproduce the original 1.5B data/training recipe.
"""
import torch
import torch.nn.functional as F
from .model import training_mask
from .train import batch, loss as clean_loss


def masked_loss(model, raw, response, prefix, codec, mask_random, probabilities, mask_id, block_size):
    if not codec.identity:
        raise ValueError('Boundary experiment is token-only')
    b, length = raw.shape
    if response[:, 0].any():
        raise ValueError('Position zero has no preceding prediction position')
    choose = mask_random < probabilities.repeat_interleave(block_size, 1)
    valid = response | (torch.arange(length, device=raw.device)[None] < prefix[:, None])
    mask = training_mask(length, block_size, raw.device) & valid.repeat(1, 2)[:, None, None, :]
    positions = torch.arange(length, device=raw.device).repeat(2)[None].expand(2*b, -1)
    inputs, targets, row_indices, columns = [], [], [], []
    for j, selected in enumerate((choose, ~choose)):
        masked = response & selected
        noisy = torch.where(masked, mask_id, raw)
        inputs.append(torch.cat((noisy, raw), 1))
        row, col = masked.nonzero(as_tuple=True)
        row_indices.append(row + j*b)
        columns.append(col - 1)
        targets.append(raw[row, col])
    numerator = model(torch.cat(inputs), positions, mask.repeat(2, 1, 1, 1),
                      select=(torch.cat(row_indices), torch.cat(columns)), targets=torch.cat(targets))
    return numerator / response.sum().clamp_min(1)


@torch.no_grad()
def reconstruction(model, codec, rows, config, mode, seed=1234):
    """Disjoint CE partition: boundary non-EOS + interior non-EOS + EOS = all."""
    objective = clean_loss if mode == 'clean' else masked_loss
    length, block = config['length'], config['block_size']
    device = next(model.parameters()).device
    target = edges = None
    offset = 0
    counts = {}

    def before(_module, _args, kwargs):
        nonlocal target, edges, offset
        target = kwargs['targets']
        col = (kwargs['select'][1] + 1) % length
        edges = col % block == 0
        offset = 0

    def after(_module, _args, logits):
        nonlocal offset
        gold = target[offset:offset+len(logits)]
        boundary = edges[offset:offset+len(logits)]
        eos = gold == config['eos_id']
        good = logits.argmax(-1) == gold
        ce = F.cross_entropy(logits.float(), gold, reduction='none')
        for name, selected in (('all', torch.ones_like(eos)), ('boundary', boundary & ~eos),
                               ('ordinary', ~boundary & ~eos), ('eos', eos), ('block_first', boundary)):
            counts[name][0] += int((good & selected).sum())
            counts[name][1] += int(selected.sum())
            counts[name][2] += float(ce[selected].sum())
        offset += len(logits)

    h1 = model.register_forward_pre_hook(before, with_kwargs=True)
    h2 = model.lm_head.register_forward_hook(after)
    results = {}
    try:
        for probability in (.25, .5, .75):
            counts = {k: [0, 0, 0.] for k in ('all', 'boundary', 'ordinary', 'eos', 'block_first')}
            for i in range(len(rows)):
                raw, response, prefix = batch(rows, [i], length, config['pad_id'], device)
                rng = torch.Generator().manual_seed(seed+i)
                noise = torch.rand(1, length, generator=rng).to(device)
                probs = torch.full((1, length//block), probability, device=device)
                objective(model, raw, response, prefix, codec, noise, probs, config['mask_id'], block)
                if offset != len(target):
                    raise AssertionError('Missing instrumented targets')
            n = counts['all'][1]
            results[str(probability)] = dict(cross_entropy=counts['all'][2]/n,
                categories={k: dict(correct=v[0], total=v[1], accuracy=v[0]/v[1] if v[1] else None,
                    ce_sum=v[2], cross_entropy=v[2]/v[1] if v[1] else None,
                    loss_contribution=v[2]/n) for k, v in counts.items()})
    finally:
        h1.remove()
        h2.remove()
    return results
