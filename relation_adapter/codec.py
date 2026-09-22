"""Matched token, random reversible, and fitted relation coordinates."""
import json
import random
from pathlib import Path
import torch
from relation_block.codec import Codec


def specifications(data, seed=1234):
    relation = json.loads((Path(data) / 'codec.json').read_text(encoding='utf-8'))
    rng = random.Random(seed)
    protected = set(relation['protected'])
    vocab = relation['vocab_size']
    random_spec = dict(relation)
    random_spec['fitting'] = 'same fitted anchors and frequent followers; randomly reassigned reversible code IDs'
    used = set()
    swaps = []
    for anchor, source, fitted_code in relation['swaps']:
        while True:
            code = rng.randrange(vocab)
            if code not in protected and code not in (anchor, source, fitted_code) and code not in used:
                break
        used.add(code)
        swaps.append([anchor, source, code])
    random_spec['swaps'] = swaps
    token = dict(relation)
    token['fitting'] = 'identity control, no token conversion'
    return dict(token=token, random=random_spec, relation=relation)


def make_codec(spec, arm, device):
    return Codec(spec, identity=arm == 'token').to(device)


@torch.no_grad()
def coverage(codec, rows, limit=256):
    changed = eligible = 0
    for row in rows[:limit]:
        ids = torch.tensor([row['ids']], device=codec.left.device)
        prefix = row['prefix']
        z = codec(ids, prefix)
        if not torch.equal(codec(z, prefix), ids):
            raise AssertionError('Codec did not invert exactly')
        changed += int((z[:, prefix:] != ids[:, prefix:]).sum())
        eligible += len(row['ids']) - prefix
    return changed / max(eligible, 1)
