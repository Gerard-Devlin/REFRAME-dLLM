"""CPU-only data contracts, matching and cost accounting."""
import hashlib
import json
import math
import random
from pathlib import Path

MODEL_ID = 'Efficient-Large-Model/Fast_dLLM_v2_1.5B'
REVISION = 'da5608172d2b74380e4e780baa19c71645e4f981'
CODE_HASH = 'd363ee4a4d4bf52958645d5c715712c5b027525bb90611a52178e58695e09b50'
MASK, EOS, BLOCK, SMALL = 151665, 151645, 32, 8
OPTIONS = dict(block_size=BLOCK, small_block_size=SMALL, threshold=.90,
               max_new_tokens=512, temperature=0, use_block_cache=False)

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def implementation_hash():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob('*.py'))})

def configuration(data, dataset):
    return dict(revision=REVISION, code=CODE_HASH, options=OPTIONS,
                data=digest(read(Path(data)/'manifest.json')), evaluation=digest(read(dataset)),
                implementation=implementation_hash(), seed=1234)

def prompt_splits(data, eval_ids, train=2000, validation=128, audit=32):
    """Remove heldout/evaluation overlap before seeded prompt-level splitting."""
    data = Path(data)
    if read(data/'manifest.json')['revision'] != REVISION:
        raise ValueError('Prepared data revision differs')
    def unique(name):
        out = {}
        for row in read(data/f'{name}.json'):
            n = int(row['prefix'])
            if not 0 < n <= len(row['ids']): raise ValueError('Invalid prompt prefix')
            ids = row['ids'][:n]; out[digest(ids)] = dict(id=digest(ids), ids=ids)
        return out
    pool, held = unique('train'), unique('heldout')
    forbidden = set(held) | {digest(ids) for ids in eval_ids}
    rows = [pool[k] for k in sorted(pool) if k not in forbidden]
    random.Random(1234).shuffle(rows)
    if len(rows) < train+validation+audit: raise ValueError('Insufficient unique prompts')
    return dict(audit=rows[:audit], validation=rows[audit:audit+validation],
                train=rows[audit+validation:audit+validation+train])

def apply_action(canvas, action):
    out = list(canvas)
    for i, token in action:
        if out[i] != MASK or token == MASK: raise ValueError('Invalid native action')
        out[i] = token
    return out

def legal_pair(a, b, special):
    if a['kind'] != 'denoise' or b['kind'] != 'denoise': return False
    if a['index']+1 != b['index'] or a['history'] != b['history']: return False
    if a['start'] != b['start']: return False
    if not a.get('action') or not b.get('action'): return False
    if EOS in a['canvas'] or EOS in b['canvas']: return False
    if any(t in special for _, t in a['action']+b['action']): return False
    return apply_action(a['canvas'], a['action']) == b['canvas']

def optimal_savings(calls, special):
    """Maximum-weight matching on consecutive calls; cannot overlap pairs."""
    dp = [0.] * (len(calls)+1)
    for n in range(1, len(calls)+1):
        dp[n] = dp[n-1]
        if n > 1 and legal_pair(calls[n-2], calls[n-1], special):
            dp[n] = max(dp[n], dp[n-2]+calls[n-1]['seconds'])
    return dp[-1]

def windows(row, special, limit=32):
    pairs = [(a,b) for a,b in zip(row['calls'],row['calls'][1:]) if legal_pair(a,b,special)]
    if len(pairs)>limit:
        pairs = [pairs[round(i*(len(pairs)-1)/(limit-1))] for i in range(limit)] if limit>1 else pairs[:1]
    result=[]
    for a,b in pairs:
        item=dict(prompt_id=row['prompt_id'], history=a['history'], canvas=a['canvas'],
                  start=a['start'], first=a['action'], second=b['action'],
                  native_first=a['action'], native_second=b['action'])
        item['id']=digest(item); validate_record(item); result.append(item)
    return result

def validate_record(r):
    if len(r['canvas'])!=BLOCK or r['start'] not in range(0,BLOCK,SMALL):
        raise ValueError('Invalid block/subblock')
    actions=r['first']+r['second']; positions=[p for p,_ in actions]
    if not r['first'] or not r['second'] or len(set(positions))!=len(positions):
        raise ValueError('Empty or overlapping targets')
    for pos,tok in actions:
        if not r['start']<=pos<r['start']+SMALL or r['canvas'][pos]!=MASK or tok in (MASK,EOS):
            raise ValueError('Future leakage or invalid target')
    if any(len(chunk)%BLOCK or not chunk or MASK in chunk for chunk in r['history']):
        raise ValueError('History must contain clean native cache-write inputs')

def batches(n, world, accum=2, seed=1234):
    order=list(range(n)); random.Random(seed).shuffle(order)
    return [order[i:i+world*accum] for i in range(0,n,world*accum)]

def rank_microbatches(batch, rank, world, accum=2):
    return [batch[j*world+rank] if j*world+rank<len(batch) else None for j in range(accum)]

def loss_counts(records):
    target=sum(len(r['first'])+len(r['second']) for r in records)
    other=sum(sum(t==MASK for t in r['canvas'][r['start']:r['start']+SMALL])-
              len(r['first'])-len(r['second']) for r in records)
    return target,other,len(records)

def lr_factor(step, total):
    warm=max(1,math.ceil(total*.05))
    if step<warm: return (step+1)/warm
    return .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
