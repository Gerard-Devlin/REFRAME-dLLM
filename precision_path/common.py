import hashlib
import json
from pathlib import Path
import random

MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
REVISION = "da5608172d2b74380e4e780baa19c71645e4f981"
CODE_HASH = "d363ee4a4d4bf52958645d5c715712c5b027525bb90611a52178e58695e09b50"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def load_prompts(data,role,count,seed):
    """Use existing prepared prompts only. Split BEFORE running any model.

    Development/cost and calibration draw disjoint hash partitions of train;
    evaluation uses heldout. All heldout duplicates are removed from train.
    """
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if manifest['revision']!=REVISION:
        raise ValueError('Prepared tokenizer revision differs from pinned model')
    split={}
    for name in ('train','heldout'):
        unique={}
        for row in json.loads((data/f'{name}.json').read_text(encoding='utf-8')):
            prefix=int(row['prefix'])
            if not 0<prefix<=len(row['ids']):
                raise ValueError('Invalid prefix')
            ids=row['ids'][:prefix]; key=digest(ids)
            unique[key]=dict(id=key,ids=ids)
        split[name]=unique
    for key in split['heldout']:
        split['train'].pop(key,None)
    if role=='evaluation':
        rows=list(split['heldout'].values())
    elif role in ('development','calibration'):
        bucket=0 if role=='development' else 1
        rows=[r for r in split['train'].values() if int(r['id'][:8],16)%2==bucket]
    else:
        raise ValueError('Unknown prompt role')
    rows.sort(key=lambda r:r['id']); random.Random(seed).shuffle(rows)
    if not 0<count<=len(rows):
        raise ValueError(f'Requested {count}; only {len(rows)} unique {role} prompts')
    return rows[:count]


def snapshot():
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(MODEL_ID,revision=REVISION,local_files_only=True))
    if sha256(path/'modeling.py')!=CODE_HASH:
        raise ValueError('Pinned official source hash mismatch')
    return path
