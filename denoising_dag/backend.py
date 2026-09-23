"""Official frozen Fast-dLLM-v2 forwards under immutable prefix caches."""
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .search import Context, Prediction


MODELS = {
    '1.5b': ('Efficient-Large-Model/Fast_dLLM_v2_1.5B','da5608172d2b74380e4e780baa19c71645e4f981'),
    '7b': ('Efficient-Large-Model/Fast_dLLM_v2_7B','0661abf5f9f0ee338970d091052a26c8efa51974'),
}
MASK_ID, EOS_ID = 151665,151645


def snapshot_path(size, download=False):
    from huggingface_hub import snapshot_download
    repo,revision=MODELS[size]
    return Path(snapshot_download(repo,revision=revision,local_files_only=not download,
        allow_patterns=['*.json','*.py','*.safetensors','*.txt','*.jinja'],
        max_workers=int(os.environ.get('DENOISING_DAG_DOWNLOAD_WORKERS','1')) if download else 4))


@dataclass(frozen=True)
class FrozenCache:
    pairs: tuple

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,index):
        return self.pairs[index]

    def get_seq_length(self,layer_idx=0):
        return self.pairs[layer_idx][0].shape[-2] if self.pairs else 0

    def update(self,*args,**kwargs):
        raise RuntimeError('Search cannot mutate its frozen prefix cache')

    def expand(self,batch):
        return FrozenCache(tuple((k.expand(batch,-1,-1,-1),v.expand(batch,-1,-1,-1)) for k,v in self.pairs))

    def stamp(self):
        return tuple((tuple(t.shape),t.data_ptr(),t._version) for pair in self.pairs for t in pair)

    @classmethod
    def capture(cls,cache):
        if cache is None:
            return cls(())
        return cls(tuple(tuple(t.detach().clone() for t in cache[i]) for i in range(len(cache))))


@dataclass(frozen=True)
class Snapshot:
    ids: tuple[int,...]
    cache: FrozenCache
    context: Context
    reference_logits: object = None


def prompt_ids(tokenizer,question):
    return tokenizer.apply_chat_template([
        dict(role='user',content=question+'\nExplain your reasoning and end with #### followed by the final number.')
    ],tokenize=True,add_generation_prompt=True)


class Backend:
    def __init__(self,model,tokenizer,revision,block_size=32,batch_size=4):
        if model.training or batch_size<1 or block_size<2:
            raise ValueError('Evaluation mode and positive batch/block sizes required')
        self.model,self.tokenizer,self.revision=model,tokenizer,revision
        self.block_size,self.batch_size=block_size,batch_size
        self.device=next(model.parameters()).device
        self.reset_stats()

    @classmethod
    def load(cls,size,block_size=32,batch_size=4):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path=snapshot_path(size)
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        return cls(model,tokenizer,MODELS[size][1],block_size,batch_size)

    def reset_stats(self):
        self.forward_calls=self.rows=0
        self.events=[]

    @torch.no_grad()
    def call(self,profile=True,**kwargs):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        out=self.model(**kwargs)
        end.record()
        if profile:
            self.events.append((begin,end))
            self.forward_calls+=1
            self.rows+=kwargs['input_ids'].shape[0]
        return out

    def stats(self):
        torch.cuda.synchronize(self.device)
        return dict(forward_calls=self.forward_calls,physical_rows=self.rows,
                    model_gpu_seconds=sum(a.elapsed_time(b) for a,b in self.events)/1000)

    def make_snapshot(self,ids,cache,name):
        frozen=FrozenCache.capture(cache)
        start=frozen.get_seq_length()
        values=tuple(int(x) for x in ids)
        if len(values)!=self.block_size or values[0]==MASK_ID:
            raise ValueError('Need one complete-width block with its first token already known')
        context=Context(name,self.revision,tuple(range(start,start+len(values))),
                        f'block-causal:{self.block_size}:prefix:{start}:no-subblock-cache')
        return Snapshot(values,frozen,context)

    @torch.no_grad()
    def logits(self,snapshot,states,profile=True):
        if self.model.training:
            raise RuntimeError('Training mode invalidates exact memoization')
        if snapshot.context.model_revision!=self.revision:
            raise ValueError('Snapshot model revision differs')
        ids=torch.tensor(states,dtype=torch.long,device=self.device)
        pos=torch.tensor(snapshot.context.positions,dtype=torch.long,device=self.device)[None]
        cache=snapshot.cache.expand(len(states))
        stamp=snapshot.cache.stamp()
        out=self.call(profile=profile,input_ids=ids,position_ids=pos,past_key_values=cache,
            use_cache=True,update_past_key_values=False,use_block_cache=False,block_size=self.block_size)
        if snapshot.cache.stamp()!=stamp:
            raise AssertionError('Frozen cache mutated during prediction')
        # v2 logit at i-1 predicts token i. Position zero is never expanded.
        return torch.cat((out.logits[:,:1],out.logits[:,:-1]),dim=1)

    @torch.no_grad()
    def predict(self,snapshot,states):
        result=[]
        for offset in range(0,len(states),self.batch_size):
            logits=self.logits(snapshot,states[offset:offset+self.batch_size]).float()
            logits[...,MASK_ID]=-torch.inf
            logp=logits.log_softmax(-1)
            values,tokens=logp.max(-1)
            # The forbidden MASK category contributes exactly zero to entropy.
            entropy=-(logp.exp()*logp.masked_fill(~torch.isfinite(logp),0)).sum(-1)
            packed=torch.stack((tokens.float(),values,entropy),dim=-1).cpu().tolist()
            for row in packed:
                result.append(Prediction(tuple(int(x[0]) for x in row),
                                         tuple(x[1] for x in row),tuple(x[2] for x in row)))
        return result

    @torch.no_grad()
    def numerical_check(self,snapshot):
        """Full-vocabulary duplicate-row comparison, outside timing trials."""
        one=self.logits(snapshot,[snapshot.ids],profile=False).float()
        many=self.logits(snapshot,[snapshot.ids]*self.batch_size,profile=False).float()
        delta=many-one
        native=snapshot.reference_logits
        native_error=float((one.cpu()-native.float()).abs().max()) if native is not None else None
        return dict(batch_size=self.batch_size,
                    singleton_native_exact=torch.equal(one.cpu(),native.float()) if native is not None else None,
                    singleton_native_max_abs=native_error,
                    bitwise_equal=bool(torch.equal(many,one.expand_as(many))),
                    max_abs=float(delta.abs().max()),
                    relative_rms=float(delta.square().mean().sqrt()/one.square().mean().sqrt().clamp_min(1e-12)),
                    top1_agreement=float((many.argmax(-1)==one.argmax(-1)).float().mean()))

    @torch.no_grad()
    def native(self,prompt,max_new_tokens=128,threshold=.9,capture_count=0,request_id='probe'):
        """Run published generate; optionally observe immutable block snapshots."""
        snapshots=[]
        seen=set()
        original_forward=self.model.forward
        calls=[0]

        def observe(*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            calls[0]+=1
            captured=None
            if (capture_count and len(snapshots)<capture_count and ids is not None
                    and ids.shape[1]==self.block_size and not kwargs.get('update_past_key_values',False)
                    and not kwargs.get('use_block_cache',False)):
                cache=kwargs.get('past_key_values')
                start=cache.get_seq_length() if cache is not None else 0
                if start not in seen and int(ids[0,0])!=MASK_ID:
                    seen.add(start)
                    captured=self.make_snapshot(ids[0].tolist(),cache,f'{request_id}:prefix:{start}')
            result=original_forward(*args,**kwargs)
            if captured is not None:
                shifted=torch.cat((result.logits[:,:1],result.logits[:,:-1]),dim=1)
                snapshots.append(replace(captured,reference_logits=shifted.detach().cpu()))
            return result

        self.model.forward=observe
        try:
            x=torch.tensor([prompt],device=self.device)
            out=self.model.generate(x,tokenizer=self.tokenizer,max_new_tokens=max_new_tokens,
                block_size=self.block_size,small_block_size=8,threshold=threshold,
                temperature=0.,use_block_cache=False)
            generated=out[0,len(prompt):].tolist()
            stop=generated.index(EOS_ID) if EOS_ID in generated else None
            ended=stop is not None and stop<max_new_tokens
            if stop is not None:
                generated=generated[:stop]
            return generated[:max_new_tokens],snapshots,dict(forward_calls=calls[0],
                truncated=not ended,eos_reached=ended,
                cap_note='Unmodified official generate uses a block-count cap; actual output can be shorter than requested even without EOS.')
        finally:
            self.model.forward=original_forward


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)
