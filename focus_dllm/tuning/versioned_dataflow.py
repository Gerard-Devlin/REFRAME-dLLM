"""One-layer non-speculative row dataflow prototype, not a persistent compiler.

All queries read every current K/V row. Native FlashAttention remains intact;
there is no early consumption inside its key loop and no HBM-eliding fusion.
The experiment isolates removing the attention-to-MLP global phase barrier.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Value:
    role: str
    tile: int
    epoch: int


@dataclass(frozen=True)
class Task:
    name: str
    reads: tuple
    writes: tuple


def layer_plan(tiles, epoch=1, refreshed=None, cache_epoch=0):
    """Explicit SSA versions; cache choice is prescribed, never a readiness guess."""
    if tiles<1 or epoch<=cache_epoch:
        raise ValueError('Invalid tile count or monotonically increasing epoch')
    refreshed=set(range(tiles)) if refreshed is None else set(refreshed)
    if not refreshed<=set(range(tiles)):
        raise ValueError('Invalid refresh set')
    external={Value('input',i,epoch) for i in range(tiles)}
    external|={Value(role,i,cache_epoch) for i in range(tiles) if i not in refreshed for role in ('k','v')}
    tasks=[]
    for i in range(tiles):
        roles=('q','k','v') if i in refreshed else ('q',)
        tasks.append(Task(f'produce:{i}',(Value('input',i,epoch),),tuple(Value(r,i,epoch) for r in roles)))
    keys=tuple(Value(r,i,epoch if i in refreshed else cache_epoch) for i in range(tiles) for r in ('k','v'))
    for i in range(tiles):
        tasks.append(Task(f'attention:{i}',(Value('q',i,epoch),)+keys,(Value('attention',i,epoch),)))
        tasks.append(Task(f'post:{i}',(Value('input',i,epoch),Value('attention',i,epoch)),(Value('output',i,epoch),)))
    tasks.append(Task('commit',tuple(Value('output',i,epoch) for i in range(tiles)),(Value('committed',0,epoch),)))
    return tuple(tasks),frozenset(external)


def validate_order(tasks, external, order):
    """Reject missing/stale reads, duplicate writers and premature commit."""
    indexed={t.name:t for t in tasks}
    if len(indexed)!=len(tasks) or len(order)!=len(tasks) or set(order)!=set(indexed):
        raise ValueError('Every task must execute exactly once')
    available=set(external);readers={v:0 for v in external}
    for task in tasks:
        for value in task.reads:readers[value]=readers.get(value,0)+1
    reclaim=[]
    for name in order:
        task=indexed[name]
        if not set(task.reads)<=available:
            raise ValueError(f'Unready or wrong-version input: {name}')
        if set(task.writes)&available:
            raise ValueError(f'Conflicting write: {name}')
        available.update(task.writes)
        for value in task.reads:
            readers[value]-=1
            if readers[value]==0:reclaim.append(value)
    return tuple(reclaim)


def ranges(length,tile):
    if length<1 or tile<1:raise ValueError('Positive geometry required')
    return tuple((i,min(i+tile,length)) for i in range(0,length,tile))


class LayerFlow:
    """Fixed private epoch buffers; only all-current-KV -> per-row post is tested.

    The serial phased and pipelined versions invoke identical tile operators,
    shapes, rounding points and per-output Flash key order. Their equivalence
    does not imply equivalence to a differently shaped native whole-row GEMM.
    GPU joins prevent a later replay from overwriting live epoch buffers.
    """
    def __init__(self,block,hidden,tile):
        if block.training or block.q_norm is not None or block.k_norm is not None:
            raise ValueError('Native eval LLaDA without Q/K norm required')
        if hidden.shape[0]!=1 or hidden.device.type!='cuda':
            raise ValueError('B1 CUDA layer input required')
        if block.flash_attn_func is None:raise ValueError('FlashAttention required')
        self.block=block;self.x=hidden.detach().clone();self.parts=ranges(hidden.shape[1],tile)
        self.heads=block.config.n_heads;self.kv_heads=block.config.effective_n_kv_heads
        self.head_dim=hidden.shape[-1]//self.heads
        self.q=torch.empty(1,self.heads,hidden.shape[1],self.head_dim,device=hidden.device,dtype=hidden.dtype)
        self.k=torch.empty(1,self.kv_heads,hidden.shape[1],self.head_dim,device=hidden.device,dtype=hidden.dtype)
        self.v=torch.empty_like(self.k);self.a=torch.empty_like(hidden);self.output=torch.empty_like(hidden)
        self.producer=torch.cuda.Stream();self.attender=torch.cuda.Stream();self.consumer=torch.cuda.Stream()
        self.produced=torch.cuda.Event();self.all_attended=torch.cuda.Event()
        self.attended=[torch.cuda.Event() for _ in self.parts]
        self.graph=None;self.graph_output=None
        tasks,external=layer_plan(len(self.parts))
        order=[f'produce:{i}' for i in range(len(self.parts))]
        order+=[s for i in range(len(self.parts)) for s in (f'attention:{i}',f'post:{i}')]+['commit']
        validate_order(tasks,external,order)
        self.plan=tasks
        if block.config.rope:
            sin,cos=block.rotary_emb.get_rotary_embedding(hidden.shape[1],hidden.device)
            dtype=torch.float32 if block.config.rope_full_precision else hidden.dtype
            self.sin,self.cos=sin.to(dtype=dtype),cos.to(dtype=dtype)

    def produce(self,index):
        a,b=self.parts[index];block=self.block
        norm=block.attn_norm(self.x[:,a:b])
        q=block.q_proj(norm).view(1,b-a,self.heads,self.head_dim).transpose(1,2)
        k=block.k_proj(norm).view(1,b-a,self.kv_heads,self.head_dim).transpose(1,2)
        v=block.v_proj(norm).view(1,b-a,self.kv_heads,self.head_dim).transpose(1,2)
        if block.config.rope:
            q0,k0=(q.float(),k.float()) if block.config.rope_full_precision else (q,k)
            # Absolute positions, same multiply/add/cast boundaries as native RoPE.
            indices=torch.arange(a,b,device=self.x.device,dtype=torch.long)
            sin=self.sin.index_select(2,indices);cos=self.cos.index_select(2,indices)
            q=block.rotary_emb.apply_rotary_pos_emb(sin,cos,q0).type_as(q)
            k=block.rotary_emb.apply_rotary_pos_emb(sin,cos,k0).type_as(k)
        self.q[:,:,a:b].copy_(q);self.k[:,:,a:b].copy_(k);self.v[:,:,a:b].copy_(v)

    def attend(self,index):
        a,b=self.parts[index]
        att=self.block._scaled_dot_product_attention(self.q[:,:,a:b],self.k,self.v,
                    attn_mask=None,dropout_p=0.,is_causal=False)
        self.a[:,a:b].copy_(att.transpose(1,2).contiguous().view(1,b-a,-1))

    def post(self,index):
        a,b=self.parts[index];block=self.block
        x=self.x[:,a:b]+block.dropout(block.attn_out(self.a[:,a:b]))
        norm=block.ff_norm(x)
        mlp=block.act(block.ff_proj(norm))*block.up_proj(norm)
        out=x+block.dropout(block.ff_out(mlp))
        self.output[:,a:b].copy_(out)

    def phased(self):
        # Same task shapes and stream roles as pipeline, with a global phase barrier.
        master=torch.cuda.current_stream()
        self.producer.wait_stream(master);self.attender.wait_stream(master);self.consumer.wait_stream(master)
        with torch.cuda.stream(self.producer):
            for i in range(len(self.parts)):self.produce(i)
            self.produced.record()
        with torch.cuda.stream(self.attender):
            self.attender.wait_event(self.produced)
            for i in range(len(self.parts)):self.attend(i)
            self.all_attended.record()
        with torch.cuda.stream(self.consumer):
            self.consumer.wait_event(self.all_attended)
            for i in range(len(self.parts)):self.post(i)
        master.wait_stream(self.producer);master.wait_stream(self.attender);master.wait_stream(self.consumer)
        return self.output

    def pipeline(self):
        master=torch.cuda.current_stream()
        self.producer.wait_stream(master);self.attender.wait_stream(master);self.consumer.wait_stream(master)
        with torch.cuda.stream(self.producer):
            for i in range(len(self.parts)):self.produce(i)
            self.produced.record()
        with torch.cuda.stream(self.attender):self.attender.wait_event(self.produced)
        for i in range(len(self.parts)):
            with torch.cuda.stream(self.attender):
                self.attend(i);self.attended[i].record()
            with torch.cuda.stream(self.consumer):
                self.consumer.wait_event(self.attended[i]);self.post(i)
        master.wait_stream(self.producer);master.wait_stream(self.attender);master.wait_stream(self.consumer)
        return self.output

    def close(self):
        torch.cuda.synchronize()
        if self.graph is not None:self.graph.reset()
        self.graph=None;self.graph_output=None


class GraphCall:
    """Common capture/replay setup for native, phased and pipelined operators."""
    def __init__(self,function):
        self.function=function;self.graph=None;self.output=None

    def prepare(self):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.function()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=self.function()
        torch.cuda.synchronize()

    def __call__(self):
        self.graph.replay();return self.output

    def close(self):
        torch.cuda.synchronize()
        if self.graph is not None:self.graph.reset()
        self.graph=None;self.output=None
