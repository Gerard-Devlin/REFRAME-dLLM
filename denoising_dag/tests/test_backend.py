from types import SimpleNamespace
import pytest
import torch
from denoising_dag.backend import FrozenCache,Backend,Snapshot,MASK_ID
from denoising_dag.search import Context,Prediction
from denoising_dag.generation import generate


def test_cache_capture_isolated_and_expanded_readonly():
    k=torch.randn(1,2,4,8)
    v=torch.randn_like(k)
    cache=FrozenCache.capture([(k,v)])
    before=cache[0][0].clone()
    k.zero_()
    assert torch.equal(cache[0][0],before)
    assert cache.expand(3)[0][0].shape==(3,2,4,8)
    assert cache.get_seq_length()==4
    with pytest.raises(RuntimeError,match='cannot mutate'):
        cache.update(k,v)


class SyntheticBackend:
    device='cpu'
    block_size=4

    def call(self,input_ids,past_key_values=None,**kwargs):
        length=input_ids.shape[1]+(past_key_values.get_seq_length() if past_key_values else 0)
        cache=FrozenCache(((torch.zeros(1,1,length,1),torch.zeros(1,1,length,1)),))
        logits=torch.zeros(1,1,MASK_ID+1)
        logits[...,3]=10
        return SimpleNamespace(logits=logits,past_key_values=cache)

    def make_snapshot(self,ids,cache,name):
        cache=cache or FrozenCache(())
        return Snapshot(tuple(ids),cache,Context(name,'test',tuple(range(4)),'block'))

    def predict(self,snap,states):
        return [Prediction((3,)*4,(-.1,)*4,(.4,)*4) for _ in states]


def test_complete_generation_preserves_output_and_clears_context():
    backend=SyntheticBackend()
    a=generate(backend,[1,2],False,depth=2,width=3,max_new_tokens=10)
    b=generate(backend,[1,2],True,depth=2,width=3,max_new_tokens=10)
    assert a['tokens']==b['tokens']==[3]*10
    assert a['decisions']==b['decisions']
    assert a['logical_rows']==b['logical_rows']
    assert b['search_physical_rows']<a['search_physical_rows']
    assert [r['start'] for r in b['blocks']]==[0,4,8]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU forward adapter test')
def test_backend_shift_batch_and_immutable_cache_on_gpu(monkeypatch):
    from relation_block.tests.test_full_training import model
    from relation_block.model import clean_mask
    class OfficialInterface(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner=model().cuda().eval()

        def forward(self,input_ids,position_ids,past_key_values,block_size,**kwargs):
            past=past_key_values.pairs or None
            mask=clean_mask(input_ids.shape[1],block_size,input_ids.device,past=past_key_values.get_seq_length())
            logits,_=self.inner(input_ids,position_ids,mask,past=past)
            return SimpleNamespace(logits=logits)
    interface=OfficialInterface().eval()
    backend=Backend(interface,None,'tiny',block_size=4,batch_size=2)
    snap=backend.make_snapshot([2,3,4,5],None,'one')
    actual=backend.logits(snap,[snap.ids,snap.ids])
    original=interface(input_ids=torch.tensor([snap.ids,snap.ids],device='cuda'),
        position_ids=torch.arange(4,device='cuda')[None],past_key_values=FrozenCache(()),block_size=4).logits
    assert torch.equal(actual[:,1:],original[:,:-1])
    assert torch.equal(actual[:,:1],original[:,:1])
    stats=backend.stats()
    assert stats['forward_calls']==1 and stats['physical_rows']==2
    assert stats['model_gpu_seconds']>0
    monkeypatch.setattr('denoising_dag.backend.MASK_ID',40)
    predictions=backend.predict(snap,[snap.ids,snap.ids])
    shifted=actual.float()
    shifted[...,40]=-torch.inf
    expected=shifted.log_softmax(-1)
    assert predictions[0].token==tuple(expected[0].argmax(-1).tolist())
    assert predictions[0].logp==tuple(expected[0].max(-1).values.tolist())
    assert all(x>=0 for x in predictions[0].entropy)
