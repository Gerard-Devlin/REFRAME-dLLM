"""Conservative periodic history refresh ablation; same native commits."""
import torch

from . import fast_zero
from .streaming import generate_stream
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import prompt_ids
from ..llada_evaluate import postprocess_output

fast_zero.optimizations.experimental.run.VARIANTS.update({
    'stream2_l4': dict(kind='stream',layer=4,keep=0.,threshold=.90,refresh=2),
    'stream4_l4': dict(kind='stream',layer=4,keep=0.,threshold=.90,refresh=4),
    'stream8_l4': dict(kind='stream',layer=4,keep=0.,threshold=.90,refresh=8),
    'stream4_full': dict(kind='stream',layer=4,keep=1.,threshold=.90,refresh=4),
})
previous_run = fast_zero.run_one


def run_one(model,tokenizer,sample,config,gen_length):
    if config['kind'] != 'stream':
        return previous_run(model,tokenizer,sample,config,gen_length)
    ids = prompt_ids(tokenizer,sample['paper_prompt'],'gsm8k',preformatted=True)
    with LLaDAAttentionBackend(model,'flash') as backend:
        result,_,warm_queries = generate_stream(model,torch.tensor([ids],device=model.device),
            gen_length=gen_length,layer=config['layer'],threshold=config['threshold'],refresh_every=config['refresh'],keep=config['keep'])
    tokens = result.output[0,len(ids):].tolist()
    text,count = postprocess_output(tokenizer,tokens,sample,'gsm8k')
    return dict(text=text,token_ids=tokens,nfe=result.nfe,seconds=result.seconds,
        peak_gib=result.peak_gib,backend=backend.report(),output_tokens=count,
        truncated=126081 not in tokens,canvas_tokens=gen_length,warm_query_tokens=warm_queries)


fast_zero.optimizations.experimental.run.run_one = run_one

if __name__ == '__main__':
    fast_zero.optimizations.experimental.run.main()
