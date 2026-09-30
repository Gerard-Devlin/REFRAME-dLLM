"""Next sweep: remove pruning synchronization, or preserve discarded K/V."""
import torch

from . import run
from .tensor_pruning import generate_tensor
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import prompt_ids
from ..llada_evaluate import postprocess_output

run.VARIANTS.update({
    'gpu31_l4': dict(kind='tensor', layer=4, keep=.3125, threshold=.90),
    'gpu50_l4': dict(kind='tensor', layer=4, keep=.5, threshold=.90),
    'ref00_l4': dict(kind='reference', layer=4, keep=0., threshold=.90),
    'ref12_l4': dict(kind='reference', layer=4, keep=.125, threshold=.90),
    'ref31_l4': dict(kind='reference', layer=4, keep=.3125, threshold=.90),
    'ref12_l8': dict(kind='reference', layer=8, keep=.125, threshold=.90),
})
original_run = run.run_one


def experimental_run(model, tokenizer, sample, config, gen_length):
    if config['kind'] not in {'tensor', 'reference'}:
        return original_run(model, tokenizer, sample, config, gen_length)
    ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
    with LLaDAAttentionBackend(model, 'flash') as backend:
        result, _ = generate_tensor(model, torch.tensor([ids], device=model.device),
                    reference=config['kind'] == 'reference', gen_length=gen_length,
                    layer=config['layer'], keep=config['keep'], threshold=config['threshold'])
    tokens = result.output[0, len(ids):].tolist()
    text, count = postprocess_output(tokenizer, tokens, sample, 'gsm8k')
    return dict(text=text, token_ids=tokens, nfe=result.nfe, seconds=result.seconds,
                peak_gib=result.peak_gib, backend=backend.report(), output_tokens=count,
                truncated=126081 not in tokens, canvas_tokens=gen_length)


run.run_one = experimental_run

if __name__ == '__main__':
    run.main()
