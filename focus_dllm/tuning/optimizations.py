"""Compare rotated-key reuse against both v1 cache baselines."""
import torch

from . import experimental
from .rotation import generate_rotated
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import prompt_ids
from ..llada_decode import generate_dual_cache
from ..llada_evaluate import postprocess_output

experimental.run.VARIANTS.update({
    'v1_dual': dict(kind='dual', layer=4, keep=1., threshold=.90),
    'rot00_l4': dict(kind='rotated', layer=4, keep=0., threshold=.90),
    'rot12_l4': dict(kind='rotated', layer=4, keep=.125, threshold=.90),
    'rot00_l8': dict(kind='rotated', layer=8, keep=0., threshold=.90),
})
previous_run = experimental.experimental_run


def run_one(model, tokenizer, sample, config, gen_length):
    if config['kind'] not in {'dual', 'rotated'}:
        return previous_run(model, tokenizer, sample, config, gen_length)
    ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
    prompt = torch.tensor([ids], device=model.device)
    with LLaDAAttentionBackend(model, 'flash') as backend:
        if config['kind'] == 'dual':
            result = generate_dual_cache(model, prompt, gen_length=gen_length,
                        block_length=32, threshold=config['threshold'])
        else:
            result, _ = generate_rotated(model, prompt, gen_length=gen_length,
                        layer=config['layer'], keep=config['keep'], threshold=config['threshold'])
    tokens = result.output[0, len(ids):].tolist()
    text, count = postprocess_output(tokenizer, tokens, sample, 'gsm8k')
    return dict(text=text, token_ids=tokens, nfe=result.nfe, seconds=result.seconds,
                peak_gib=result.peak_gib, backend=backend.report(), output_tokens=count,
                truncated=126081 not in tokens, canvas_tokens=gen_length)


experimental.run.run_one = run_one

if __name__ == '__main__':
    experimental.run.main()
