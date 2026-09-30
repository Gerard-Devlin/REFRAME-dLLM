"""Specialized future-query freeze ablation with reused static K/V layouts."""
import torch

from . import optimizations
from .zero_support import generate_zero
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import prompt_ids
from ..llada_evaluate import postprocess_output

optimizations.experimental.run.VARIANTS.update({
    'zero_l1': dict(kind='zero', layer=1, keep=0., threshold=.90),
    'zero_l2': dict(kind='zero', layer=2, keep=0., threshold=.90),
    'zero_l4': dict(kind='zero', layer=4, keep=0., threshold=.90),
})
previous_run = optimizations.run_one


def run_one(model, tokenizer, sample, config, gen_length):
    if config['kind'] != 'zero':
        return previous_run(model, tokenizer, sample, config, gen_length)
    ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
    with LLaDAAttentionBackend(model, 'flash') as backend:
        result, _ = generate_zero(model, torch.tensor([ids], device=model.device),
                    gen_length=gen_length, layer=config['layer'], keep=0., threshold=config['threshold'])
    tokens = result.output[0, len(ids):].tolist()
    text, count = postprocess_output(tokenizer, tokens, sample, 'gsm8k')
    return dict(text=text, token_ids=tokens, nfe=result.nfe, seconds=result.seconds,
                peak_gib=result.peak_gib, backend=backend.report(), output_tokens=count,
                truncated=126081 not in tokens, canvas_tokens=gen_length)


optimizations.experimental.run.run_one = run_one

if __name__ == '__main__':
    optimizations.experimental.run.main()
