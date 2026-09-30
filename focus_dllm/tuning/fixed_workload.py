"""Same input, canvas, targets and KV for all measured refinement forwards."""
import argparse
import json
from pathlib import Path
import time

import torch

from ..common import write_json
from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import prompt_ids, MASK_ID
from ..llada_decode import _selected_positions
from ..llada_evaluate import load_model
from ..llada_pruning import Config
from .backend import selected_forward
from .tensor_pruning import ReferenceForward, TensorForward
from .rotation import RotatedForward
from .zero_support import ZeroForward


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = load_model('cuda:0')
    sample = json.loads(args.dataset.read_text())['development'][0]
    ids = prompt_ids(tokenizer, sample['paper_prompt'], 'gsm8k', preformatted=True)
    x = torch.full((1, len(ids)+256), MASK_ID, device=model.device)
    x[0, :len(ids)] = torch.tensor(ids, device=model.device)
    target = torch.arange(len(ids), len(ids)+32, device=model.device)
    with LLaDAAttentionBackend(model, 'flash'):
        warm = selected_forward(model, x, target, use_cache=True)
        top = warm.logits.argmax(-1)
        conf = warm.logits.double().softmax(-1).gather(-1, top.unsqueeze(-1)).squeeze(-1)
        commit = _selected_positions(conf, .90)
        x[0, target[commit]] = top[0, commit]
        local = (x[0, len(ids):len(ids)+32] == MASK_ID).nonzero().flatten()
        if not len(local):
            raise ValueError('Fixed input has no remaining targets')
        suffix = x[:, len(ids):]
        past = [tuple(t[:, :, :len(ids)] for t in pair) for pair in warm.past_key_values]
        candidates = {}
        for cls in (TensorForward, ReferenceForward, RotatedForward, ZeroForward):
            obj = cls(model, Config(prune_after_layer=4, support_keep_ratio=0., target_only_head=True))
            obj.reference = warm.past_key_values
            candidates[cls.__name__] = obj
        functions = {
            'v1_full_head': lambda: model(suffix, past_key_values=past, use_cache=True).logits[:, local],
            'active_exact': lambda: selected_forward(model, suffix, local, past_key_values=past, use_cache=False).logits,
        }
        for name, forward in candidates.items():
            functions[name] = lambda forward=forward: forward(suffix, local, past_key_values=past)
        logits, metrics = {}, {}
        for name, fn in functions.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits[name] = fn()
            torch.cuda.synchronize()
            cold = time.perf_counter()-start
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(12):
                fn()
            torch.cuda.synchronize()
            metrics[name] = dict(first_call_ms=cold*1000,
                                 steady_mean_ms=(time.perf_counter()-start)*1000/12)
        a, b = logits['ReferenceForward'], logits['RotatedForward']
        def action(z):
            top = z.argmax(-1)
            conf = z.double().softmax(-1).gather(-1, top.unsqueeze(-1)).squeeze(-1)
            take = _selected_positions(conf, .90)
            return take, top[0, take]
        parity = dict(max_logit_error=float((a.float()-b.float()).abs().max()),
                      top1_equal=torch.equal(a.argmax(-1), b.argmax(-1)),
                      actions_equal=all(torch.equal(u,v) for u,v in zip(action(a), action(b))))
        if not parity['actions_equal']:
            raise AssertionError(f'Rotated-key implementation action mismatch: {parity}')
    result = dict(metrics=metrics, rotation_parity=parity,
                  input_tokens=len(ids), canvas_tokens=256, active_positions=len(local),
                  scope='Fixed-state microbenchmark. Real generation includes per-block setup cost.')
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
