"""Frozen v2 backbone with matched trainable residual input/output charts."""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from relation_block.model import Model


class InputChart(nn.Module):
    def __init__(self, width, rank, device):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=.02)
        nn.init.zeros_(self.up.weight)
        self.scale = rank ** -.5

    def forward(self, embeddings):
        delta = self.up(self.down(embeddings.float())) * self.scale
        return embeddings + delta.to(embeddings.dtype)


class OutputChart(nn.Module):
    def __init__(self, hidden, vocab, rank, device):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, vocab, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=.02)
        nn.init.zeros_(self.up.weight)
        self.scale = rank ** -.5

    def forward(self, hidden, base_head):
        base = base_head(hidden)
        delta = self.up(self.down(hidden.float())) * self.scale
        return base + delta.to(base.dtype)


class FrozenChart(nn.Module):
    def __init__(self, base: Model, rank=64):
        super().__init__()
        if rank < 1:
            raise ValueError('Chart rank must be positive')
        base.requires_grad_(False)
        self.backbone = base.model
        self.base_head = base.lm_head
        self.config = base.config
        device = next(base.parameters()).device
        self.input_chart = InputChart(self.config['hidden_size'], rank, device)
        self.output_chart = OutputChart(self.config['hidden_size'], self.config['vocab_size'], rank, device)
        self.rank = rank
        self.gradient_checkpointing = True
        self._assert_frozen()

    def _assert_frozen(self):
        if any(p.requires_grad for p in self.backbone.parameters()) or any(p.requires_grad for p in self.base_head.parameters()):
            raise AssertionError('Frozen backbone or original head has trainable weights')
        if not all(p.requires_grad for p in list(self.input_chart.parameters()) + list(self.output_chart.parameters())):
            raise AssertionError('A chart parameter is frozen')

    @property
    def lm_head(self):
        return self.output_chart

    def forward(self, ids, positions, mask, past=None, cache=False, select=None, targets=None):
        x = self.input_chart(self.backbone.embed_tokens(ids))
        saved = []
        for i, layer in enumerate(self.backbone.layers):
            if self.training and self.gradient_checkpointing:
                if past is not None or cache:
                    raise ValueError('Training does not accept KV cache')
                x = checkpoint(lambda h, layer=layer: layer(h, positions, mask)[0], x, use_reentrant=False)
            else:
                x, kv = layer(x, positions, mask, None if past is None else past[i], cache)
                if cache:
                    saved.append(kv)
        x = self.backbone.norm(x)
        if select is not None:
            hidden = x[select]
            if targets is None:
                return hidden
            losses = []
            for start in range(0, len(targets), 32):
                h, t = hidden[start:start+32], targets[start:start+32]
                def head_loss(h, t):
                    logits = self.output_chart(h, self.base_head)
                    return F.cross_entropy(logits.float(), t, reduction='sum')
                losses.append(checkpoint(head_loss, h, t, use_reentrant=False))
            if not losses:
                raise ValueError('No supervised positions')
            return torch.stack(losses).sum()
        return self.output_chart(x, self.base_head), saved if cache else None

    def trainable_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.named_parameters() if v.requires_grad}

    def load_trainable_state(self, state):
        params = {k: v for k, v in self.named_parameters() if v.requires_grad}
        if set(state) != set(params):
            raise ValueError('Chart parameter keys differ')
        with torch.no_grad():
            for key, parameter in params.items():
                parameter.copy_(state[key])


def bf16_base_copy(base):
    """Same rounding as the original v2 evaluation, preserving FP32 masters."""
    with torch.device('meta'):
        copy = Model(base.config)
    state = {k: v.detach().to(dtype=torch.bfloat16, copy=True) for k, v in base.state_dict().items()}
    copy.load_state_dict(state, assign=True)
    if base.config.get('tie_word_embeddings', False):
        copy.lm_head.weight = copy.model.embed_tokens.weight
    return copy.to(next(base.parameters()).device).eval()


def evaluation_copy(base_bf16, chart):
    result = FrozenChart(base_bf16, chart.rank)
    result.load_trainable_state(chart.trainable_state())
    return result.eval()
