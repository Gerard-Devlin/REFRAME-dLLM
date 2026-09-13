"""Small bidirectional masked denoiser and fixed-path distributions.

This is a from-scratch byte model, not a loader for pretrained LLaDA weights.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 257  # bytes 0..255, boundary 256; MASK is 257
    length: int = 128
    width: int = 384
    layers: int = 6
    heads: int = 6

    def __post_init__(self):
        if min(self.vocab_size, self.length, self.width, self.layers, self.heads) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, cfg.width * 3)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
                                 nn.Linear(cfg.width * 4, cfg.width))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), dropout_p=0.0, is_causal=False)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class Denoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size + 1, cfg.width)
        self.position = nn.Embedding(cfg.length, cfg.width)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size)

    def forward(self, x):
        h = self.token(x) + self.position(torch.arange(x.shape[1], device=x.device))
        return self.head(self.norm(self.blocks(h)))


def update_batch(n_rows, batch_size, length, prefix, seed, step, objective="diffusion"):
    """Stateless global batches/noise, identical across codecs and world sizes.

    Construct on CPU, then partition the global batch across DDP ranks. No
    dropout is used, so resume does not require rank-specific RNG state.
    """
    if not 0 <= prefix < length or n_rows < 1 or batch_size < 1:
        raise ValueError("Invalid batch dimensions")
    if objective not in {"diffusion", "one-step"}:
        raise ValueError("Unknown objective")
    gen = torch.Generator().manual_seed(seed + 1000003 * step)
    indices = torch.randint(n_rows, (batch_size,), generator=gen)
    p = 0.001 + 0.999 * torch.rand(batch_size, 1, generator=gen)
    draws = torch.rand(batch_size, length, generator=gen)
    if objective == "one-step":
        p.fill_(1)
    mask = draws < p
    mask[:, :prefix] = False
    return indices, mask, p


def masked_loss(logits, clean, mask, probability, prefix):
    # Keep a connected, finite zero loss even when a microbatch has no masks.
    ce = F.cross_entropy(logits.float().transpose(1, 2), clean, reduction="none")
    return (ce * mask / probability).sum() / (clean.shape[0] * (clean.shape[1] - prefix))


def reveal_groups(length, prefix, steps):
    if not 0 <= prefix < length or not 1 <= steps <= length - prefix:
        raise ValueError("Need 1 <= steps <= response length")
    return [(prefix + i * (length - prefix) // steps,
             prefix + (i + 1) * (length - prefix) // steps) for i in range(steps)]


@torch.no_grad()
def path_nll(model, clean_codes, prefix, steps):
    """Exact NLL of this FIXED reveal path, not a marginalized diffusion NLL.

    Given an invertible response code and unchanged prefix, it is also the
    original sequence NLL under the induced path distribution. Reveal only
    previous ground-truth groups; never expose the group currently scored.
    """
    x = torch.full_like(clean_codes, model.cfg.vocab_size)
    x[:, :prefix] = clean_codes[:, :prefix]
    losses = torch.zeros(len(x), device=x.device, dtype=torch.float32)
    for start, end in reveal_groups(x.shape[1], prefix, steps):
        logits = model(x)[:, start:end].float()
        losses += F.cross_entropy(logits.transpose(1, 2), clean_codes[:, start:end],
                                  reduction="none").sum(1)
        x[:, start:end] = clean_codes[:, start:end]
    return losses


@torch.no_grad()
def sample(model, prefix_tokens, length, steps, generator=None, greedy=False):
    prefix = prefix_tokens.shape[1]
    x = torch.full((len(prefix_tokens), length), model.cfg.vocab_size,
                   device=prefix_tokens.device, dtype=torch.long)
    x[:, :prefix] = prefix_tokens
    for start, end in reveal_groups(length, prefix, steps):
        logits = model(x)[:, start:end].float()
        if greedy:
            pred = logits.argmax(-1)
        else:
            pred = torch.multinomial(logits.softmax(-1).flatten(0, 1), 1,
                                     generator=generator).reshape(len(x), end - start)
        x[:, start:end] = pred
    return x
