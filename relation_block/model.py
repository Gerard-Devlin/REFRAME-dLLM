"""Explicit Qwen2 backbone for controlled masks and selective loss heads.

Loads the original safetensors with strict key checking. No remote Python is
executed by this backend. Server preflight MUST compare it against the pinned
official implementation before training. This backend is not the official
optimized inference benchmark.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return self.weight * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.h, self.k = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        self.d = c.get("head_dim") or d // self.h
        self.groups = self.h // self.k
        self.q_proj = nn.Linear(d, self.h * self.d, bias=True)
        self.k_proj = nn.Linear(d, self.k * self.d, bias=True)
        self.v_proj = nn.Linear(d, self.k * self.d, bias=True)
        self.o_proj = nn.Linear(self.h * self.d, d, bias=False)
        self.theta = c.get("rope_theta", 1000000.)

    def forward(self, x, positions, mask, past=None, cache=False):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.k, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.k, self.d).transpose(1, 2)
        # Recompute in FP32: module.to(bfloat16) must not round RoPE frequencies.
        inv_freq = 1 / (self.theta ** (torch.arange(0, self.d, 2, device=x.device).float() / self.d))
        angles = positions.float()[..., None] * inv_freq
        angles = torch.cat((angles, angles), -1)
        co, si = angles.cos().to(q.dtype).unsqueeze(1), angles.sin().to(q.dtype).unsqueeze(1)
        def rotate(t):
            return torch.cat((-t[..., self.d // 2:], t[..., :self.d // 2]), -1)
        q, k = q * co + rotate(q) * si, k * co + rotate(k) * si
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        # Match Transformers' SDPA wrapper for masked CUDA attention. With a
        # block mask it explicitly expands KV heads instead of enable_gqa=True.
        # The two paths are algebraically equivalent but use different BF16
        # kernels and diverge measurably after many layers. Keep the compact KV
        # form in cache and expand only the tensors passed to attention.
        saved = (k, v) if cache else None
        if self.groups != 1:
            k_attn = k[:, :, None, :, :].expand(b, self.k, self.groups, k.shape[-2], self.d)
            v_attn = v[:, :, None, :, :].expand(b, self.k, self.groups, v.shape[-2], self.d)
            k_attn = k_attn.reshape(b, self.h, k.shape[-2], self.d)
            v_attn = v_attn.reshape(b, self.h, v.shape[-2], self.d)
        else:
            k_attn, v_attn = k, v
        out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask,
                                             is_causal=False, scale=self.d ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1)), saved


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, m = c["hidden_size"], c["intermediate_size"]
        self.gate_proj = nn.Linear(d, m, bias=False)
        self.up_proj = nn.Linear(d, m, bias=False)
        self.down_proj = nn.Linear(m, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def forward(self, x, positions, mask, past=None, cache=False):
        a, kv = self.self_attn(self.input_layernorm(x), positions, mask, past, cache)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([Layer(c) for _ in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("rope_scaling") or config.get("use_sliding_window", False):
            raise ValueError("This pilot supports default RoPE and full attention only")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Unsupported activation")
        self.config = config
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.gradient_checkpointing = False

    def forward(self, ids, positions, mask, past=None, cache=False, select=None):
        x = self.model.embed_tokens(ids)
        saved = []
        for i, layer in enumerate(self.model.layers):
            if self.training and self.gradient_checkpointing:
                if past is not None or cache:
                    raise ValueError("Training checkpoint path does not accept KV cache")
                x = checkpoint(lambda h, layer=layer: layer(h, positions, mask)[0],
                               x, use_reentrant=False)
            else:
                x, kv = layer(x, positions, mask, None if past is None else past[i], cache)
                if cache:
                    saved.append(kv)
        x = self.model.norm(x)
        # During training return selected hidden states; compute vocab loss in
        # recomputed chunks to avoid retaining B*2L*150k logits.
        if select is not None:
            return x[select]
        return self.lm_head(x), saved if cache else None

    @classmethod
    def load(cls, directory, device="cuda", dtype=torch.bfloat16):
        from safetensors.torch import load_file
        directory = Path(directory)
        c = json.loads((directory / "config.json").read_text())
        # Six DDP ranks must not each allocate a redundant FP32 model before
        # loading BF16 weights. Meta initialization avoids that RAM spike.
        with torch.device("meta"):
            m = cls(c)
        state = {}
        for p in sorted(directory.glob("model*.safetensors")):
            state.update(load_file(p))
        if not state:
            raise ValueError("Missing model safetensors")
        if c.get("tie_word_embeddings", False):
            state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])
        state = {n: t.to(dtype=dtype) for n, t in state.items()}
        m.load_state_dict(state, strict=True, assign=True)
        if c.get("tie_word_embeddings", False):
            m.lm_head.weight = m.model.embed_tokens.weight
        return m.to(device)


class LoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return self.base(x) + delta.to(x.dtype)


def add_lora(model, rank=16):
    model.requires_grad_(False)
    for layer in model.model.layers:
        for owner, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                             (layer.mlp, ("up_proj", "gate_proj", "down_proj"))):
            for name in names:
                setattr(owner, name, LoRA(getattr(owner, name), rank, 2 * rank))


def adapter_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    add_lora(model, ckpt["meta"]["rank"])
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(ckpt["adapter"]) != expected:
        raise ValueError("Adapter parameter mismatch")
    with torch.no_grad():
        for n, x in ckpt["adapter"].items():
            params[n].copy_(x)
    return ckpt["meta"]


def clean_mask(length, block_size, device, past=0):
    q = torch.arange(past, past + length, device=device)
    k = torch.arange(past + length, device=device)
    return (q[:, None] // block_size >= k[None, :] // block_size)[None, None]


def training_mask(length, block_size, device):
    idx = torch.arange(2 * length, device=device)
    clean = idx >= length
    blocks = (idx % length) // block_size
    nq, nk = ~clean[:, None], ~clean[None, :]
    same = blocks[:, None] == blocks[None, :]
    earlier = blocks[:, None] > blocks[None, :]
    allowed = (nq & nk & same) | (nq & ~nk & earlier) | (~nq & ~nk & (same | earlier))
    return allowed[None, None]
