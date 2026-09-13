"""Parallel table lookups for SMALL vocabularies; deliberately not BPE-scalable.

Dense 257x257 tables are appropriate for this byte pilot. Do not allocate
vocabulary-squared tables when later adapting the sparse codec to LLaDA.
"""
import numpy as np
import torch
from torch import nn

from .codec import Codec, PointwiseRename, SparseCoupling, complete_sparse_permutation, fit_coupling


def codec_spec(kind, train, prefix, depth=1, seed=1234, top_k=16):
    vocab = 257
    if kind not in {"identity", "rename", "random", "relation"} or depth not in (1, 2):
        raise ValueError("Unknown codec or unsupported depth")
    spec = dict(kind=kind, vocab_size=vocab, prefix=prefix, length=train.shape[1], layers=[])
    z = train.copy()
    rng = np.random.default_rng(seed)
    if kind in {"identity", "rename"}:
        return spec
    for level in range(depth):
        pairs = tuple((i, i + 1) for i in range(level % 2, train.shape[1] - prefix - 1, 2))
        if kind == "relation":
            layer = fit_coupling(z, vocab, pairs, top_k=top_k, min_count=2,
                                 protected_ids=(256,), prefix_len=prefix)
        else:
            # Same sparse target/code count as the statistical candidate,
            # with random sources. EOS and its anchor rows remain identity.
            tables = {a: complete_sparse_permutation(rng.choice(256, top_k, replace=False),
                                                     range(top_k)) for a in range(256)}
            layer = SparseCoupling(vocab, pairs, tables)
        spec["layers"].append(dict(pairs=layer.pairs, tables=layer.tables))
        table = dense_tables(layer)[0]
        original = z.copy()
        for a, b in pairs:
            z[:, b + prefix] = table[original[:, a + prefix], original[:, b + prefix]]
    return spec


def dense_tables(layer):
    v = layer.vocab_size
    if v > 1024:
        raise ValueError("Dense byte pilot only; BPE requires a sparse GPU implementation")
    forward = np.tile(np.arange(v, dtype=np.int64), (v, 1))
    for a, table in layer.tables.items():
        for source, target in table.items():
            forward[a, source] = target
    inverse = np.argsort(forward, axis=1).astype(np.int64)
    return forward, inverse


class TorchCodec(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec, self.prefix, self.kind = spec, spec["prefix"], spec["kind"]
        if spec["vocab_size"] != 257:
            raise ValueError("This training pilot only supports bytes plus boundary")
        for i, raw in enumerate(spec["layers"]):
            layer = SparseCoupling(257, raw["pairs"], raw["tables"])
            f, inv = dense_tables(layer)
            self.register_buffer(f"forward_{i}", torch.from_numpy(f))
            self.register_buffer(f"inverse_{i}", torch.from_numpy(inv))
            self.register_buffer(f"anchors_{i}", torch.tensor([a + self.prefix for a, b in layer.pairs], dtype=torch.long))
            self.register_buffer(f"targets_{i}", torch.tensor([b + self.prefix for a, b in layer.pairs], dtype=torch.long))

    def apply_code(self, tokens, inverse=False):
        if tokens.ndim != 2 or tokens.shape[1] != self.spec["length"]:
            raise ValueError("Codec length mismatch")
        z = tokens.clone()
        if self.kind == "rename":
            body = z[:, self.prefix:]
            z[:, self.prefix:] = torch.where(body == 256, body,
                                             (body + (-1 if inverse else 1)) % 256)
        order = range(len(self.spec["layers"]))
        if inverse:
            order = reversed(order)
        for i in order:
            table = getattr(self, f"{'inverse' if inverse else 'forward'}_{i}")
            a, b = getattr(self, f"anchors_{i}"), getattr(self, f"targets_{i}")
            z[:, b] = table[z[:, a], z[:, b]]
        return z

    def encode(self, tokens):
        return self.apply_code(tokens)

    def decode(self, tokens):
        return self.apply_code(tokens, inverse=True)
