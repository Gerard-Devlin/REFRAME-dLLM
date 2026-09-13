"""NumPy reference codecs, not GPU kernels or a pretrained-model adapter.

All positions in coupling pairs are relative to the response (after prefix_len).
Only complete clean sequences may be encoded/decoded; MASK is out of vocabulary.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


def checked(x, vocab_size, prefix_len=0):
    x = np.asarray(x)
    if vocab_size < 2 or x.ndim != 2 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Expected integer [batch, length] and vocab_size >= 2")
    if not 0 <= prefix_len <= x.shape[1]:
        raise ValueError("Invalid prefix length")
    if np.any(x < 0) or np.any(x >= vocab_size):
        raise ValueError("Encode/decode clean tokens only; MASK must stay outside the vocabulary")
    return x.astype(np.int64, copy=False)


def complete_sparse_permutation(sources, codes):
    """Assign distinct sources to codes, completing a bijection by sparse swaps."""
    sources, codes = tuple(sources), tuple(codes)
    if len(sources) != len(codes) or len(set(sources)) != len(sources) or len(set(codes)) != len(codes):
        raise ValueError("Sources and codes must be distinct and have equal length")
    forward, backward = {}, {}
    for source, code in zip(sources, codes):
        old_code = forward.get(source, source)
        displaced_source = backward.get(code, code)
        forward[source], forward[displaced_source] = code, old_code
        backward[code], backward[old_code] = source, displaced_source
    return {int(k): int(v) for k, v in forward.items() if k != v}


@dataclass
class SparseCoupling:
    """A shallow conditional permutation with a fixed, answer-independent partition.

    tables[anchor_token] is a sparse permutation of target tokens. Missing entries
    are identity. Anchors and targets are disjoint within a layer, so every target
    can be decoded in parallel. Different layers invert in reverse order.
    """

    vocab_size: int
    pairs: tuple
    tables: dict

    def __post_init__(self):
        self.pairs = tuple(tuple(pair) for pair in self.pairs)
        anchors = {a for a, _ in self.pairs}
        targets = [b for _, b in self.pairs]
        if anchors.intersection(targets) or len(set(targets)) != len(targets):
            raise ValueError("Anchors must not be targets; each target is written once")
        if any(min(pair) < 0 for pair in self.pairs):
            raise ValueError("Pair positions must be nonnegative")
        self.tables = {int(a): {int(k): int(v) for k, v in t.items() if k != v}
                       for a, t in self.tables.items()}
        for anchor, table in self.tables.items():
            ids = [anchor, *table.keys(), *table.values()]
            if any(i < 0 or i >= self.vocab_size for i in ids):
                raise ValueError("Permutation contains out-of-vocabulary tokens")
            if set(table) != set(table.values()) or len(set(table.values())) != len(table):
                raise ValueError("Sparse table must complete an actual permutation")
        self.inverse_tables = {a: {v: k for k, v in t.items()} for a, t in self.tables.items()}

    def _apply(self, x, prefix_len, inverse):
        x = checked(x, self.vocab_size, prefix_len)
        if self.pairs and max(max(pair) for pair in self.pairs) >= x.shape[1] - prefix_len:
            raise ValueError("Coupling pair exceeds response length")
        out = x.copy()
        tables = self.inverse_tables if inverse else self.tables
        for a, b in self.pairs:
            a, b = a + prefix_len, b + prefix_len
            for anchor, table in tables.items():
                rows = x[:, a] == anchor
                for source, target in table.items():
                    out[rows & (x[:, b] == source), b] = target
        return out

    def encode(self, x, prefix_len=0):
        return self._apply(x, prefix_len, False)

    def decode(self, z, prefix_len=0):
        return self._apply(z, prefix_len, True)


def fit_coupling(train_tokens, vocab_size, pairs, top_k=2, min_count=2,
                 protected_ids=(), prefix_len=0):
    """Fit sparse conditional rank codes on TRAINING data only, then freeze.

    This simple frequency heuristic does not optimize the denoising objective.
    No vocabulary-squared table is allocated. Protected ids stay unchanged.
    """
    train_tokens = checked(train_tokens, vocab_size, prefix_len)
    if top_k < 1 or min_count < 1:
        raise ValueError("top_k and min_count must be positive")
    pairs = tuple(pairs)
    SparseCoupling(vocab_size, pairs, {}).encode(train_tokens, prefix_len)
    protected = set(protected_ids)
    if any(i < 0 or i >= vocab_size for i in protected):
        raise ValueError("Protected ids must be in vocabulary")
    codes = [i for i in range(vocab_size) if i not in protected][:top_k]
    counts = defaultdict(Counter)
    for a, b in pairs:
        joint, freqs = np.unique(train_tokens[:, [a + prefix_len, b + prefix_len]], axis=0,
                                return_counts=True)
        for (anchor, target), count in zip(joint, freqs):
            if anchor not in protected and target not in protected:
                counts[int(anchor)][int(target)] += int(count)
    tables = {}
    for anchor, row in counts.items():
        ordered = sorted(row, key=lambda token: (-row[token], token))
        sources = [token for token in ordered if row[token] >= min_count][:len(codes)]
        tables[anchor] = complete_sparse_permutation(sources, codes[:len(sources)])
    return SparseCoupling(vocab_size, pairs, tables)


@dataclass
class Codec:
    vocab_size: int
    layers: tuple = ()

    def __post_init__(self):
        if any(layer.vocab_size != self.vocab_size for layer in self.layers):
            raise ValueError("All layers must use the same vocabulary")

    def encode(self, x, prefix_len=0):
        z = checked(x, self.vocab_size, prefix_len).copy()
        for layer in self.layers:
            z = layer.encode(z, prefix_len)
        return z

    def decode(self, z, prefix_len=0):
        x = checked(z, self.vocab_size, prefix_len).copy()
        for layer in reversed(self.layers):
            x = layer.decode(x, prefix_len)
        return x


@dataclass
class PointwiseRename:
    vocab_size: int

    def encode(self, x, prefix_len=0):
        out = checked(x, self.vocab_size, prefix_len).copy()
        out[:, prefix_len:] = (out[:, prefix_len:] + 1) % self.vocab_size
        return out

    def decode(self, z, prefix_len=0):
        out = checked(z, self.vocab_size, prefix_len).copy()
        out[:, prefix_len:] = (out[:, prefix_len:] - 1) % self.vocab_size
        return out


@dataclass
class ChainDifference:
    """Diagnostic reference: arithmetic ids are not proposed language relations.

    Its inverse is a prefix scan, unlike one/two shallow coupling layers. A wrong
    anchor can affect the whole suffix. Keep its results separate from the method.
    """

    vocab_size: int

    def encode(self, x, prefix_len=0):
        x = checked(x, self.vocab_size, prefix_len)
        out = x.copy()
        if x.shape[1] > prefix_len:
            out[:, prefix_len + 1:] = (x[:, prefix_len + 1:] - x[:, prefix_len:-1]) % self.vocab_size
        return out

    def decode(self, z, prefix_len=0):
        out = checked(z, self.vocab_size, prefix_len).copy()
        out[:, prefix_len:] = np.cumsum(out[:, prefix_len:], axis=1) % self.vocab_size
        return out
