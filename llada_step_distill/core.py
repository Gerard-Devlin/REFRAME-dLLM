from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
MASK_ID = 126336
EOS_ID = 126081
BLOCK_LENGTH = 32
GEN_LENGTH = 512
MAX_PROMPT_LENGTH = 512
MAX_SEQUENCE_LENGTH = MAX_PROMPT_LENGTH + GEN_LENGTH
SEED = 1234

DOMAIN_COUNTS = {
    "math": 22_066_397,
    "code": 10_108_883,
    "science": 708_920,
    "chat": 39_792,
    "safety": 31_426,
}
TRAIN_QUOTAS = {
    "math": 6_695_833,
    "code": 3_067_442,
    "science": 215_115,
    "chat": 12_074,
    "safety": 9_536,
}
assert sum(TRAIN_QUOTAS.values()) == 10_000_000

DATA_FILES = {
    "math": ("SFT/math/math_v1.jsonl", "SFT/math/math_v1.1.jsonl"),
    "code": ("SFT/code/code_v1.jsonl", "SFT/code/code_v1.1.jsonl"),
    "science": ("SFT/science/science.jsonl",),
    "chat": ("SFT/chat/chat.jsonl",),
    "safety": ("SFT/safety/safety.jsonl",),
}


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = MODEL_ID
    revision: str = REVISION
    rank: int = 256
    alpha: int = 512
    dropout: float = 0.0
    max_prompt_length: int = MAX_PROMPT_LENGTH
    generation_length: int = GEN_LENGTH
    block_length: int = BLOCK_LENGTH
    seed: int = SEED
    margin: float = 0.2
    anchor_positions: int = 8
    anchor_topk: int = 32
    stage_a_lr: float = 1e-5
    stage_b_lr: float = 5e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accumulation: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def word_ngrams(text: str, n: int = 5) -> frozenset[tuple[str, ...]]:
    words = normalize_text(text).split()
    if len(words) < n:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


class Decontaminator:
    """Exact and indexed 5-gram matching against the small GSM8K corpus."""

    def __init__(self, questions: Sequence[str], threshold: float = 0.8):
        self.threshold = threshold
        self.normalized = {normalize_text(q) for q in questions}
        self.grams: list[frozenset] = []
        self.index: dict[tuple[str, ...], list[int]] = {}
        for question in questions:
            grams = word_ngrams(question)
            idx = len(self.grams)
            self.grams.append(grams)
            for gram in grams:
                self.index.setdefault(gram, []).append(idx)

    def contaminated(self, text: str) -> bool:
        normalized = normalize_text(text)
        if normalized in self.normalized:
            return True
        grams = word_ngrams(text)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(self.index.get(gram, ()))
        return any(jaccard(grams, self.grams[i]) >= self.threshold for i in candidates)


def user_text(messages: Sequence[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")


def stable_u64(*parts: object) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "big")


def validation_quotas(total: int = 20_000) -> dict[str, int]:
    count = sum(DOMAIN_COUNTS.values())
    raw = {k: total * v / count for k, v in DOMAIN_COUNTS.items()}
    out = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[key] += 1
    assert sum(out.values()) == total
    return out


def choose_kind(sample_id: str) -> str:
    """Exactly one deterministic retention bucket in ten in expectation."""
    return "retention" if stable_u64("kind", sample_id) % 10 == 0 else "acceleration"


def is_real_transition(sample_id: str) -> bool:
    """Approximately 200k of 9M acceleration rows, finalized exactly in prepare."""
    return stable_u64("real", sample_id) % 45 == 0


def shard_order(count: int, seed: int) -> list[int]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


def shuffled_indices(count: int, seed: int) -> list[int]:
    return shard_order(count, seed)


def schedule_factor(step: int, total: int, warmup_ratio: float = 0.03) -> float:
    warmup = max(1, math.ceil(total * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def bootstrap_interval(differences: Sequence[float], samples: int = 10_000, seed: int = SEED) -> list[float]:
    if not differences:
        raise ValueError("empty paired differences")
    rng = random.Random(seed)
    n = len(differences)
    values = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    return [values[int(0.025 * samples)], values[int(0.975 * samples) - 1]]


def iter_jsonl(paths: Iterable[Path]) -> Iterator[tuple[str, int, dict]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if line.strip():
                    yield str(path), line_number, json.loads(line)


def implementation_hash(root: Path) -> str:
    files = sorted(Path(root).glob("*.py"))
    return digest({p.name: file_sha256(p) for p in files})
