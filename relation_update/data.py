"""Prompt-disjoint native trajectories; no reference answers in head training."""

from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset


INPUT_KEYS = ("hidden", "candidates", "base_log_probs", "commit_ids", "committed", "eligible")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def prompt_key(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_prompts(data, train_limit, heldout_limit, seed):
    """Read only the prompt prefix; deduplicate before any trajectory sampling."""
    data = Path(data)
    splits = {}
    for split, filename in (("train", "train.json"), ("heldout", "heldout.json")):
        unique = {}
        for row in json.loads((data / filename).read_text(encoding="utf-8")):
            prefix = int(row["prefix"])
            if not 0 < prefix <= len(row["ids"]):
                raise ValueError("Invalid prepared prompt prefix")
            ids = row["ids"][:prefix]
            key = prompt_key(ids)
            unique.setdefault(key, dict(id=key, ids=ids))
        splits[split] = unique
    # Exclude overlaps with ALL held-out prompts, not only the selected subset.
    for key in splits["heldout"]:
        splits["train"].pop(key, None)
    result = {}
    for split, limit in (("train", train_limit), ("heldout", heldout_limit)):
        rows = sorted(splits[split].values(), key=lambda row: row["id"])
        random.Random(seed + (split == "heldout")).shuffle(rows)
        if limit < 1 or limit > len(rows):
            raise ValueError(f"Requested {limit} {split} prompts; only {len(rows)} unique prompts")
        result[split] = rows[:limit]
    return result


def coarsen(logits, candidates):
    """Top-K current candidates plus all remaining vocabulary in one OTHER bin."""
    log_probs = logits.float().log_softmax(-1)
    selected = log_probs.gather(-1, candidates).exp()
    tail = (1.0 - selected.sum(-1, keepdim=True)).clamp_min(1e-8)
    bins = torch.cat((selected, tail), dim=-1).clamp_min(1e-30)
    return bins / bins.sum(-1, keepdim=True)


def build_inputs(records, feature_table, device):
    """Whitelist current-state features; teacher outputs never enter the head."""
    batch = {key: torch.stack([r[key] for r in records]).to(device) for key in INPUT_KEYS}
    batch["candidate_features"] = feature_table[batch["candidates"]]
    batch["commit_features"] = (feature_table[batch["commit_ids"]] * batch["committed"].unsqueeze(-1))
    return batch


class TraceDataset(Dataset):
    def __init__(self, root, manifest, split):
        self.root = Path(root)
        self.files = [item for item in manifest["shards"] if item["split"] == split and item["pairs"] > 0]
        self.ends = []
        total = 0
        for item in self.files:
            total += item["pairs"]
            self.ends.append(total)
        if not total:
            raise ValueError(f"No transitions in {split}; increase prompt count/length before training")

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=64)
    def _load(self, file_index):
        return torch.load(self.root / self.files[file_index]["file"], map_location="cpu",
                          weights_only=True, mmap=True)["records"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect_right(self.ends, index)
        start = self.ends[file_index - 1] if file_index else 0
        return self._load(file_index)[index - start]
