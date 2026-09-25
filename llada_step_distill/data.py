from __future__ import annotations

import json
import hashlib
import math
import os
import random
import shutil
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import torch

from .core import (
    BLOCK_LENGTH,
    DATA_FILES,
    DOMAIN_COUNTS,
    EOS_ID,
    GEN_LENGTH,
    MASK_ID,
    MAX_PROMPT_LENGTH,
    SEED,
    TRAIN_QUOTAS,
    Decontaminator,
    atomic_json,
    digest,
    read_json,
    stable_u64,
    user_text,
    validation_quotas,
)

FORMAT_VERSION = 1
DEFAULT_SHARD_SIZE = 2048


def resolve_nemotron_root(path: Path) -> Path:
    path = Path(path).resolve()
    if (path / "SFT").is_dir():
        return path
    snapshots = sorted(path.glob("snapshots/*"))
    matches = [p for p in snapshots if (p / "SFT").is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve one Nemotron snapshot below {path}")
    return matches[0]


def proportional_quotas(total: int, counts: dict[str, int] = DOMAIN_COUNTS) -> dict[str, int]:
    if total == 10_000_000 and counts == DOMAIN_COUNTS:
        return dict(TRAIN_QUOTAS)
    denominator = sum(counts.values())
    raw = {key: total * value / denominator for key, value in counts.items()}
    result = {key: int(value) for key, value in raw.items()}
    missing = total - sum(result.values())
    for key in sorted(raw, key=lambda k: raw[k] - result[k], reverse=True)[:missing]:
        result[key] += 1
    return result


def load_gsm8k_questions(evaluation_json: Path | None = None) -> list[str]:
    questions: list[str] = []
    if evaluation_json:
        rows = read_json(evaluation_json)
        questions.extend(str(row["question"]) for row in rows)
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main")
        for split in ("train", "test"):
            questions.extend(str(row["question"]) for row in dataset[split])
    except Exception:
        if not questions:
            raise RuntimeError("GSM8K train/test must already be cached or supplied as JSON")
    return list(dict.fromkeys(questions))


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_size: int = DEFAULT_SHARD_SIZE):
        self.root = Path(root) / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.rows: list[dict] = []
        self.shards: list[dict] = []
        self.count = 0

    def append(self, row: dict) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        index = len(self.shards)
        path = self.root / f"shard_{index:06d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"format_version": FORMAT_VERSION, "records": self.rows}, tmp)
        os.replace(tmp, path)
        self.shards.append({"file": str(path.relative_to(self.root.parent)), "records": len(self.rows)})
        self.rows = []

    def close(self) -> list[dict]:
        self.flush()
        return self.shards


def _eligible_row(row: dict, tokenizer, decontaminator: Decontaminator) -> dict | None:
    messages = row.get("input")
    response = row.get("output")
    if not isinstance(messages, list) or not isinstance(response, str) or not response.strip():
        return None
    question = user_text(messages)
    if not question or decontaminator.contaminated(question):
        return None
    try:
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        return None
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > MAX_PROMPT_LENGTH or len(response_ids) < BLOCK_LENGTH:
        return None
    response_ids = list(response_ids[: GEN_LENGTH - 1])
    while response_ids and response_ids[-1] == int(tokenizer.eos_token_id or EOS_ID):
        response_ids.pop()
    response_ids.append(int(tokenizer.eos_token_id or EOS_ID))
    identity = digest({"prompt": prompt_ids, "response": response_ids})
    return {
        "sample_id": identity,
        "prompt_ids": [int(v) for v in prompt_ids],
        "response_ids": [int(v) for v in response_ids],
    }


def prepare(
    nemotron_root: Path,
    output: Path,
    tokenizer,
    evaluation_json: Path | None = None,
    train_size: int = 10_000_000,
    validation_size: int = 20_000,
    shard_size: int = DEFAULT_SHARD_SIZE,
    oversample: float = 1.05,
) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepare output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    root = resolve_nemotron_root(nemotron_root)
    decontaminator = Decontaminator(load_gsm8k_questions(evaluation_json))
    train_quota = proportional_quotas(train_size)
    validation_quota = proportional_quotas(validation_size)
    writers = {
        "train": ShardWriter(output, "train", shard_size),
        "validation": ShardWriter(output, "validation", shard_size),
    }
    database = sqlite3.connect(output / "seen.sqlite")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute("CREATE TABLE IF NOT EXISTS seen (sample_id TEXT PRIMARY KEY)")
    counters = {split: {key: 0 for key in DATA_FILES} for split in writers}
    selection_hashers = {split: hashlib.sha256() for split in writers}
    train_index = acceleration_index = 0
    for domain, relatives in DATA_FILES.items():
        needed = train_quota[domain] + validation_quota[domain]
        threshold = min(1.0, needed * oversample / DOMAIN_COUNTS[domain])
        accepted = 0
        for relative in relatives:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream):
                    if accepted >= needed:
                        break
                    if stable_u64(SEED, domain, relative, line_number) / 2**64 > threshold:
                        continue
                    row = json.loads(line)
                    item = _eligible_row(row, tokenizer, decontaminator)
                    if item is None:
                        continue
                    cursor = database.execute("INSERT OR IGNORE INTO seen(sample_id) VALUES (?)", (item["sample_id"],))
                    if cursor.rowcount != 1:
                        continue
                    if counters["train"][domain] < train_quota[domain]:
                        split = "train"
                        # Exactly one retention row per ten training rows.
                        kind = "retention" if train_index % 10 == 0 else "acceleration"
                        item["kind"] = kind
                        if kind == "acceleration":
                            item["real_transition"] = acceleration_index % 45 == 0
                            acceleration_index += 1
                        else:
                            item["real_transition"] = False
                        train_index += 1
                    else:
                        split = "validation"
                        item["kind"] = "acceleration"
                        item["real_transition"] = True
                    item["domain"] = domain
                    selection_hashers[split].update(item["sample_id"].encode("ascii"))
                    selection_hashers[split].update(b"\n")
                    writers[split].append(item)
                    counters[split][domain] += 1
                    accepted += 1
                    if (train_index + writers["validation"].count) % 10_000 == 0:
                        database.commit()
                        print(f"prepare selected={train_index + writers['validation'].count:,}", flush=True)
            if accepted >= needed:
                break
        if accepted != needed:
            raise RuntimeError(
                f"{domain}: selected {accepted:,}/{needed:,}; increase --oversample and use a new output"
            )
    database.commit()
    database.close()
    shards = {split: writer.close() for split, writer in writers.items()}
    retention = sum(1 for i in range(train_size) if i % 10 == 0)
    acceleration = train_size - retention
    real = sum(1 for i in range(acceleration) if i % 45 == 0)
    manifest = {
        "format_version": FORMAT_VERSION,
        "source": str(root),
        "model_revision": getattr(tokenizer, "name_or_path", "pinned-snapshot"),
        "train_size": train_size,
        "validation_size": validation_size,
        "train_quotas": train_quota,
        "validation_quotas": validation_quota,
        "actual": counters,
        "selection_sha256": {split: value.hexdigest() for split, value in selection_hashers.items()},
        "retention_records": retention,
        "acceleration_records": acceleration,
        "real_transition_records": real,
        "shards": shards,
        "gsm8k_decontamination": {"exact": True, "word_5gram_jaccard": 0.8},
        "seed": SEED,
    }
    manifest["data_hash"] = digest({k: v for k, v in manifest.items() if k != "source"})
    atomic_json(output / "manifest.json", manifest)
    # Keep the unique index only while selecting. It is not an input to later stages.
    for suffix in ("", "-wal", "-shm"):
        (output / f"seen.sqlite{suffix}").unlink(missing_ok=True)
    return manifest


def load_shard(path: Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported shard format: {path}")
    return payload["records"]


class AlignedShardDataset:
    """Locality-preserving, no-replacement iteration over prepared/supervision shards."""

    def __init__(self, prepared: Path, supervision: Path, split: str, seed: int = SEED):
        self.prepared = Path(prepared)
        self.supervision = Path(supervision)
        self.split = split
        self.seed = seed
        p_manifest = read_json(self.prepared / "manifest.json")
        s_manifest = read_json(self.supervision / "manifest.json")
        if s_manifest["prepared_hash"] != p_manifest["data_hash"]:
            raise ValueError("Supervision was collected from different prepared data")
        self.p_shards = p_manifest["shards"][split]
        self.s_shards = s_manifest["shards"][split]
        if [x["records"] for x in self.p_shards] != [x["records"] for x in self.s_shards]:
            raise ValueError("Prepared/supervision shard boundaries differ")
        self.count = sum(x["records"] for x in self.p_shards)

    def iter_epoch(self, epoch: int = 0) -> Iterator[tuple[dict, dict]]:
        order = list(range(len(self.p_shards)))
        random.Random(self.seed + epoch).shuffle(order)
        for shard_index in order:
            prepared = load_shard(self.prepared / self.p_shards[shard_index]["file"])
            supervision = load_shard(self.supervision / self.s_shards[shard_index]["file"])
            if len(prepared) != len(supervision):
                raise ValueError("Aligned shard record count changed")
            rows = list(range(len(prepared)))
            random.Random(self.seed ^ (epoch << 20) ^ shard_index).shuffle(rows)
            for row in rows:
                if prepared[row]["sample_id"] != supervision[row]["sample_id"]:
                    raise ValueError("Aligned sample id mismatch")
                yield prepared[row], supervision[row]


def padded_response(response_ids: list[int]) -> list[int]:
    values = list(response_ids[:GEN_LENGTH])
    return values + [EOS_ID] * (GEN_LENGTH - len(values))


def choose_block(response_ids: list[int], seed: int) -> int:
    blocks = max(1, min(GEN_LENGTH, len(response_ids)) // BLOCK_LENGTH)
    return random.Random(seed).randrange(blocks)


def build_acceleration_state(row: dict, block: int, reveal_mask: int) -> tuple[list[int], list[int], int]:
    reference = padded_response(row["response_ids"])
    start = block * BLOCK_LENGTH
    end = start + BLOCK_LENGTH
    if not 0 <= start < GEN_LENGTH:
        raise ValueError("block outside generation canvas")
    canvas = [MASK_ID] * GEN_LENGTH
    canvas[:start] = reference[:start]
    for offset in range(BLOCK_LENGTH):
        if reveal_mask & (1 << offset):
            canvas[start + offset] = reference[start + offset]
    return row["prompt_ids"] + canvas, reference, len(row["prompt_ids"]) + start


def deterministic_state(row: dict, attempt: int = 0) -> tuple[int, int]:
    seed = stable_u64("state", row["sample_id"], attempt)
    block = choose_block(row["response_ids"], seed)
    rng = random.Random(seed)
    # Cover all noise levels while always leaving at least two candidates masked.
    reveal_count = rng.randrange(0, BLOCK_LENGTH - 1)
    positions = list(range(BLOCK_LENGTH))
    rng.shuffle(positions)
    mask = 0
    for position in positions[:reveal_count]:
        mask |= 1 << position
    return block, mask


def build_retention_state(row: dict, seed: int) -> tuple[list[int], list[int], list[bool], float]:
    reference = padded_response(row["response_ids"])
    rng = random.Random(seed)
    p_mask = (1.0 - 1e-3) * rng.random() + 1e-3
    response_length = min(len(row["response_ids"]), GEN_LENGTH)
    masked = [rng.random() < p_mask if index < response_length else False for index, _ in enumerate(reference)]
    if not any(masked):
        masked[rng.randrange(len(masked))] = True
    noisy = [MASK_ID if use_mask else token for token, use_mask in zip(reference, masked)]
    return row["prompt_ids"] + noisy, reference, masked, p_mask
