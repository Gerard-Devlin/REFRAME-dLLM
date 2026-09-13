"""Prepare train/validation byte blocks and frozen train-only codecs.

The test split is deliberately never loaded. No GSM8K/HumanEval training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .torch_codec import TorchCodec, codec_spec


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pack(texts, length, max_blocks):
    # Keep the official split; pack UTF-8 text with an explicit line boundary.
    values = []
    for text in texts:
        if text.strip():
            values.extend(text.encode("utf-8"))
            values.append(256)
        if max_blocks and len(values) >= max_blocks * length:
            break
    count = len(values) // length
    if max_blocks:
        count = min(count, max_blocks)
    if count < 1:
        raise ValueError("Not enough text to make one complete block")
    return np.asarray(values[:count * length], dtype=np.uint16).reshape(count, length)


def independent_diagnostic(spec, train, valid):
    """Train-only categorical counts, heldout code NLL mapped to original space.

    Ignores visible prefix content. This cheap statistical model is neither a
    neural denoiser nor an estimate/bound of its future performance.
    """
    codec = TorchCodec(spec)
    ztrain = codec.encode(torch.tensor(train, dtype=torch.long)).numpy()
    zvalid = codec.encode(torch.tensor(valid, dtype=torch.long)).numpy()
    total = 0.0
    for pos in range(spec["prefix"], spec["length"]):
        counts = np.bincount(ztrain[:, pos], minlength=257).astype(np.float64) + 1.0
        probability = counts / counts.sum()
        total += -np.log2(probability[zvalid[:, pos]]).sum()
    return dict(bits_per_symbol=total / (len(valid) * (spec["length"] - spec["prefix"])),
                fitting_examples=len(train), heldout_examples=len(valid), smoothing=1.0,
                scope="Independent per-position categorical model, prefix content ignored. Cheap hint only; not a bound on neural denoising quality.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--train-text", type=Path)
    p.add_argument("--validation-text", type=Path)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--prefix", type=int, default=32)
    p.add_argument("--max-train-blocks", type=int, default=0)
    p.add_argument("--max-validation-blocks", type=int, default=0)
    p.add_argument("--fit-blocks", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if not 0 <= args.prefix < args.length or args.length - args.prefix < 2:
        p.error("Need at least two response positions")
    if min(args.max_train_blocks, args.max_validation_blocks) < 0 or args.fit_blocks < 1:
        p.error("Invalid block limit")
    if bool(args.train_text) != bool(args.validation_text):
        p.error("Supply both local split files or neither")
    if args.output.exists():
        p.error("Output exists; choose a new data directory")
    started = time.perf_counter()
    fingerprints = {}
    if args.train_text:
        if args.train_text.resolve() == args.validation_text.resolve():
            p.error("Train and validation files must differ")
        fingerprints = {split: digest(path) for split, path in
                        (("train", args.train_text), ("validation", args.validation_text))}
        if fingerprints["train"] == fingerprints["validation"]:
            p.error("Train and validation contents are identical")
        texts = {s: path.read_text(encoding="utf-8").splitlines() for s, path in
                 (("train", args.train_text), ("validation", args.validation_text))}
    else:
        from datasets import load_dataset
        texts = {}
        for split in ("train", "validation"):
            ds = load_dataset(args.dataset, args.dataset_config, split=split)
            fingerprints[split] = ds._fingerprint
            texts[split] = ds["text"]
    train = pack(texts["train"], args.length, args.max_train_blocks)
    valid = pack(texts["validation"], args.length, args.max_validation_blocks)
    # Remove exact validation block duplicates from TRAIN, never fit on heldout.
    validation_rows = {row.tobytes() for row in valid}
    keep = np.array([r.tobytes() not in validation_rows for r in train])
    removed = int((~keep).sum())
    train = train[keep]
    if len(train) == 0:
        raise ValueError("No training blocks remain after exact overlap removal")
    rng = np.random.default_rng(args.seed)
    fit_ids = rng.choice(len(train), min(args.fit_blocks, len(train)), replace=False)
    fit = train[fit_ids].astype(np.int64)
    args.output.mkdir(parents=True)
    np.save(args.output / "train.npy", train)
    np.save(args.output / "validation.npy", valid)
    np.save(args.output / "fit_indices.npy", fit_ids)
    codecs, diagnostics = {}, {}
    for name, kind, depth in (("identity", "identity", 1), ("rename", "rename", 1),
                              ("random", "random", 2), ("relation1", "relation", 1),
                              ("relation2", "relation", 2)):
        spec = codec_spec(kind, fit, args.prefix, depth, args.seed)
        codec = TorchCodec(spec)
        for source in (fit[:128], valid[:128].astype(np.int64)):
            original = torch.from_numpy(source)
            assert torch.equal(codec.decode(codec.encode(original)), original)
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        codecs[name] = digest(path)
        diagnostics[name] = independent_diagnostic(spec, fit, valid[:1024].astype(np.int64))
    # Pure renaming must leave this likelihood unchanged: a useful control.
    assert abs(diagnostics["identity"]["bits_per_symbol"] - diagnostics["rename"]["bits_per_symbol"]) < 1e-10
    (args.output / "independent_diagnostic.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    manifest = dict(format_version=1, tokenizer="utf8-bytes+boundary256", vocab_size=257,
                    length=args.length, prefix=args.prefix, train_blocks=len(train),
                    validation_blocks=len(valid), removed_exact_train_overlaps=removed,
                    fit_blocks=len(fit), fit_seed=args.seed, fingerprints=fingerprints,
                    source="local" if args.train_text else f"{args.dataset}/{args.dataset_config}",
                    train_sha256=digest(args.output / "train.npy"),
                    validation_sha256=digest(args.output / "validation.npy"), codecs=codecs,
                    prepare_seconds=time.perf_counter() - started,
                    note="Frozen train-only codes; test split not loaded; byte pilot, not BPE or benchmark accuracy")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("No-neural-training diagnostic:", json.dumps(diagnostics))


if __name__ == "__main__":
    main()
