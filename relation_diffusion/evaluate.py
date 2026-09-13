"""Original-space fixed-path NLL plus separately measured generation/decoding."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from .neural import Denoiser, ModelConfig, path_nll, sample
from .prepare import digest
from .torch_codec import TorchCodec
from .train import amp, synchronize


@torch.no_grad()
def timed_generation(model, codec, prefix, steps, dtype, repeats, warmup=2, greedy=False):
    rows, last = [], None
    for repeat in range(-warmup, repeats):
        gen = torch.Generator(device=prefix.device).manual_seed(4321 + max(0, repeat))
        synchronize(prefix.device)
        start = time.perf_counter()
        with amp(prefix.device, dtype):
            codes = sample(model, prefix, model.cfg.length, steps, gen, greedy)
        synchronize(prefix.device)
        middle = time.perf_counter()
        last = codec.decode(codes)
        synchronize(prefix.device)
        end = time.perf_counter()
        if repeat >= 0:
            rows.append(dict(forward_sampling_seconds=middle - start,
                             decode_seconds=end - middle, total_seconds=end - start))
    return dict(repeats=repeats, warmup=warmup, batch_size=len(prefix), nfe=steps,
                sampling="greedy" if greedy else "categorical",
                mean_seconds=statistics.mean(r["total_seconds"] for r in rows),
                median_seconds=statistics.median(r["total_seconds"] for r in rows),
                mean_decode_seconds=statistics.mean(r["decode_seconds"] for r in rows),
                measurements=rows), last


@torch.no_grad()
def error_spread(codec, raw):
    z = codec.encode(raw)
    changes = []
    for pos in range(codec.prefix, raw.shape[1]):
        changed = z.clone()
        changed[:, pos] = (changed[:, pos] + 1) % 257
        changes.append((codec.decode(changed) != raw).sum(1))
    counts = torch.stack(changes)
    return dict(mean_changed_original_positions=float(counts.float().mean()),
                max_changed_original_positions=int(counts.max()),
                note="Single +1 code perturbations on held-out clean sequences; diagnostic, not a probabilistic error bound")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--steps", default="1,2,4,8,16")
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.output.exists() or min(args.limit, args.batch, args.repeats) < 1:
        p.error("Choose a fresh output and positive counts")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = checkpoint["metadata"]
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest != meta["data_manifest"] or digest(args.data / "validation.npy") != manifest["validation_sha256"]:
        raise ValueError("Checkpoint and validation corpus differ")
    model = Denoiser(ModelConfig(**meta["model"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    codec = TorchCodec(meta["codec_spec"]).to(device)
    raw = np.load(args.data / "validation.npy", mmap_mode="r")[:args.limit]
    scores = []
    for steps in map(int, args.steps.split(",")):
        if meta["objective"] == "one-step" and steps != 1:
            continue
        losses = []
        for begin in range(0, len(raw), args.batch):
            tokens = torch.tensor(np.array(raw[begin:begin + args.batch]), dtype=torch.long, device=device)
            with torch.inference_mode(), amp(device, meta["dtype"]):
                values = path_nll(model, codec.encode(tokens), manifest["prefix"], steps)
            losses.extend(values.cpu().tolist())
        first = torch.tensor(np.array(raw[:1]), dtype=torch.long, device=device)
        timing, generated = timed_generation(model, codec, first[:, :manifest["prefix"]],
                                              steps, meta["dtype"], args.repeats)
        response = generated[0, manifest["prefix"]:].cpu().tolist()
        rendered = bytes(x for x in response if x < 256).decode("utf-8", errors="replace")
        row = dict(steps=steps, path_bits_per_symbol=statistics.mean(losses) /
                   ((manifest["length"] - manifest["prefix"]) * math.log(2)),
                   nll_per_example_nats=losses, timing=timing, sample_text=rendered)
        scores.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in {"nll_per_example_nats", "sample_text"}}), flush=True)
    spread = error_spread(codec, torch.tensor(np.array(raw[:16]), dtype=torch.long, device=device))
    result = dict(codec=meta["codec"], objective=meta["objective"], seed=meta["seed"],
                  trained_steps=checkpoint["step"], seen_symbols=checkpoint["step"] * meta["global_batch"] * manifest["length"],
                  model=meta["model"], parameter_count=meta["parameter_count"],
                  global_batch=meta["global_batch"], dtype=meta["dtype"], lr=meta["lr"],
                  planned_steps=meta["planned_steps"], data_sha256=manifest["validation_sha256"],
                  train_sha256=manifest["train_sha256"], examples=len(raw),
                  scores=scores, error_spread=spread,
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  train_seconds=checkpoint["train_seconds"],
                  metric="Exact NLL of a specified fixed reveal-path model, mapped to original symbols by a bijection. Not conventional AR perplexity, task accuracy, or the marginalized diffusion likelihood.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
