"""Read-only checkpoint audit: context use, precision and matched frequency control.

No optimizer, parameter updates, new data downloads, or generation speed claims.
Shuffled-context losses are diagnostic interventions, not model likelihoods.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .neural import Denoiser, ModelConfig, reveal_groups, update_batch
from .prepare import digest, independent_diagnostic
from .torch_codec import TorchCodec
from .train import amp


@torch.no_grad()
def intervened_path_nll(model, clean, donor, prefix, steps, intervention):
    if intervention not in {"normal", "shuffle_prefix", "shuffle_history"}:
        raise ValueError(intervention)
    x = torch.full_like(clean, model.cfg.vocab_size)
    x[:, :prefix] = donor[:, :prefix] if intervention == "shuffle_prefix" else clean[:, :prefix]
    losses = torch.zeros(len(x), device=x.device)
    for start, end in reveal_groups(x.shape[1], prefix, steps):
        logits = model(x)[:, start:end].float()
        losses += F.cross_entropy(logits.transpose(1, 2), clean[:, start:end], reduction="none").sum(1)
        # Always score the original target, then reveal either its true history
        # or another example's history. Current/future targets remain hidden.
        x[:, start:end] = donor[:, start:end] if intervention == "shuffle_history" else clean[:, start:end]
    return losses


def audit_checkpoint(checkpoint, data_dir, limit, batch, device, steps):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = saved["metadata"]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest != meta["data_manifest"]:
        raise ValueError("Checkpoint and data manifest differ")
    for split in ("train", "validation"):
        if digest(data_dir / f"{split}.npy") != manifest[f"{split}_sha256"]:
            raise ValueError(f"Changed {split} data")
    raw = np.array(np.load(data_dir / "validation.npy", mmap_mode="r")[:limit], dtype=np.int64)
    if len(raw) < 2:
        raise ValueError("Need at least two validation examples")
    train = np.load(data_dir / "train.npy", mmap_mode="r")
    fit_ids = np.load(data_dir / "fit_indices.npy")
    frequency = independent_diagnostic(meta["codec_spec"], np.array(train[fit_ids], dtype=np.int64), raw)
    model = Denoiser(ModelConfig(**meta["model"])).to(device).eval()
    model.load_state_dict(saved["model"])
    trained_steps = saved["step"]
    del saved
    codec = TorchCodec(meta["codec_spec"]).to(device)
    codes = codec.encode(torch.tensor(raw, device=device))
    # A fixed global permutation, independent of batch size or precision.
    # Pair distant blocks to reduce same-document adjacency; this is not a
    # guarantee of independent documents, so do not report significance here.
    donor_ids = torch.arange(len(codes), device=device).roll(len(codes) // 2)
    donors = codes[donor_ids]
    _, masks, _ = update_batch(len(raw), len(raw), manifest["length"], manifest["prefix"], 9281, 0)
    masks = masks.to(device)
    denominator = len(raw) * (manifest["length"] - manifest["prefix"]) * math.log(2)
    results = {}
    for dtype in dict.fromkeys([meta["dtype"], "float32"]):
        paths = {}
        for count in steps:
            row = {}
            for condition in ("normal", "shuffle_prefix", "shuffle_history"):
                values = []
                for start in range(0, len(codes), batch):
                    with amp(device, dtype):
                        losses = intervened_path_nll(model, codes[start:start + batch], donors[start:start + batch],
                                                     manifest["prefix"], count, condition)
                    values.extend(losses.cpu().tolist())
                row[condition] = dict(bits_per_symbol=sum(values) / denominator,
                                      per_example_nll_nats=values)
            for condition in ("shuffle_prefix", "shuffle_history"):
                row[condition]["delta_bits_vs_normal"] = row[condition]["bits_per_symbol"] - row["normal"]["bits_per_symbol"]
            paths[str(count)] = row
            print(f"{meta['codec']} {dtype} {count} steps: normal={row['normal']['bits_per_symbol']:.6f}, "
                  f"shuffle_prefix_delta={row['shuffle_prefix']['delta_bits_vs_normal']:.6f}, "
                  f"shuffle_history_delta={row['shuffle_history']['delta_bits_vs_normal']:.6f}", flush=True)
        random_mask = {}
        for condition in ("normal", "shuffle_visible"):
            total = 0.0
            for start in range(0, len(codes), batch):
                target, mask = codes[start:start + batch], masks[start:start + batch]
                visible = target if condition == "normal" else donors[start:start + batch]
                noisy = visible.masked_fill(mask, model.cfg.vocab_size)
                with torch.no_grad(), amp(device, dtype):
                    ce = F.cross_entropy(model(noisy).float().transpose(1, 2), target, reduction="none")
                    total += float((ce * mask).sum())
            random_mask[condition] = total / (int(masks.sum()) * math.log(2))
        random_mask["delta_bits_vs_normal"] = random_mask["shuffle_visible"] - random_mask["normal"]
        random_mask["note"] = "Same Bernoulli mask pattern; score masked targets only, without 1/p weighting. Within-model context intervention, not original-space sequence NLL."
        results[dtype] = dict(paths=paths, random_mask=random_mask)
        print(f"{meta['codec']} {dtype} random-mask context delta: {random_mask['delta_bits_vs_normal']:.6f}", flush=True)
    return dict(codec=meta["codec"], objective=meta["objective"], trained_steps=trained_steps,
                signature={k: meta[k] for k in ("model", "seed", "global_batch", "planned_steps", "lr", "data_manifest")},
                examples=len(raw), frequency_control=frequency, precision_results=results,
                note="Positive shuffle delta suggests useful context on this sample; zero/negative is not a proof of no dependence. No universal threshold or significance claim.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", default="1,16")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()
    if args.output.exists() or args.limit < 2 or args.batch < 1:
        p.error("Choose a fresh output, limit >= 2 and positive batch")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    results = []
    for name in ("identity", "relation2"):
        checkpoint = args.source_run / f"{name}-diffusion-seed{args.seed}" / "checkpoint.pt"
        result = audit_checkpoint(checkpoint, args.data, args.limit, args.batch, device,
                                  list(map(int, args.steps.split(","))))
        if results and (result["signature"] != results[0]["signature"] or result["trained_steps"] != results[0]["trained_steps"]):
            raise ValueError("Unmatched checkpoint budgets or data")
        results.append(result)
        print(f"{name} matched-validation frequency control: {result['frequency_control']['bits_per_symbol']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope=__doc__, results=results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
