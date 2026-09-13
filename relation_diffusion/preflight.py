"""No-training cost gate. Passing says nothing about learned quality."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate import error_spread, timed_generation
from .neural import Denoiser, ModelConfig
from .torch_codec import TorchCodec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.output.exists() or args.repeats < 1:
        p.error("Choose a fresh output and positive repeats")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(1234)
    model = Denoiser(ModelConfig(length=manifest["length"], width=args.width,
                                 layers=args.layers, heads=args.heads)).to(device).eval()
    raw = torch.tensor(np.array(np.load(args.data / "validation.npy")[:16]), dtype=torch.long, device=device)
    records = {}
    for name in ("identity", "rename", "random", "relation1", "relation2"):
        codec = TorchCodec(json.loads((args.data / f"{name}.json").read_text(encoding="utf-8"))).to(device)
        assert torch.equal(codec.decode(codec.encode(raw)), raw)
        records[name] = dict(error_spread=error_spread(codec, raw), timings={})
        for steps in (1, 2, 4, 8, 16):
            timing, _ = timed_generation(model, codec, raw[:1, :manifest["prefix"]],
                                          steps, args.dtype, args.repeats)
            records[name]["timings"][str(steps)] = timing
            print(f"{name} steps={steps}: {timing['mean_seconds']:.6f}s", flush=True)
    base = records["identity"]["timings"]["16"]["mean_seconds"]
    candidate = records["relation2"]["timings"]["4"]["mean_seconds"]
    output = dict(scope="Untrained random model: COST ONLY, no quality or achievable acceleration evidence",
                  ratio_identity16_to_relation4=base / candidate,
                  cost_gate_pass=base / candidate >= 1.5,
                  next_step="Manual decision: at most a bounded matched training pilot; never auto-launch training",
                  model=vars(model.cfg), device=str(device), dtype=args.dtype,
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
