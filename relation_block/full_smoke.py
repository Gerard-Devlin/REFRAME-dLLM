"""Bounded actual-topology optimizer/save/restore smoke before a full epoch."""
import argparse
import json
import os
import math
import statistics
import shutil
from pathlib import Path
import subprocess
import sys
from .common import digest, write_json
from .preflight import implementation_hashes, require_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--global-batch", type=int, default=12)
    p.add_argument("--micro-batch", type=int, default=1)
    args = p.parse_args()
    require_gate(args.data)
    write_json(args.data / "full_smoke.json", {"pass": False, "status": "running"})
    if args.world_size < 1:
        raise ValueError("GPU count must be positive")
    import torch
    if torch.cuda.device_count() != args.world_size:
        raise ValueError("Smoke must use all explicitly selected GPUs")
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={args.world_size}", "-m", "relation_block.train"]
    results = {}
    for arm in ("token", "relation"):
        output = args.output / arm / "train"
        train = command + ["--data", str(args.data), "--output", str(output), "--arm", arm,
                           "--steps", "8", "--save-every", "4", "--eval-every", "0", "--smoke",
                           "--global-batch", str(args.global_batch), "--micro-batch", str(args.micro_batch)]
        subprocess.run(train + ["--stop-after", "4"], check=True)
        before = json.loads((output / "status.json").read_text())
        if before["completed_steps"] != 4 or before["status"] != "paused":
            raise RuntimeError("Smoke did not save the deliberate pause")
        subprocess.run(train + ["--resume", str(output)], check=True)
        after = json.loads((output / "status.json").read_text())
        if after["completed_steps"] != 8 or after["status"] != "complete":
            raise RuntimeError("Smoke resume did not complete")
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        if [x["step"] for x in records] != list(range(1, 9)):
            raise RuntimeError("Resume repeated or skipped training steps")
        results[arm] = after
        # Discard the first step after each process launch; average over varied examples.
        timings = [r["step_seconds"] for r in records if r["step"] not in (1, 5)]
        after["smoke_median_step_seconds"] = statistics.median(timings)
        manifest = json.loads((args.data / "manifest.json").read_text())
        after["estimated_epoch_compute_seconds"] = math.ceil(manifest["train_examples"] / args.global_batch) * statistics.median(timings)
        # Exercise exported full weights after actual optimizer updates.
        eval_env = dict(os.environ)
        eval_env["CUDA_VISIBLE_DEVICES"] = eval_env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        subprocess.run([sys.executable, "-u", "-m", "relation_block.evaluate", "--data", str(args.data),
                        "--checkpoint", str(output), "--output", str(args.output / arm / "eval"),
                        "--limit", "2", "--rounds", "2", "--max-new-tokens", "64"], env=eval_env, check=True)
    gate = dict(pass_=True, implementation=implementation_hashes(),
                data_hash=digest(args.data / "manifest.json"), world_size=args.world_size,
                global_batch=args.global_batch, micro_batch=args.micro_batch, results=results,
                note="Actual full-parameter optimizer, save, resume and exported-weight evaluation. No quality claim.")
    gate["pass"] = gate.pop("pass_")
    write_json(args.data / "full_smoke.json", gate)
    write_json(args.output / "full_smoke.json", gate)
    # Smoke checkpoints are temporary validation artifacts, not research weights.
    # Keep metrics, status, gate and generation outputs; free tens of GB only
    # after BOTH arms have successfully resumed and exported/evaluated weights.
    for arm in ("token", "relation"):
        root = (args.output / arm / "train").resolve()
        for checkpoint in root.glob("step_[0-9]*"):
            if checkpoint.resolve().parent != root:
                raise ValueError("Smoke cleanup path escapes its training directory")
            shutil.rmtree(checkpoint)
        (root / "latest.json").unlink()
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
