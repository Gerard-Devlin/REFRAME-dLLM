"""Guarded server launcher for the read-only Relay row-work probe."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPO = Path("/home/xuyouwen/REFRAME-dLLM")
ROOT = Path((REPO / "focus_dllm/dllm-eval/runs/latest_tuning.txt").read_text().strip())
PYTHON = "/home/xuyouwen/.conda/envs/fastdllm311/bin/python"
UUID = "GPU-43d49500-e070-14e1-48b2-6a468fa01f8b"
DATA = [Path("/home/xuyouwen/hf_home_local/benchmarks") / name for name in
        ("humaneval_test.json", "mbpp_paper_3shot.json", "math_paper_4shot_v2.json")]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    files = [*sorted((REPO / "focus_v5").glob("*.py")),
             *sorted((REPO / "focus_dllm/tuning").glob("*.py"))]
    return {str(path.relative_to(REPO)): digest(path) for path in files}


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def guard() -> dict:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    parsed = [list(map(str.strip, row.split(","))) for row in rows]
    gpu = next(row for row in parsed if row[0] == "1")
    if gpu[1] != UUID:
        raise AssertionError("physical GPU1 UUID changed")
    free_mib = int(gpu[3])
    if free_mib < 22_000:
        raise AssertionError(f"GPU1 has only {free_mib} MiB free; did not stop or reset another process")
    running = subprocess.check_output(["pgrep", "-af", "focus_v5.relay_ceiling_probe"], text=True
                                      ).splitlines() if subprocess.run(
        ["pgrep", "-f", "focus_v5.relay_ceiling_probe"], stdout=subprocess.DEVNULL).returncode == 0 else []
    running = [line for line in running if "launch_relay_ceiling.py" not in line]
    if running:
        raise AssertionError("Relay ceiling probe already running")
    if (REPO / "focus_dllm/README.md").stat().st_size != 0:
        raise AssertionError("focus_dllm README must remain empty")
    return {"time": time.time(), "gpu": 1, "uuid": UUID,
            "memory_used_mib": int(gpu[2]), "memory_free_mib": free_mib,
            "utilization_percent": int(gpu[4]), "shared_existing_allocation_allowed_by_user": True}


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = ROOT / f"focus_v5_relay_ceiling_{stamp}"
    queue = ROOT / "focus_v5_relay_queue.json"
    with (ROOT / "gpu1_research.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = guard()
        output.mkdir(exist_ok=False)
        source = source_hashes()
        manifest = {
            "implementation": source,
            "datasets": {str(path): digest(path) for path in DATA},
            "guard": before,
            "third_party": str(ROOT / "third_party/pinned_20261001"),
            "scope": "Read-only pinned Flash trajectory; reused six development prompts; row-work ceiling only.",
        }
        write(output / "launch_manifest.json", manifest)
        environment = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(key, None)
        environment.update(
            CUDA_VISIBLE_DEVICES=UUID,
            FOCUS_RESEARCH_GPU_UUID=UUID,
            HF_HOME="/home/xuyouwen/hf_home_local",
            HF_HUB_CACHE="/home/xuyouwen/hf_hub_local",
            HF_DATASETS_CACHE="/home/xuyouwen/hf_home_local/datasets",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            HF_EVALUATE_OFFLINE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUTF8="1",
            PYTHONPATH=f"{REPO}/focus_dllm/dllm-eval:{REPO}",
        )
        command = [PYTHON, "-u", "-m", "focus_v5.relay_ceiling_probe",
                   "--third-party", str(ROOT / "third_party/pinned_20261001"),
                   "--datasets", *map(str, DATA), "--output", str(output / "probe")]
        with (output / "probe.log").open("x") as log:
            process = subprocess.Popen(command, cwd=REPO, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            write(queue, {"status": "running", "pid": process.pid, "gpu": 1,
                          "uuid": UUID, "output": str(output), "source": source})
            code = process.wait()
        (output / "exit_code").write_text(f"{code}\n")
        if source != source_hashes():
            raise AssertionError("research source changed while probe was running")
        if code or not (output / "probe/complete").exists():
            write(queue, {"status": "failed", "pid": process.pid, "exit_code": code,
                          "gpu": 1, "output": str(output)})
            raise RuntimeError("Relay ceiling probe failed; artifacts preserved")
        after = guard()
        report = json.loads((output / "probe/diagnostic.json").read_text())
        if len(report["prompts"]) != 6:
            raise AssertionError("incomplete prompt set")
        write(queue, {"status": "complete", "gpu": 1, "output": str(output),
                      "summary": report["summary"], "guard_after": after})
        (output / "complete").write_text("OK\n")


if __name__ == "__main__":
    main()
