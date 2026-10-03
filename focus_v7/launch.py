"""Guarded single-GPU mechanism probe; no automatic generation expansion."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path("/home/xuyouwen/REFRAME-dLLM")
ROOT = Path((REPO / "focus_dllm/dllm-eval/runs/latest_tuning.txt").read_text().strip())
PYTHON = "/home/xuyouwen/.conda/envs/fastdllm311/bin/python"
UUID = "GPU-43d49500-e070-14e1-48b2-6a468fa01f8b"
DATA = [Path("/home/xuyouwen/hf_home_local/benchmarks") / name for name in
        ("humaneval_test.json", "mbpp_paper_3shot.json", "math_paper_4shot_v2.json")]
REFERENCE = ROOT / "focus_v5_relay_ceiling_20261003_123323/probe/diagnostic.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    dirs = ("focus_v5", "focus_v6", "focus_v7", "focus_dllm/tuning")
    return {str(p.relative_to(REPO)): digest(p) for d in dirs for p in sorted((REPO/d).rglob("*.py"))}


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2)); tmp.replace(path)


def guard():
    previous = json.loads((ROOT / "focus_v6_atomic_audit_queue.json").read_text())
    output = Path(previous["output"])
    assert previous["status"] == "complete" and (output/"exit_code").read_text().strip() == "0"
    assert (output/"probe/complete").exists()
    main = Path((REPO / "focus_dllm/dllm-eval/runs/latest_run.txt").read_text().strip())
    state = json.loads((main / "elastic/state.json").read_text())
    assert state["max_total_gpus"] == 5 and state["used_gpus"] == 0 and not state["workers"]
    assert (main / "exit_code").read_text().strip() == "0"
    protected = json.loads((main/"elastic/manifest.json").read_text())["implementation"]
    assert all(digest(REPO/"focus_dllm"/name) == h for name,h in protected.items())
    assert (REPO/"focus_dllm/README.md").stat().st_size == 0
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    gpu = next([s.strip() for s in r.split(",")] for r in rows if r.split(",")[0].strip() == "1")
    assert gpu[1] == UUID and int(gpu[3]) >= 22000
    # The user explicitly permits sharing the ~3.7GB foreign allocation. Only
    # our lock controls our own concurrency; nobody's process is killed/reset.
    return {"gpu": 1, "uuid": UUID, "free_mib": int(gpu[3]), "used_mib": int(gpu[2]),
            "utilization": int(gpu[4]), "main_max": 5, "main_used": 0,
            "protected": protected, "shared_gpu_authorized": True}


def main():
    with (ROOT/"gpu1_research.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = guard()
        frozen = sources()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = ROOT/f"focus_v7_prefix_cache_{stamp}"
        output.mkdir(exist_ok=False)
        snapshot = ROOT/"source_snapshots"/f"before_focus_v7_prefix_cache_{stamp}"
        snapshot.mkdir(exist_ok=False)
        for name in frozen:
            destination = snapshot/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, destination)
        write(output/"launch_manifest.json", {"sources": frozen, "guard": before, "snapshot": str(snapshot),
             "datasets": {str(p): digest(p) for p in DATA}, "reference": digest(REFERENCE),
             "scope": "six reused development prompts, bounded read-only cache-producing prefix packet gate"})
        env = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                    "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=UUID, FOCUS_RESEARCH_GPU_UUID=UUID,
                   HF_HOME="/home/xuyouwen/hf_home_local", HF_HUB_CACHE="/home/xuyouwen/hf_hub_local",
                   HF_DATASETS_CACHE="/home/xuyouwen/hf_home_local/datasets",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_EVALUATE_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1",
                   PYTHONPATH=f"{REPO}/focus_dllm/dllm-eval:{REPO}")
        with (output/"cpu_tests.log").open("x") as log:
            tests = subprocess.run([PYTHON, "-m", "unittest", "discover", "-s", "focus_v7/tests", "-v"],
                                   cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=""), stdout=log, stderr=subprocess.STDOUT)
        assert tests.returncode == 0, "CPU tests failed; no GPU task started"
        guard(); assert sources() == frozen
        queue = ROOT/"focus_v7_prefix_cache_queue.json"
        command = [PYTHON, "-u", "-m", "focus_v7.probe", "--third-party", str(ROOT/"third_party/pinned_20261001"),
                   "--datasets", *map(str, DATA), "--reference", str(REFERENCE), "--output", str(output/"probe")]
        with (output/"probe.log").open("x") as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(queue, {"status": "running", "pid": child.pid, "gpu": 1, "uuid": UUID,
                          "output": str(output), "sources": frozen})
            code = child.wait()
        (output/"exit_code").write_text(f"{code}\n")
        assert sources() == frozen, "research source edited while GPU probe was live"
        complete = code == 0 and (output/"probe/complete").exists()
        report = json.loads((output/"probe/diagnostic.json").read_text()) if complete else {}
        write(queue, {"status": "complete" if complete else "failed", "pid": child.pid,
                      "exit_code": code, "output": str(output), "gpu": 1,
                      "summary": report.get("summary"), "gate_passed": report.get("gate_passed"),
                      "automatic_expansion": False})
        if not complete:
            raise RuntimeError("failure preserved; no automatic retry")
        guard()
        (output/"complete").write_text("OK\n")


if __name__ == "__main__":
    main()

