"""Wait until a stable set of physical NVIDIA GPUs is genuinely idle."""

import argparse
import subprocess
import sys
import time


def csv(command):
    return subprocess.check_output(command, text=True).strip().splitlines()


def snapshot(candidates, max_memory_mib, max_utilization):
    busy_uuids = set()
    for line in csv([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]):
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[0].startswith("GPU-"):
            busy_uuids.add(fields[0])

    result = []
    for line in csv([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]):
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        index, memory, utilization = int(index), int(memory), int(utilization)
        if index not in candidates:
            continue
        idle = (uuid not in busy_uuids and memory <= max_memory_mib
                and utilization <= max_utilization)
        result.append((index, idle, memory, utilization, uuid in busy_uuids))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--candidates", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=5)
    args = parser.parse_args()
    candidates = tuple(dict.fromkeys(int(value) for value in args.candidates.split(",")))
    if args.count <= 0 or args.count > len(candidates):
        raise SystemExit("Invalid requested GPU count")

    previous, stable = None, 0
    while True:
        try:
            rows = snapshot(set(candidates), args.max_memory_mib, args.max_utilization)
        except (subprocess.CalledProcessError, ValueError) as error:
            print(f"GPU query failed: {error}; retrying", file=sys.stderr, flush=True)
            previous, stable = None, 0
            time.sleep(args.poll_seconds)
            continue
        by_index = {row[0]: row for row in rows}
        free = tuple(index for index in candidates
                     if index in by_index and by_index[index][1])
        selected = free[:args.count] if len(free) >= args.count else None
        if selected is not None and selected == previous:
            stable += 1
        elif selected is not None:
            previous, stable = selected, 1
        else:
            previous, stable = None, 0
        state = " ".join(
            f"{idx}:{'free' if idle else 'busy'}({mem}MiB,{util}%,pid={int(pid)})"
            for idx, idle, mem, util, pid in rows
        )
        print(f"idle check {stable}/{args.stable_checks}: {state}",
              file=sys.stderr, flush=True)
        if selected is not None and stable >= args.stable_checks:
            print(",".join(map(str, selected)), flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
