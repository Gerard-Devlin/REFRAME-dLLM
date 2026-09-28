"""Opportunistically schedule the paper matrix without exceeding a GPU cap."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .wait_for_idle_gpus import snapshot


@dataclass(frozen=True)
class Job:
    task: str
    gen: int
    label: str
    cache: str
    decoding: str
    methods: str
    dataset: str
    limit: int

    @property
    def name(self):
        return f"{self.task}_g{self.gen}_{self.label}"

    @property
    def cost(self):
        factor = {"llada": 1.0, "cache": 0.30, "parallel_ours": 0.32,
                  "fastdllm_ours_cache": 0.12}[self.label]
        return self.limit * self.gen * factor


def build_jobs(datasets):
    limits = {"gsm8k": 1319, "math": 5000, "humaneval": 164, "mbpp": 500}
    configs = (
        ("llada", "none", "single", "flash_native"),
        ("cache", "prefix", "single", "flash_native"),
        ("parallel_ours", "none", "threshold", "flash_native flash_fastv_head"),
        ("fastdllm_ours_cache", "prefix", "threshold",
         "flash_native flash_fastv_head"),
    )
    return [Job(task, gen, label, cache, decoding, methods,
                str(datasets[task]), limits[task])
            for task in limits for gen in (256, 512)
            for label, cache, decoding, methods in configs]


def complete(run):
    run = Path(run)
    if (run / "exit_code").exists() and (run / "exit_code").read_text().strip() == "0":
        return True
    summary_path = run / "output" / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        expected = len(summary["ids"])
        records = []
        for path in sorted((run / "output").glob("rank_*.jsonl")):
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        if sorted(row["index"] for row in records) != list(range(expected)):
            return False
        methods = [key for key in summary["results"] if key != "attribution"]
        return bool(methods) and all(
            summary["results"][key]["examples"] == expected for key in methods
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.root = args.run_root
        self.state_dir = args.state_dir
        self.jobs = build_jobs({
            "gsm8k": args.gsm8k_dataset, "math": args.math_dataset,
            "humaneval": args.humaneval_dataset, "mbpp": args.mbpp_dataset,
        })
        self.active = {}
        self.current_active = not complete(args.current_run)
        self.current_gpus = set(args.current_gpus)
        self.free_streak = {index: 0 for index in args.candidates}
        self.failed = None
        self.stop = False
        self.event_log = self.state_dir / "events.log"

    def event(self, message):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
        print(line, flush=True)
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def transition_main(self):
        if not self.current_active or not (self.args.current_run / "exit_code").exists():
            return
        code = (self.args.current_run / "exit_code").read_text().strip()
        if code != "0" and not complete(self.args.current_run):
            self.failed = f"Current main run failed with {code}"
            return
        subprocess.run(["tmux", "kill-session", "-t", self.args.main_tmux],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.current_active = False
        self.event(f"took over after {self.args.current_run.name}; released "
                   f"GPUs={','.join(map(str, sorted(self.current_gpus)))}")

    def poll_children(self):
        for name, item in list(self.active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            run = item["run"]
            (run / "exit_code").write_text(str(code) + "\n")
            (run / "finished_at").write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + "\n")
            del self.active[name]
            if code == 0 and complete(run):
                self.event(f"complete job={name} GPUs={item['gpus']}")
                continue
            text = (run / "job.log").read_text(errors="replace")
            failed = run.with_name(run.name + ".scheduler_failed_" + time.strftime("%Y%m%d_%H%M%S"))
            shutil.move(str(run), str(failed))
            if "Selected GPU already has compute PID" in text or "Visible GPU count mismatch" in text:
                self.event(f"GPU race job={name}; requeued as {failed.name}")
            else:
                self.failed = f"job={name} exit={code}; log={failed / 'job.log'}"
                self.event("FAILED " + self.failed)

    def used_gpus(self):
        used = sum(len(item["gpus"]) for item in self.active.values())
        if self.current_active:
            used += len(self.current_gpus)
        if used > self.args.max_total_gpus:
            raise RuntimeError(f"GPU cap violated: {used}>{self.args.max_total_gpus}")
        return used

    def eligible_jobs(self):
        active = set(self.active)
        jobs = [job for job in self.jobs
                if job.name not in active and not complete(self.root / job.name)]
        if self.current_active:
            # These small, distant jobs cannot collide with the forward-order
            # legacy campaign before the handoff completes.
            jobs = [job for job in jobs if job.task in ("humaneval", "mbpp")]
        return sorted(jobs, key=lambda job: (-job.cost, job.name))

    def launch(self, job, gpus):
        run = self.root / job.name
        if run.exists():
            stale = run.with_name(run.name + ".scheduler_stale_" + time.strftime("%Y%m%d_%H%M%S"))
            shutil.move(str(run), str(stale))
        run.mkdir(parents=True)
        env = os.environ.copy()
        env.update({
            "REPO": str(self.args.repo), "GPU_IDS": ",".join(map(str, gpus)),
            "MODE": "evaluate", "RUN_DIR": str(run), "TASK": job.task,
            "DATASET": job.dataset, "LIMIT": str(job.limit), "GEN_LENGTH": str(job.gen),
            "BLOCK_LENGTH": "32", "THRESHOLD": "0.90", "DECODING_MODE": job.decoding,
            "CACHE_MODE": job.cache, "METHODS": job.methods, "PRUNE_AFTER_LAYER": "4",
            "SUPPORT_KEEP_RATIO": "0.3125", "REQUIRE_IDLE": "1", "SKIP_TESTS": "1",
        })
        log = (run / "job.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            ["bash", str(self.args.repo / "fastv_dllm" / "scripts" / "job.sh")],
            cwd=self.args.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active[job.name] = {"process": process, "gpus": tuple(gpus),
                                 "run": run, "log": log, "job": job}
        (run / "scheduler_assignment.json").write_text(json.dumps({
            "job": asdict(job), "gpus": gpus, "pid": process.pid,
            "started_at": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        }, indent=2))
        self.event(f"start job={job.name} GPUs={','.join(map(str, gpus))} pid={process.pid}")

    def write_state(self, rows):
        state = {
            "time": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            "max_total_gpus": self.args.max_total_gpus,
            "used_gpus": self.used_gpus(), "current_main_active": self.current_active,
            "active": {name: {"gpus": item["gpus"], "pid": item["process"].pid}
                       for name, item in self.active.items()},
            "pending": [job.name for job in self.eligible_jobs()],
            "gpu_snapshot": rows, "failed": self.failed,
        }
        tmp = self.state_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_dir / "state.json")

    def terminate(self):
        for item in self.active.values():
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self):
        self.event(f"scheduler online cap={self.args.max_total_gpus} "
                   f"current={sorted(self.current_gpus) if self.current_active else []}")
        while not self.stop:
            self.transition_main()
            self.poll_children()
            try:
                rows = snapshot(set(self.args.candidates), self.args.max_memory_mib,
                                self.args.max_utilization)
            except Exception as error:
                self.event(f"GPU query error: {error}")
                time.sleep(self.args.poll_seconds)
                continue
            idle_now = {index for index, idle, *_ in rows if idle}
            assigned = {gpu for item in self.active.values() for gpu in item["gpus"]}
            if self.current_active:
                assigned |= self.current_gpus
            for index in self.free_streak:
                self.free_streak[index] = (self.free_streak[index] + 1
                                           if index in idle_now and index not in assigned else 0)
            slots = self.args.max_total_gpus - self.used_gpus()
            eligible_gpus = [index for index in self.args.candidates
                             if self.free_streak[index] >= self.args.stable_checks]
            jobs = self.eligible_jobs()
            if slots > 0 and eligible_gpus and jobs and not self.failed:
                selected = eligible_gpus[:slots]
                # Recheck immediately before process creation to narrow the race window.
                verify = snapshot(set(selected), self.args.max_memory_mib,
                                  self.args.max_utilization)
                selected = [index for index, idle, *_ in verify if idle]
                if selected:
                    self.launch(jobs[0], selected)
                    for index in selected:
                        self.free_streak[index] = 0
            self.write_state(rows)
            all_done = all(complete(self.root / job.name) for job in self.jobs)
            if all_done and not self.active and not self.current_active:
                (self.state_dir / "complete").write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                self.event("entire paper matrix complete")
                return 0
            if self.failed and not self.active:
                return 1
            time.sleep(self.args.poll_seconds)
        return 130


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--main-tmux", required=True)
    parser.add_argument("--current-gpus", default="1,7")
    parser.add_argument("--candidates", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-total-gpus", type=int, default=6)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=5)
    for task in ("gsm8k", "math", "humaneval", "mbpp"):
        parser.add_argument(f"--{task}-dataset", type=Path, required=True)
    args = parser.parse_args()
    args.current_gpus = tuple(int(value) for value in args.current_gpus.split(","))
    args.candidates = tuple(int(value) for value in args.candidates.split(","))
    if args.max_total_gpus > len(args.candidates):
        parser.error("GPU cap exceeds candidate count")
    return args


def main():
    args = parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=False)
    scheduler = Scheduler(args)

    def stop(*_):
        scheduler.stop = True
        scheduler.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(scheduler.run())


if __name__ == "__main__":
    main()
