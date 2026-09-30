"""Schedule independent paper-evaluation workers with a strict GPU cap."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

from .elastic_work import ElasticQueue, _owner_dead, _process_info


def plan_launches(candidates, free_streak, occupied, *, max_total_gpus,
                  stable_checks, pending_prompts):
    """Count live/reserved workers even before they create a CUDA context."""
    occupied = set(occupied)
    if len(occupied) > max_total_gpus:
        raise RuntimeError("Existing workers already exceed the GPU cap")
    slots = min(max_total_gpus - len(occupied), pending_prompts)
    return [gpu for gpu in candidates if gpu not in occupied
            and free_streak.get(gpu, 0) >= stable_checks][:slots]


def queue_accounting(jobs, statuses, worker_count):
    total = sum(statuses[name]["total"] for name in statuses)
    completed = sum(statuses[name]["completed"] for name in statuses)
    active = sum(statuses[name]["active"] for name in statuses)
    pending = sum(statuses[name]["pending"] for name in statuses)
    remaining_gpu_seconds = sum(
        max(float(job["seconds_per_prompt"]), 0.0)
        * (statuses[job["name"]]["pending"] + statuses[job["name"]]["active"])
        for job in jobs)
    return dict(total=total, completed=completed, active=active, pending=pending,
                remaining_gpu_hours=remaining_gpu_seconds / 3600,
                eta_seconds_at_current_workers=(remaining_gpu_seconds / worker_count
                                               if worker_count else None))


def gpu_snapshot():
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    by_uuid = {}
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].startswith("GPU-"):
            by_uuid.setdefault(parts[0], set()).add(int(parts[1]))
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True)
    result = {}
    for line in devices.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        result[int(index)] = dict(index=int(index), uuid=uuid, memory_mib=int(memory),
                                  utilization=int(utilization),
                                  compute_pids=sorted(by_uuid.get(uuid, set())))
    return result


def process_snapshot():
    raw = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    result = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            argv = shlex.split(parts[2])
        except ValueError:
            argv = parts[2].split()
        result[int(parts[0])] = dict(ppid=int(parts[1]), argv=argv)
    return result


def _argument(argv, name):
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _is_python_module(argv, module):
    # ps flattens argument quoting, so an ssh/bash '-c' script can contain the
    # same tokens. Only the actual Python launcher is an owned worker.
    return bool(argv and Path(argv[0]).name.startswith("python") and any(
        value == "-m" and index + 1 < len(argv) and argv[index + 1] == module
        for index, value in enumerate(argv)))


def refresh_queue_statuses(queues):
    recovered, statuses = {}, {}
    for name, queue in queues.items():
        indices = queue.recover_dead()
        if indices:
            recovered[name] = indices
        statuses[name] = queue.status()
    return statuses, recovered


def find_workers(processes, run_root):
    """Return known same-run workers and other evaluation processes to block on."""
    workers, unknown = {}, []
    for pid, process in processes.items():
        argv = process["argv"]
        if _is_python_module(argv, "fastv_dllm.llada_evaluate"):
            unknown.append(pid)
            continue
        if not _is_python_module(argv, "fastv_dllm.elastic_paper") or "worker" not in argv:
            continue
        root, gpu = _argument(argv, "--run-root"), _argument(argv, "--gpu")
        if root is None or gpu is None or Path(root).resolve() != Path(run_root).resolve():
            unknown.append(pid)
            continue
        try:
            gpu = int(gpu)
        except ValueError:
            unknown.append(pid)
            continue
        if gpu in workers:
            raise RuntimeError(f"Duplicate live workers on GPU {gpu}: {workers[gpu]}, {pid}")
        workers[gpu] = pid
    return workers, unknown


def descendant_pids(pid, processes):
    result = {pid}
    while True:
        more = {child for child, data in processes.items() if data["ppid"] in result}
        if more <= result:
            return result
        result.update(more)


@dataclass
class Worker:
    gpu: int
    pid: int
    process_start: str | None
    process: subprocess.Popen | None = None
    log: object = None
    stopping: bool = False


@dataclass
class Finalizer:
    job: str
    pid: int
    process_start: str | None
    process: subprocess.Popen | None = None
    log: object = None


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.root = args.run_root / "elastic"
        self.root.mkdir(exist_ok=True, parents=True)
        self.worker_dir = self.root / "workers"
        self.worker_dir.mkdir(exist_ok=True)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.jobs = self.manifest["jobs"]
        self.queues = {job["name"]: ElasticQueue(
            self.root / "queues" / job["name"], total=job["job"]["limit"],
            identity=job["identity"]) for job in self.jobs}
        self.workers = {}
        self.finalizer = None
        self.free_streak = {gpu: 0 for gpu in args.candidates}
        self.failed = None
        self.stop = False

    def event(self, message):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
        print(line, flush=True)
        with (self.root / "events.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def fail(self, message):
        if self.failed is None:
            self.failed = message
            self.event("PAUSED new launches: " + message)

    def request_stop(self, worker, reason):
        if worker.stopping:
            return
        (self.worker_dir / f"{worker.gpu}.stop").write_text(reason + "\n")
        worker.stopping = True
        self.event(f"drain GPU={worker.gpu} PID={worker.pid}: {reason}")

    def adopt(self, processes):
        observed, unknown = find_workers(processes, self.args.run_root)
        if unknown:
            self.fail(f"Unmanaged evaluation processes exist: {unknown}")
        for gpu, pid in observed.items():
            if gpu in self.workers:
                if self.workers[gpu].pid != pid:
                    self.fail(f"GPU {gpu} has unexpected worker PID {pid}")
                continue
            start, _ = _process_info(pid)
            self.workers[gpu] = Worker(gpu, pid, start,
                                       stopping=(self.worker_dir / f"{gpu}.stop").exists())
            self.event(f"adopt existing worker GPU={gpu} PID={pid}")
        if len(self.workers) > self.args.max_total_gpus:
            self.fail("Existing live worker count exceeds requested GPU cap")
            # These are explicitly identified own workers, never foreign PIDs.
            for gpu in sorted(self.workers)[self.args.max_total_gpus:]:
                self.request_stop(self.workers[gpu], "GPU cap recovery")
        finalizers = []
        for pid, data in processes.items():
            argv = data["argv"]
            if not _is_python_module(argv, "fastv_dllm.elastic_paper") or "finalize" not in argv:
                continue
            root, job = _argument(argv, "--run-root"), _argument(argv, "--job")
            if root is not None and Path(root).resolve() == self.args.run_root and job:
                finalizers.append((pid, job))
        if len(finalizers) > 1:
            self.fail(f"Multiple pre-existing CPU finalizers: {finalizers}")
        elif finalizers:
            pid, job = finalizers[0]
            if self.finalizer is None:
                start, _ = _process_info(pid)
                self.finalizer = Finalizer(job, pid, start)
                self.event(f"adopt existing CPU finalizer job={job} PID={pid}")
            elif self.finalizer.pid != pid:
                self.fail(f"Unexpected extra CPU finalizer PID={pid}")

    def poll_workers(self):
        for gpu, worker in list(self.workers.items()):
            code = worker.process.poll() if worker.process is not None else None
            dead = _owner_dead(dict(pid=worker.pid, host=socket.gethostname(),
                                    process_start=worker.process_start))
            if code is None and not dead:
                continue
            if worker.log is not None:
                worker.log.close()
            if code is None:
                path = self.worker_dir / f"{gpu}.json"
                status = json.loads(path.read_text()) if path.exists() else {}
                terminal = status.get("status") in ("stopped", "complete")
                code = 0 if terminal and status.get("pid") == worker.pid else 1
            if code != 0:
                self.fail(f"worker GPU={gpu} PID={worker.pid} exit={code}; inspect worker log")
            self.event(f"worker retired GPU={gpu} PID={worker.pid} exit={code}")
            del self.workers[gpu]

    def check_collisions(self, rows, processes):
        for gpu, worker in self.workers.items():
            if gpu not in rows:
                self.fail(f"Assigned GPU {gpu} vanished from nvidia-smi")
                self.request_stop(worker, "GPU disappeared from inventory")
                continue
            own = descendant_pids(worker.pid, processes)
            foreign = set(rows[gpu]["compute_pids"]) - own
            if foreign:
                self.request_stop(worker, f"external compute PID collision {sorted(foreign)}")

    def launch(self, gpu, row):
        stop_path = self.worker_dir / f"{gpu}.stop"
        if stop_path.exists():
            stop_path.unlink()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = row["uuid"]
        env["OMP_NUM_THREADS"] = "1"
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)
        log_path = self.worker_dir / f"gpu{gpu}_{time.time_ns()}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen([
            sys.executable, "-u", "-m", "fastv_dllm.elastic_paper", "worker",
            "--run-root", str(self.args.run_root), "--gpu", str(gpu)
        ], cwd=self.args.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        start, _ = _process_info(process.pid)
        self.workers[gpu] = Worker(gpu, process.pid, start, process, log)
        self.free_streak[gpu] = 0
        self.event(f"start GPU={gpu} PID={process.pid} log={log_path.name}")

    def poll_finalizer(self):
        if self.finalizer is None:
            return
        item = self.finalizer
        code = item.process.poll() if item.process is not None else None
        dead = _owner_dead(dict(pid=item.pid, host=socket.gethostname(),
                                process_start=item.process_start))
        if code is None and not dead:
            return
        if item.log is not None:
            item.log.close()
        self.finalizer = None
        marker = self.root / "finalized" / f"{item.job}.json"
        if code is None:
            code = 0 if marker.exists() else 1
        if code != 0 or not marker.exists():
            self.fail(f"finalizer {item.job} exit={code}, marker_exists={marker.exists()}")
        else:
            self.event(f"finalized {item.job}")

    def start_finalizer(self, statuses):
        if self.finalizer is not None or self.failed or self.stop:
            return
        for job in self.jobs:
            name = job["name"]
            if not statuses[name]["complete"] or (self.root / "finalized" / f"{name}.json").exists():
                continue
            log_dir = self.root / "finalizer_logs"
            log_dir.mkdir(exist_ok=True)
            log = (log_dir / f"{name}.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            process = subprocess.Popen([
                sys.executable, "-u", "-m", "fastv_dllm.elastic_paper", "finalize",
                "--run-root", str(self.args.run_root), "--job", name
            ], cwd=self.args.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            start, _ = _process_info(process.pid)
            self.finalizer = Finalizer(name, process.pid, start, process, log)
            self.event(f"finalizing {name} CPU PID={process.pid}")
            return

    def write_state(self, rows, statuses):
        state = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                     max_total_gpus=self.args.max_total_gpus,
                     used_gpus=len(self.workers), failed=self.failed, stopping=self.stop,
                     workers={gpu: dict(pid=w.pid, stopping=w.stopping)
                              for gpu, w in self.workers.items()},
                     finalizer=(self.finalizer.job if self.finalizer else None),
                     progress=queue_accounting(self.jobs, statuses, len(self.workers)),
                     jobs={name: {key: value for key, value in status.items() if key != "claims"}
                           for name, status in statuses.items()}, gpu_snapshot=rows,
                     eta_note="Estimate from measured prompt cost; active prompts counted in full.")
        path = self.root / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(path)

    def run(self):
        self.event(f"elastic scheduler online max_total_gpus={self.args.max_total_gpus}")
        while True:
            self.poll_workers()
            self.poll_finalizer()
            if self.stop:
                for worker in self.workers.values():
                    self.request_stop(worker, "scheduler stopped")
            try:
                processes, rows = process_snapshot(), gpu_snapshot()
                self.adopt(processes)
            except Exception as error:
                self.event(f"Inventory query failed, no launches: {error}")
                time.sleep(self.args.poll_seconds)
                continue
            self.check_collisions(rows, processes)
            statuses, recovered = refresh_queue_statuses(self.queues)
            for name, indices in recovered.items():
                self.event(f"recovered proven-dead claims job={name} prompts={len(indices)}")
            all_done = all(status["complete"] for status in statuses.values())
            if self.stop or all_done:
                for worker in self.workers.values():
                    self.request_stop(worker, "scheduler stopped" if self.stop else "all prompts complete")
            for gpu in self.free_streak:
                row = rows.get(gpu)
                idle = (row is not None and not row["compute_pids"]
                        and row["memory_mib"] <= self.args.max_memory_mib
                        and row["utilization"] <= self.args.max_utilization
                        and gpu not in self.workers)
                self.free_streak[gpu] = self.free_streak[gpu] + 1 if idle else 0
            if not self.failed and not self.stop:
                choices = plan_launches(
                    self.args.candidates, self.free_streak, self.workers,
                    max_total_gpus=self.args.max_total_gpus,
                    stable_checks=self.args.stable_checks,
                    pending_prompts=sum(status["pending"] for status in statuses.values()))
                for gpu in choices:
                    # Repeat the external-process/idle check immediately before launch.
                    try:
                        check = gpu_snapshot().get(gpu)
                    except Exception as error:
                        self.event(f"Launch recheck failed: {error}")
                        break
                    if (check and not check["compute_pids"]
                            and check["memory_mib"] <= self.args.max_memory_mib
                            and check["utilization"] <= self.args.max_utilization):
                        self.launch(gpu, check)
            self.start_finalizer(statuses)
            self.write_state(rows, statuses)
            finalized = all((self.root / "finalized" / f"{job['name']}.json").exists()
                            for job in self.jobs)
            if all_done and finalized and not self.workers and self.finalizer is None:
                (self.root / "complete").write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                self.event("entire matrix finalized")
                return 0
            if (self.stop or self.failed) and not self.workers and self.finalizer is None:
                return 130 if self.stop else 1
            time.sleep(self.args.poll_seconds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--max-total-gpus", type=int, default=6)
    parser.add_argument("--candidates", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=5)
    args = parser.parse_args()
    args.repo, args.run_root = args.repo.resolve(), args.run_root.resolve()
    args.candidates = tuple(int(value) for value in args.candidates.split(","))
    if (len(set(args.candidates)) != len(args.candidates)
            or not 1 <= args.max_total_gpus <= min(6, len(args.candidates))
            or args.stable_checks < 1 or args.poll_seconds <= 0):
        parser.error("Invalid candidates, cap (maximum 6), or polling settings")
    return args


def main():
    if os.name == "nt":
        raise SystemExit("The GPU scheduler requires Linux process ownership checks")
    import fcntl
    args = parse_args()
    with ExitStack() as stack:
        directory = args.repo / "fastv_dllm" / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("paper_elastic_scheduler.lock", "paper_smart_scheduler.lock"):
            stream = stack.enter_context((directory / name).open("a"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"Another evaluation scheduler owns {name}") from None
        scheduler = Scheduler(args)
        def stop(*_):
            scheduler.stop = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        raise SystemExit(scheduler.run())


if __name__ == "__main__":
    main()
