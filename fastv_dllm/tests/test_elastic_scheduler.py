from pathlib import Path

import pytest

from fastv_dllm.elastic_scheduler import (
    Scheduler, Worker, descendant_pids, find_workers, plan_launches, queue_accounting,
    refresh_queue_statuses,
)


def test_cap_counts_live_workers_before_cuda_allocation():
    assert plan_launches(range(8), {gpu: 5 for gpu in range(8)}, {0, 1, 2, 3, 4},
                         max_total_gpus=6, stable_checks=2, pending_prompts=100) == [5]
    assert plan_launches(range(8), {gpu: 5 for gpu in range(8)}, range(6),
                         max_total_gpus=6, stable_checks=2, pending_prompts=100) == []
    with pytest.raises(RuntimeError, match="cap"):
        plan_launches(range(8), {}, range(7), max_total_gpus=6,
                      stable_checks=2, pending_prompts=100)


def test_idle_stability_and_pending_limit():
    assert plan_launches(range(4), {0: 1, 1: 2, 2: 3, 3: 3}, {2},
                         max_total_gpus=3, stable_checks=2, pending_prompts=1) == [1]
    assert plan_launches(range(4), {0: 10}, {}, max_total_gpus=3,
                         stable_checks=2, pending_prompts=0) == []


def test_eta_counts_claimed_prompts_without_claiming_done():
    jobs = [{"name": "slow", "seconds_per_prompt": 100},
            {"name": "fast", "seconds_per_prompt": 5}]
    statuses = {"slow": dict(total=10, completed=4, active=2, pending=4),
                "fast": dict(total=10, completed=9, active=1, pending=0)}
    output = queue_accounting(jobs, statuses, worker_count=3)
    assert output["total"] == 20
    assert output["completed"] == 13
    assert output["active"] == 3
    assert output["pending"] == 4
    assert output["eta_seconds_at_current_workers"] == pytest.approx(605 / 3)
    assert queue_accounting(jobs, statuses, 0)["eta_seconds_at_current_workers"] is None


def test_adopt_same_root_refuse_foreign_and_legacy(tmp_path):
    args = ["python", "-m", "fastv_dllm.elastic_paper", "worker", "--run-root", str(tmp_path)]
    processes = {10: {"argv": args + ["--gpu", "1"], "ppid": 1},
                 11: {"argv": ["python", "-m", "fastv_dllm.llada_evaluate"], "ppid": 1},
                 12: {"argv": ["python", "-m", "fastv_dllm.elastic_paper", "worker",
                               "--run-root", str(tmp_path / "other"), "--gpu", "2"], "ppid": 1}}
    workers, unknown = find_workers(processes, tmp_path)
    assert workers == {1: 10}
    assert unknown == [11, 12]
    processes[13] = {"argv": args + ["--gpu", "1"], "ppid": 1}
    with pytest.raises(RuntimeError, match="Duplicate"):
        find_workers(processes, tmp_path)


def test_shell_commands_mentioning_worker_are_not_workers(tmp_path):
    argv = ["bash", "-lc", "python", "-m", "fastv_dllm.elastic_paper", "worker",
            "--run-root", str(tmp_path), "--gpu", "1"]
    assert find_workers({10: {"argv": argv, "ppid": 1}}, tmp_path) == ({}, [])


def test_abandoned_last_claim_becomes_pending_before_scheduling():
    class Queue:
        pending = 0
        def recover_dead(self):
            self.pending = 1
            return [3]
        def status(self):
            return {"pending": self.pending}
    statuses, recovered = refresh_queue_statuses({"job": Queue()})
    assert statuses["job"]["pending"] == 1
    assert recovered == {"job": [3]}


def test_gpu_ownership_allows_own_children_only():
    processes = {10: {"ppid": 1}, 11: {"ppid": 10}, 12: {"ppid": 11}, 13: {"ppid": 1}}
    assert descendant_pids(10, processes) == {10, 11, 12}


def test_external_collision_drains_only_own_worker(tmp_path):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.worker_dir = tmp_path
    scheduler.workers = {1: Worker(1, 10, "start"), 3: Worker(3, 20, "start")}
    scheduler.failed = None
    events = []
    scheduler.event = events.append
    rows = {1: {"compute_pids": [10, 11, 99]}, 3: {"compute_pids": [20]}}
    processes = {10: {"ppid": 1}, 11: {"ppid": 10}, 20: {"ppid": 1}, 99: {"ppid": 1}}
    scheduler.check_collisions(rows, processes)
    assert scheduler.workers[1].stopping
    assert not scheduler.workers[3].stopping
    assert "99" in (tmp_path / "1.stop").read_text()
    assert not (tmp_path / "3.stop").exists()
    scheduler.check_collisions(rows, processes)
    assert len(events) == 1
