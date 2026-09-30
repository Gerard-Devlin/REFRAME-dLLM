from pathlib import Path

import pytest

from dllm_eval.scheduler import (
    Scheduler, Worker, descendant_pids, find_workers, plan_launches, queue_accounting,
    refresh_queue_statuses,
    select_task, shard_indices,
)


def test_one_task_waits_for_last_active_prompt_before_advancing():
    jobs = [{"name": "first"}, {"name": "second"}]
    status = {"first": dict(complete=False, pending=0, active=1),
              "second": dict(complete=False, pending=100, active=0)}
    assert select_task(jobs, status) == "first"
    assert select_task(jobs, status, "first") == "first"
    status["first"]["complete"] = True
    assert select_task(jobs, status, "first") == "second"
    status["second"]["complete"] = True
    assert select_task(jobs, status, "second") is None


@pytest.mark.parametrize("world", [1, 2, 4, 5, 6])
def test_round_robin_is_balanced_disjoint_and_complete(world):
    ids = [1, 4, 8, 10, 20, 23, 45, 47, 48, 50, 51, 52, 54]
    workers = {gpu*2: 100+gpu for gpu in range(world)}
    shards = shard_indices(ids, workers)
    sizes = [len(row["indices"]) for row in shards.values()]
    assert max(sizes) - min(sizes) <= 1
    assert sorted(index for row in shards.values() for index in row["indices"]) == ids
    for rank, row in enumerate(shards.values()):
        assert row["indices"] == ids[rank::world]
    assert shard_indices(ids, {}) == {}


def test_dispatch_repartitions_only_on_task_or_membership_change(tmp_path):
    from dllm_eval.queue import ElasticQueue
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.jobs = [{"name": "first"}, {"name": "second"}]
    scheduler.queues = {name: ElasticQueue(tmp_path / name, total=9, identity={})
                        for name in ("first", "second")}
    scheduler.active_job = None
    scheduler.dispatch_signature = None
    scheduler.stop = False
    scheduler.workers = {0: Worker(0, 10, None), 3: Worker(3, 20, None)}
    scheduler.dispatch_path = tmp_path / "dispatch.json"
    scheduler.event = lambda unused: None
    statuses = {name: queue.status() for name, queue in scheduler.queues.items()}
    scheduler.dispatch(statuses)
    first = scheduler.dispatch_path.read_bytes()
    scheduler.dispatch(statuses)
    assert scheduler.dispatch_path.read_bytes() == first
    scheduler.queues["first"].reconcile_completed([0, 3, 5])
    scheduler.workers[5] = Worker(5, 30, None)
    scheduler.dispatch({name: queue.status() for name, queue in scheduler.queues.items()})
    assert sorted(index for row in scheduler.assignments.values() for index in row["indices"]) == [1, 2, 4, 6, 7, 8]
    assert scheduler.active_job == "first"


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


@pytest.mark.parametrize("module", ["focus_dllm.llada_evaluate", "fastv_dllm.llada_evaluate"])
def test_adopt_same_root_refuse_foreign_and_legacy(tmp_path, module):
    args = ["python", "-m", "dllm_eval.worker", "worker", "--run-root", str(tmp_path)]
    processes = {10: {"argv": args + ["--gpu", "1"], "ppid": 1},
                 11: {"argv": ["python", "-m", module], "ppid": 1},
                 12: {"argv": ["python", "-m", "dllm_eval.worker", "worker",
                               "--run-root", str(tmp_path / "other"), "--gpu", "2"], "ppid": 1}}
    workers, unknown = find_workers(processes, tmp_path)
    assert workers == {1: 10}
    assert unknown == [11, 12]
    processes[13] = {"argv": args + ["--gpu", "1"], "ppid": 1}
    with pytest.raises(RuntimeError, match="Duplicate"):
        find_workers(processes, tmp_path)


def test_shell_commands_mentioning_worker_are_not_workers(tmp_path):
    argv = ["bash", "-lc", "python", "-m", "dllm_eval.worker", "worker",
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
