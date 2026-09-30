import json
import multiprocessing
import os
import socket
from pathlib import Path

import pytest

from dllm_eval.queue import ElasticQueue, load_completed_records


def _consume(root, output):
    queue = ElasticQueue(root, total=43, identity={"dataset": "abc"}, chunk_size=3)
    claimed = []
    while (lease := queue.claim()) is not None:
        claimed.extend(lease.indices)
        queue.complete(lease)
    Path(output).write_text(json.dumps(claimed))


def _claim_and_exit(root):
    queue = ElasticQueue(root, total=1, identity={})
    assert queue.claim().indices == (0,)


def test_concurrent_workers_claim_each_index_once(tmp_path):
    root = tmp_path / "queue"
    ElasticQueue(root, total=43, identity={"dataset": "abc"}, chunk_size=3)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_consume, args=(root, tmp_path / f"worker{i}.json"))
                 for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    indices = [index for i in range(4)
               for index in json.loads((tmp_path / f"worker{i}.json").read_text())]
    assert sorted(indices) == list(range(43))
    assert ElasticQueue(root, total=43, identity={"dataset": "abc"}).status()["complete"]


def test_resume_partial_chunk_and_import(tmp_path):
    queue = ElasticQueue(tmp_path, total=8, identity={"method": "native"},
                         chunk_size=3, completed_indices=[0, 3])
    lease = queue.claim()
    assert lease.indices == (1, 2, 4)
    queue.complete(lease, [1, 4])
    assert queue.status()["completed"] == 4
    lease2 = queue.claim()
    assert lease2.indices == (2, 5, 6)
    queue.reconcile_completed([2, 5])
    queue.complete(lease2)
    final = queue.claim()
    assert final.indices == (7,)
    queue.complete(final)
    queue.complete(final)  # idempotent after durable output reconciliation
    assert queue.status()["complete"]


def test_live_claim_never_expires_and_release(tmp_path):
    queue = ElasticQueue(tmp_path, total=1, identity={})
    lease = queue.claim()
    state = json.loads(queue.path.read_text())
    state["active"][lease.claim_id]["claimed_at"] = 0
    queue.path.write_text(json.dumps(state))
    assert queue.recover_dead() == []
    assert queue.claim() is None
    queue.release(lease)
    assert queue.claim().indices == (0,)


def test_rank_shards_and_membership_change_do_not_duplicate_live_claims(tmp_path):
    from dllm_eval.scheduler import shard_indices
    queue = ElasticQueue(tmp_path, total=13, identity={}, completed_indices=[0, 3, 7])
    plan = shard_indices(queue.remaining_indices(), {0: 100, 2: 200, 7: 300})
    leases = [queue.claim(row["indices"]) for row in plan.values()]
    assert len({lease.indices[0] for lease in leases}) == 3
    queue.complete(leases[0])
    queue.release(leases[1])  # a worker leaving makes its prompt available again
    new_plan = shard_indices(queue.remaining_indices(), {0: 100, 7: 300})
    completed = []
    for row in new_plan.values():
        while (lease := queue.claim(row["indices"])) is not None:
            completed.extend(lease.indices)
            queue.complete(lease)
    assert leases[2].indices[0] not in completed  # live claim cannot be stolen
    queue.complete(leases[2])
    assert queue.status()["complete"]
    assert queue.remaining_indices() == []
    assert queue.claim([]) is None


def test_recover_only_proven_dead_same_host(tmp_path, monkeypatch):
    import dllm_eval.queue as module
    queue = ElasticQueue(tmp_path, total=3, identity={})
    lease = queue.claim()
    state = json.loads(queue.path.read_text())
    state["active"][lease.claim_id]["host"] = "other-machine"
    queue.path.write_text(json.dumps(state))
    assert queue.recover_dead() == []
    state["active"][lease.claim_id]["host"] = socket.gethostname()
    queue.path.write_text(json.dumps(state))
    # Simulate an authoritative liveness check, avoiding live PID probing on Windows.
    monkeypatch.setattr(module, "_owner_dead", lambda owner: owner["host"] == socket.gethostname())
    assert queue.recover_dead() == [0]
    assert queue.claim().indices == (0,)
    with pytest.raises(ValueError, match="no longer exists"):
        queue.complete(lease)


def test_identity_and_index_validation(tmp_path):
    queue = ElasticQueue(tmp_path, total=2, identity={"sha": "a"})
    with pytest.raises(ValueError, match="identity"):
        ElasticQueue(tmp_path, total=2, identity={"sha": "b"})
    with pytest.raises(ValueError):
        queue.reconcile_completed([2])
    with pytest.raises(ValueError):
        queue.reconcile_completed([1, True])
    lease = queue.claim()
    with pytest.raises(ValueError, match="outside"):
        queue.complete(lease, [1])


def test_completed_record_integrity_and_partial_tail(tmp_path):
    path = tmp_path / "records.jsonl"
    row = {"index": 0, "flash_native": {"seconds": 1, "text": "answer"}}
    path.write_text(json.dumps(row) + '\n{"index":')
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_completed_records([path], expected_count=2)
    assert load_completed_records([path], expected_count=2,
                                  required_methods=["flash_native"],
                                  allow_truncated_tail=True) == {0: row}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_completed_records([path], expected_count=2)
    assert load_completed_records([path], expected_count=2,
                                  allow_identical_duplicates=True) == {0: row}
    changed = row | {"flash_native": {"seconds": 2, "text": "different"}}
    path.write_text(json.dumps(row) + "\n" + json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_completed_records([path], expected_count=2, allow_identical_duplicates=True)
    path.write_text(json.dumps({"index": 5}) + "\n")
    with pytest.raises(ValueError, match="Invalid prompt"):
        load_completed_records([path], expected_count=2)
    path.write_text(json.dumps({"index": 0}) + "\n")
    with pytest.raises(ValueError, match="missing methods"):
        load_completed_records([path], expected_count=2, required_methods=["flash_native"])
    path.write_bytes((json.dumps(row) + '\n{"index": 1, "text": "').encode() + b'\xe4\xb8')
    assert load_completed_records([path], expected_count=2,
                                  allow_truncated_tail=True) == {0: row}


@pytest.mark.skipif(os.name == "nt", reason="Authoritative PID probing uses Linux /proc")
def test_recover_real_exited_process(tmp_path):
    queue = ElasticQueue(tmp_path, total=1, identity={})
    child = multiprocessing.get_context("spawn").Process(target=_claim_and_exit, args=(tmp_path,))
    child.start()
    child.join(20)
    assert child.exitcode == 0
    assert queue.status()["active"] == 1
    assert queue.recover_dead() == [0]
    assert queue.claim().indices == (0,)
