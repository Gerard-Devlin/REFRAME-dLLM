from dllm_eval.progress import ProgressLog


def state(done=900):
    return dict(time="now", active_job="math", used_gpus=6, max_total_gpus=6,
                failed=None, finalizer=None,
                progress=dict(total=1000, completed=done, eta_seconds_at_current_workers=500),
                jobs={"math": dict(total=1000, completed=done, active=6, complete=done == 1000)})


def test_progress_resume_uses_only_new_completed_prompts_and_plain_lines():
    log = ProgressLog(30)
    lines = log.lines(state(), now=100)
    assert "900/1000" in lines[-1]
    assert "? prompt/min" in lines[-1]
    assert log.lines(state(906), now=110) == []
    lines = log.lines(state(912), now=130)
    assert "24.00 prompt/min" in lines[-1]  # 12 new prompts in 30 seconds
    assert "task ETA~00:03:40" in lines[-1]
    assert all("\r" not in line and "\x1b" not in line for line in lines)
    assert "GPUs=6/6" in lines[0]


def test_new_task_resets_rate_and_prints_immediately():
    log = ProgressLog()
    log.lines(state(), now=100)
    new = state(0)
    new["jobs"] = {"next": new["jobs"].pop("math")}
    new["active_job"] = "next"
    assert "? prompt/min" in log.lines(new, now=101)[-1]
