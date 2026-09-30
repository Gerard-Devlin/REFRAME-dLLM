from dllm_eval.protocol import main_jobs


def test_main_matrix_has_exactly_24_cells():
    jobs = main_jobs({task: task for task in ("gsm8k", "math", "mbpp", "humaneval")})
    assert len(jobs) == 24
    assert len({job.name for job in jobs}) == 24
    assert sum(len(job.methods.split()) for job in jobs) == 24
    for job in jobs:
        assert len(job.methods.split()) == 1
        assert job.label in ("llada", "fastdllm", "ours")
        assert (job.cache, job.decoding) == (("none", "single") if job.label == "llada" else ("prefix", "threshold"))
    assert sum(job.limit * len(job.methods.split()) for job in jobs) == 41898


def test_each_dataset_and_length_runs_all_three_methods_in_order():
    tasks = ("gsm8k", "humaneval", "mbpp", "math")
    jobs = main_jobs({task: task for task in tasks})
    assert [(job.task, job.gen, job.label, job.methods) for job in jobs] == [
        (task, gen, label, method)
        for task in tasks for gen in (256, 512)
        for label, method in (("llada", "flash_native"), ("fastdllm", "flash_native"),
                              ("ours", "flash_focus_head"))
    ]
