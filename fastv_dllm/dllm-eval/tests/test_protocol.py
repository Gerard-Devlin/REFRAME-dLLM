from dllm_eval.protocol import main_jobs


def test_main_matrix_has_exactly_24_cells():
    jobs = main_jobs({task: task for task in ("gsm8k", "math", "mbpp", "humaneval")})
    assert len(jobs) == 16
    assert sum(len(job.methods.split()) for job in jobs) == 24
    for job in jobs:
        assert job.label in ("llada", "fastdllm_ours_cache")
        assert (job.cache, job.decoding) == (("none", "single") if job.label == "llada" else ("prefix", "threshold"))
    assert sum(job.limit * len(job.methods.split()) for job in jobs) == 41898
