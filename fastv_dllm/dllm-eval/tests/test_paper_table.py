import json
import pytest
from dllm_eval.worker import main_jobs
from dllm_eval.table import table_rows, write_table


def test_main_matrix_has_exactly_24_cells():
    jobs = main_jobs({task: task for task in ("gsm8k", "math", "mbpp", "humaneval")})
    assert len(jobs) == 16
    assert sum(len(job.methods.split()) for job in jobs) == 24
    for job in jobs:
        assert job.label in ("llada", "fastdllm_ours_cache")
        assert (job.cache, job.decoding) == (("none", "single") if job.label == "llada" else ("prefix", "threshold"))
    assert sum(job.limit * len(job.methods.split()) for job in jobs) == 41898


def test_table_ignores_unscored_cell_and_renders_complete_cell(tmp_path):
    elastic = tmp_path / "elastic"
    (elastic / "finalized").mkdir(parents=True)
    name = "gsm8k_g256_fastdllm_ours_cache"
    job = dict(name=name, job=dict(limit=1319), identity=dict(model="llada", revision="rev", dataset_sha256="sha"))
    (elastic / "manifest.json").write_text(json.dumps(dict(matrix="main", jobs=[job])))
    output = tmp_path / name / "output"
    output.mkdir(parents=True)
    result = dict(examples=1319, accuracy=.8, throughput=42, total_seconds=3000, mean_seconds=3)
    report = dict(**job["identity"], results={"flash_native": result,
                  "flash_fastv_head": dict(result, accuracy=.79, total_seconds=2000)})
    (output / "summary.json").write_text(json.dumps(report))
    assert all(cell is None for row in table_rows(tmp_path) for cell in row["cells"].values())
    (elastic / "finalized" / f"{name}.json").write_text("{}")
    rows = table_rows(tmp_path)
    ours = rows[4]["cells"]["gsm8k"]
    assert ours["speedup_vs_fastdllm"] == 1.5
    assert ours["speedup_vs_original"] is None
    write_table(tmp_path)
    assert "79.00%" in (tmp_path / "table.md").read_text()
    assert "Pending" in (tmp_path / "table.html").read_text(encoding="utf-8")
    report["dataset_sha256"] = "wrong"
    (output / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="provenance"):
        table_rows(tmp_path)
