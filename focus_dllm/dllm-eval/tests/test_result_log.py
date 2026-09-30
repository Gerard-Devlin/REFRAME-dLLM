from dllm_eval.protocol import main_config
from dllm_eval.result_log import format_task_result, log_task_result


def report():
    return dict(configuration=dict(task="gsm8k", gen_length=256, methods=["flash_native"],
                                   cache_mode="prefix", decoding_mode="threshold"),
                results={"flash_native": dict(examples=10, accuracy=.7, provisional_accuracy=.9,
                    accuracy_metric="lm_eval.gsm8k.exact_match,flexible-extract",
                    mean_seconds=2, p50_seconds=1.9, p95_seconds=3, mean_nfe=64,
                    throughput=80, total_seconds=20)})


def test_final_metric_and_method_label_are_unambiguous():
    text = format_task_result("gsm8k_g256_fastdllm", report(), main_config())
    assert "Fast-dLLM v1" in text and "70.00" in text and "90.00" not in text
    assert "flexible-extract" in text and "NFE/req" in text
    assert "not multi-GPU wall time" in text
    data = report()
    data["configuration"].update(cache_mode="none", decoding_mode="single")
    assert "LLaDA-original" in format_task_result("original", data, main_config())


def test_zero_score_and_missing_score_are_not_confused():
    data = report()
    data["results"]["flash_native"].update(accuracy=0, accuracy_metric="pass@1")
    text = format_task_result("code", data)
    assert "pass@1" in text and "0.00" in text
    data["results"]["flash_native"]["accuracy"] = None
    del data["results"]["flash_native"]["accuracy_metric"]
    assert "provisional (not officially scored)" in format_task_result("pending", data)


def test_summary_appends_to_both_existing_logs_only(tmp_path, capsys):
    (tmp_path / "elastic").mkdir()
    for path in (tmp_path / "job.log", tmp_path / "elastic/progress.log"):
        path.write_text("existing progress\n", encoding="utf-8")
    log_task_result(tmp_path, "gsm8k_g256_fastdllm", report(), main_config())
    for path in (tmp_path / "job.log", tmp_path / "elastic/progress.log"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("existing progress\n")
        assert text.count("TASK RESULT:") == 2  # opening and closing labels
    assert "70.00" in capsys.readouterr().out
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == [
        str(__import__("pathlib").Path("elastic/progress.log")), "job.log"]
