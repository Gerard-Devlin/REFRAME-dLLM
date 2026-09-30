"""Print scored task results to the existing logs, without presentation files."""
from datetime import datetime, timezone
from pathlib import Path


def _table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(title), *(len(row[i]) for row in rows))
              for i, title in enumerate(headers)]
    def line(row):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
    return "\n".join([line(headers), line(["-" * width for width in widths]),
                      *(line(row) for row in rows)])


def format_task_result(job, report, protocol=None):
    """Use the finalized primary metric, never the provisional extraction score."""
    config = report["configuration"]
    protocol = protocol or {}
    task = config["task"]
    shots = protocol.get("datasets", {}).get(task, {}).get("fewshot", "N/A")
    quality, speed = [], []
    for method in config["methods"]:
        result = report["results"][method]
        name = next((row["name"] for row in protocol.get("rows", [])
                     if row["method"] == method and row["cache"] == config["cache_mode"]
                     and row["decoding"] == config["decoding_mode"]), method)
        def number(key, scale=1, decimals=3):
            value = result.get(key)
            return "N/A" if value is None else f"{value * scale:.{decimals}f}"
        metric = result.get("accuracy_metric", "provisional (not officially scored)")
        quality.append([task, name, str(shots), str(config["gen_length"]),
                        str(result["examples"]), metric, number("accuracy", 100, 2)])
        speed.append([name, number("mean_seconds"), number("p50_seconds"),
                      number("p95_seconds"), number("mean_nfe", decimals=2),
                      number("throughput", decimals=2), number("total_seconds", decimals=1)])
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return "\n".join([
        f"[{stamp}] TASK RESULT: {job}",
        _table(["Task", "Method", "n-shot", "Gen", "N", "Metric", "Score (%)"], quality),
        _table(["Method", "Mean s/req", "P50 s", "P95 s", "NFE/req", "Output tok/s", "Sum request s"], speed),
        "Timing: measured generation only; summed request time is not multi-GPU wall time.",
        f"END TASK RESULT: {job}",
    ])


def log_task_result(root, job, report, protocol=None):
    text = format_task_result(job, report, protocol)
    # CPU finalizers are serialized. Write the whole block in one append per log;
    # the running GPU workers and their decoder timing are unaffected.
    for path in (Path(root) / "job.log", Path(root) / "elastic" / "progress.log"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + text + "\n\n")
    print(text, flush=True)

