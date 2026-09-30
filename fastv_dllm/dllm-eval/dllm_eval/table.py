"""The fresh three-method paper table; only fully scored cells are displayed."""
import argparse
from html import escape
import json
from pathlib import Path


TASKS = (("gsm8k", "GSM8K (5-shot)"), ("math", "MATH (4-shot)"),
         ("humaneval", "HumanEval (0-shot)"), ("mbpp", "MBPP (3-shot)"))
METHODS = (("LLaDA-original", "llada", "flash_native"),
           ("Fast-dLLM v1", "fastdllm_ours_cache", "flash_native"),
           ("Ours", "fastdllm_ours_cache", "flash_fastv_head"))


def table_rows(root):
    root = Path(root)
    manifest = json.loads((root / "elastic" / "manifest.json").read_text())
    if manifest.get("matrix") != "main":
        raise ValueError("Only the fresh main matrix belongs in this table")
    jobs = {job["name"]: job for job in manifest["jobs"]}
    rows = []
    for method, suffix, key in METHODS:
        for length in (256, 512):
            row = dict(method=method, gen_length=length, cells={})
            for task, _ in TASKS:
                name = f"{task}_g{length}_{suffix}"
                marker = root / "elastic" / "finalized" / f"{name}.json"
                cell = None
                if marker.exists():
                    report = json.loads((root / name / "output" / "summary.json").read_text())
                    job = jobs[name]
                    identity = job["identity"]
                    if (report["dataset_sha256"] != identity["dataset_sha256"] or
                            report["model"] != identity["model"] or report["revision"] != identity["revision"]):
                        raise ValueError("Table cell provenance mismatch")
                    cell = dict(report["results"][key])
                    if cell["examples"] != job["job"]["limit"] or cell["accuracy"] is None:
                        raise ValueError("Table requires fully scored, complete cells")
                    cell["source"] = str(Path(name) / "output" / "summary.json")
                    cell["dataset_sha256"] = identity["dataset_sha256"]
                    if method == "Ours":
                        base = report["results"]["flash_native"]
                        cell["speedup_vs_fastdllm"] = base["total_seconds"] / cell["total_seconds"]
                        cell["accuracy_delta_vs_fastdllm_pp"] = 100 * (cell["accuracy"] - base["accuracy"])
                row["cells"][task] = cell
            rows.append(row)
    for row in rows:
        original = next(other for other in rows if other["method"] == "LLaDA-original"
                        and other["gen_length"] == row["gen_length"])
        for task, cell in row["cells"].items():
            base = original["cells"][task]
            if cell is not None:
                cell["speedup_vs_original"] = base["total_seconds"] / cell["total_seconds"] if base else None
    return rows


def format_cell(cell):
    if cell is None:
        return "Pending"
    factor = cell.get("speedup_vs_original")
    speed = f"{factor:.2f}x" if factor is not None else "pending baseline"
    return (f"{cell['accuracy']*100:.2f}%<br>"
            f"{cell['throughput']:.2f} tok/s; {cell['mean_seconds']:.3f} s/question<br>{speed}")


def write_table(root):
    root = Path(root)
    rows = table_rows(root)
    headers = ["Method", "Gen length"] + [label for _, label in TASKS]
    values = [[row["method"], str(row["gen_length"])] +
              [format_cell(row["cells"][task]) for task, _ in TASKS] for row in rows]
    note = ("All methods: pinned LLaDA-8B-Instruct, BF16, FlashAttention, batch=1. "
            "Fast-dLLM v1 here explicitly means PrefixCache + parallel decoding (threshold=0.90, block=32); "
            "Ours adds FastV (layer=4, support keep=0.3125) and active-position LM head. "
            "This is not a claim about the best DualCache configuration. "
            "Speedup is total decode latency of original / method, not pooled six-GPU throughput. "
            "tok/s uses postprocessed output tokens. MATH primary metric is official Minerva exact_match; "
            "math_verify is retained separately in JSON. Completed cells only; historical runs excluded.")
    markdown = "# Fresh main evaluation\n\n" + " | ".join(headers) + "\n"
    markdown += " | ".join(["---"] * len(headers)) + "\n"
    markdown += "\n".join(" | ".join(value) for value in values) + "\n\n" + note + "\n"
    html = ("<!doctype html><html><meta charset='utf-8'><title>LLaDA comparison</title>"
            "<style>body{font:16px system-ui;margin:36px;background:#fafbfc;color:#172536}"
            "table{border-collapse:collapse;width:100%;background:white}"
            "th,td{padding:16px;border:1px solid #d7dde5;text-align:center}"
            "th{background:#eaf0f7}td:first-child{text-align:left}p{line-height:1.6;max-width:1200px}</style>"
            "<h1>LLaDA-8B-Instruct — fresh comparison</h1><table><thead><tr>" +
            "".join(f"<th>{escape(header)}</th>" for header in headers) + "</tr></thead><tbody>" +
            "".join("<tr>" + "".join(f"<td>{escape(value).replace('&lt;br&gt;', '<br>')}</td>"
                                       for value in line) + "</tr>" for line in values) +
            "</tbody></table><p>" + escape(note) + "</p></html>")
    for filename, content in (("table.md", markdown), ("table.html", html),
                              ("table.json", json.dumps(dict(rows=rows, note=note), indent=2))):
        target = root / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    write_table(parser.parse_args().run_root)
