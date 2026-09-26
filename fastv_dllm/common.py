"""Dataset and reporting helpers shared by the original-LLaDA experiment."""

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_ids(tokenizer, question, task="gsm8k", *, preformatted=False):
    """Apply the LLaDA chat template to a task prompt.

    ``preformatted`` is used for prompts produced by lm-eval.  Those strings
    already contain the paper's few-shot demonstrations and task formatting;
    appending the local zero-shot instruction would change the benchmark.
    """
    if task == "gsm8k" and not preformatted:
        question = question + "\nExplain your reasoning and end with #### followed by the final number."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
    )


def extract_answer(text, gold=False):
    if gold:
        text = text.split("####")[-1]
    else:
        explicit = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
        boxed = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", text)
        values = explicit or boxed or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not values:
            return None
        text = values[-1]
    try:
        return str(Decimal(text.strip().replace(",", "")).normalize())
    except InvalidOperation:
        return None


def load_samples(path, limit, task="gsm8k"):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not 0 < limit <= len(rows):
        raise ValueError(f"Requested {limit} samples; dataset has {len(rows)}")
    required = {
        "gsm8k": {"id", "question", "answer"},
        "math": {"id", "paper_prompt", "answer", "generation_kwargs"},
        "humaneval": {"task_id", "prompt", "canonical_solution", "test", "entry_point"},
        "mbpp": {"task_id", "paper_prompt", "test_list", "generation_kwargs"},
    }[task]
    if any(not required.issubset(row) for row in rows[:limit]):
        raise ValueError(f"Dataset rows do not contain the required {task} fields")
    return rows[:limit]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
