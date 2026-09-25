"""Pinned-model contracts and evaluation helpers."""

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
REVISION = "da5608172d2b74380e4e780baa19c71645e4f981"
CODE_HASH = "d363ee4a4d4bf52958645d5c715712c5b027525bb90611a52178e58695e09b50"
MASK_ID = 151665
EOS_ID = 151645


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    actual = sha256(path / "modeling.py")
    if actual != CODE_HASH:
        raise RuntimeError(f"Pinned modeling.py mismatch: {actual}; expected {CODE_HASH}")
    return path


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + "\nExplain your reasoning and end with #### followed by the final number."}],
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


def load_samples(path, limit):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not 0 < limit <= len(rows):
        raise ValueError(f"Requested {limit} samples; dataset has {len(rows)}")
    required = {"id", "question", "answer"}
    if any(not required.issubset(row) for row in rows[:limit]):
        raise ValueError("Dataset rows must contain id/question/answer")
    return rows[:limit]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
