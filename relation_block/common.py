import hashlib
import json
from pathlib import Path

MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
# Published upstream revision, not a floating main. All arms share these weights.
REVISION = "da5608172d2b74380e4e780baa19c71645e4f981"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def snapshot(offline=True):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=offline,
        allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.jinja"]))


def manifest(data):
    data = Path(data)
    m = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in m["files"].items():
        if digest(data / name) != expected:
            raise ValueError(f"Data changed: {name}; prepare a new directory")
    return m
