"""Download pinned Fast-dLLM v2 checkpoints into the configured HF cache."""

import argparse
import json
import os
import time
from pathlib import Path

from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

from .common import MODEL_ID, REVISION


MODELS = {
    "1.5b": (MODEL_ID, REVISION),
    "7b": (
        "Efficient-Large-Model/Fast_dLLM_v2_7B",
        "0661abf5f9f0ee338970d091052a26c8efa51974",
    ),
}


def download(size: str, retries: int = 20) -> Path:
    from huggingface_hub import snapshot_download

    if size not in MODELS:
        raise ValueError(f"Unknown model size: {size}")
    if retries < 1:
        raise ValueError("retries must be positive")
    repo_id, revision = MODELS[size]
    for attempt in range(1, retries + 1):
        try:
            path = Path(snapshot_download(
                repo_id,
                revision=revision,
                local_files_only=False,
                allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.jinja"],
                max_workers=int(os.environ.get("WEIGHTS_DOWNLOAD_WORKERS", "1")),
            ))
            index = path / "model.safetensors.index.json"
            if index.exists():
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            else:
                shards = {item.name for item in path.glob("*.safetensors")}
            if not shards or any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in shards):
                raise RuntimeError(f"Incomplete model weights in {path}")
            print(f"COMPLETE {size}: {path} ({len(shards)} weight shards)", flush=True)
            return path
        except (ChunkedEncodingError, ConnectionError, Timeout) as exc:
            print(f"{size} interrupted on attempt {attempt}/{retries}: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(min(15 * attempt, 120))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1.5b")
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()
    for size in args.sizes.lower().split(","):
        download(size.strip(), args.retries)


if __name__ == "__main__":
    main()
