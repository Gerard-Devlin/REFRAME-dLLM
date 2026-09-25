"""Contracts for the original LLaDA-8B-Instruct checkpoint."""

from pathlib import Path

from .common import extract_answer, load_samples, prompt_ids, sha256, write_json

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c33c00c7563e963b07"
MASK_ID = 126336


def snapshot():
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    if not (path / "config.json").is_file():
        raise RuntimeError("Pinned LLaDA checkpoint is incomplete")
    return path


__all__ = [
    "MODEL_ID", "REVISION", "MASK_ID", "snapshot", "extract_answer", "load_samples",
    "prompt_ids", "sha256", "write_json",
]
