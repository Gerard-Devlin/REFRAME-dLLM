# Prompt/answer handling adapted from v1/llada/eval_llada.py.
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""lm-eval entry point, sharing upstream prompts/scoring/model initialization.

Only generation is replaced. Original v1 source and globals remain untouched.
Use --model reframe_llada --batch_size 1 (NOT duplicate batch_size in args).
"""
from dataclasses import asdict
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from tqdm import tqdm

from eval_llada import LLaDAEvalHarness
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from reframe_dllm.generate import generate_reframe
from reframe_dllm.model import ReframeConfig


@register_model("reframe_llada")
class ReframeEvalHarness(LLaDAEvalHarness):
    def __init__(self, reframe_kind="pair", reframe_pilots=16,
                 reframe_refresh_blocks=2, reframe_ridge=1e-3,
                 reframe_max_pilot_error=0.25, reframe_backend="flash",
                 reframe_materialize=False, reframe_log="reframe/results/metrics",
                 **kwargs):
        if int(kwargs.get("batch_size", 1)) != 1:
            raise ValueError("REFRAME supports --batch_size 1 only")
        if kwargs.get("save_dir") is not None:
            raise ValueError("Resume is not implemented in REFRAME; use a fresh evaluation run")
        kwargs["batch_size"] = 1
        super().__init__(**kwargs)
        if self.threshold is None and self.factor is None:
            raise ValueError("Specify threshold=0.9 or factor=1.0")
        self.reframe_config = ReframeConfig(kind=reframe_kind, pilots=int(reframe_pilots),
                                           refresh_blocks=int(reframe_refresh_blocks),
                                           ridge=float(reframe_ridge),
                                           max_pilot_error=float(reframe_max_pilot_error),
                                           backend=reframe_backend,
                                           materialize=reframe_materialize)
        rank = self.accelerator.process_index if self.accelerator is not None else 0
        self.reframe_log = Path(reframe_log) / f"rank_{rank}.jsonl"
        self.reframe_log.parent.mkdir(parents=True, exist_ok=True)
        if self.reframe_log.exists():
            raise ValueError(f"Metrics file already exists: {self.reframe_log}; set a new reframe_log")
        self.reframe_log.touch(exist_ok=False)

    @torch.no_grad()
    def generate_until(self, requests):
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        output = []
        total_time, total_tokens, total_nfe = 0.0, 0, 0
        with self.reframe_log.open("a", encoding="utf-8") as log:
            for req in tqdm(requests, desc="REFRAME generation"):
                question = req.args[0]
                if self.is_instruct:
                    question = self.tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                                  add_generation_prompt=True, tokenize=False)
                prompt = torch.tensor(self.tokenizer(question)["input_ids"], device=self.device).unsqueeze(0)
                result, nfe, stats = generate_reframe(model, prompt, gen_length=self.gen_length,
                                                     block_length=self.block_length, threshold=self.threshold,
                                                     factor=self.factor, mask_id=self.mask_id,
                                                     config=self.reframe_config)
                ids = result[0, prompt.shape[1]:]
                # Keep v1's HumanEval treatment and general stop extraction.
                if self.is_instruct and str(req.doc.get("task_id", "")).lower().startswith("humaneval"):
                    count = int((ids != 126081).sum().item())
                    answer = self.tokenizer.decode(ids, skip_special_tokens=True)
                else:
                    answer = self.tokenizer.decode(ids, skip_special_tokens=False)
                    for stop in req.args[1]["until"]:
                        if stop and stop in answer:
                            answer = answer.split(stop)[0]
                    extracted = self.tokenizer(answer)["input_ids"]
                    count = sum(t != 126081 for t in extracted)
                    answer = self.tokenizer.decode(extracted, skip_special_tokens=True)
                output.append(answer)
                total_time += stats["elapsed_seconds"]
                total_tokens += count
                total_nfe += nfe
                log.write(json.dumps(dict(doc_id=req.doc_id, config=asdict(self.reframe_config),
                                          useful_tokens_v1=count, stats=stats)) + "\n")
                log.flush()
        if self.show_speed:
            print(json.dumps(dict(generation_seconds=total_time, tokens_v1=total_tokens, nfe=total_nfe,
                                  tokens_per_second=total_tokens / max(total_time, 1e-12),
                                  timing_scope="generation only; includes init/pilots/fit/commit/fallback")))
        return output


if __name__ == "__main__":
    cli_evaluate()
