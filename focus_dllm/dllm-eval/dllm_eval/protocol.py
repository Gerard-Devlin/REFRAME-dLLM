"""Three-method experiment configuration kept separate from the scheduler."""
from dataclasses import dataclass
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "tasks" / "main_table.json"


def main_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Job:
    task: str
    gen: int
    label: str
    cache: str
    decoding: str
    methods: str
    dataset: str
    limit: int

    @property
    def name(self):
        return f"{self.task}_g{self.gen}_{self.label}"


def main_jobs(datasets):
    config = main_config()
    # One row is one complete task, even when two rows share decoding settings.
    # All GPUs finish this method's prompts before the next method begins.
    return [Job(task, gen, row["job_label"], row["cache"], row["decoding"],
                row["method"], str(datasets[task]), spec["examples"])
            for task, spec in config["datasets"].items() for gen in config["generation_lengths"]
            for row in config["rows"]]
