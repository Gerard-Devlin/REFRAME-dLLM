"""CPU-only, newline-safe progress meters for multi-GPU evaluation logs."""
from collections import deque
import time


def duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressLog:
    """Aggregate all ranks; never count imported results as newly generated."""

    def __init__(self, interval=30):
        self.interval = interval
        self.history = {}
        self.last_print = -float("inf")
        self.last_job = None

    def sample(self, key, count, now):
        history = self.history.setdefault(key, deque())
        if history and count < history[-1][1]:
            history.clear()
        history.append((now, count))
        while len(history) > 2 and history[1][0] < now - 300:
            history.popleft()
        elapsed = now - history[0][0]
        return ((count - history[0][1]) / elapsed if elapsed >= 10 else None)

    def lines(self, state, now=None):
        from tqdm import tqdm
        now = time.monotonic() if now is None else now
        name = state.get("active_job")
        progress = state["progress"]
        rate = self.sample(name, state["jobs"][name]["completed"], now) if name else None
        changed = name != self.last_job
        if not changed and now - self.last_print < self.interval:
            return []
        self.last_print, self.last_job = now, name
        prefix = f"[{state['time']}]"
        total = progress["total"]
        finished = sum(job["complete"] for job in state["jobs"].values())
        scoring = state.get("finalizer") or "idle"
        lines = [f"{prefix} TOTAL {progress['completed']}/{total} prompts | "
                 f"generated tasks={finished}/{len(state['jobs'])} | "
                 f"GPUs={state['used_gpus']}/{state['max_total_gpus']} | "
                 f"matrix ETA~{duration(progress['eta_seconds_at_current_workers'])} | "
                 f"CPU scoring={scoring}"]
        if name:
            job = state["jobs"][name]
            hist = self.history[name]
            # tqdm's default rate inference would include pre-existing records.
            # Supply our measured delta-rate, and render unknown ETA during warm-up.
            meter = tqdm.format_meter(
                job["completed"], job["total"], now - hist[0][0],
                rate=rate if rate and rate > 0 else 1e-99,
                prefix=f"{name} ", ascii=True, unit="prompt",
                bar_format="{desc}{percentage:5.1f}%|{bar:24}| {n_fmt}/{total_fmt}")
            eta = ((job["total"] - job["completed"]) / rate if rate and rate > 0 else None)
            speed = f"{rate*60:.2f}" if rate and rate > 0 else "?"
            lines.append(f"{prefix} {meter} | {speed} prompt/min | "
                         f"task ETA~{duration(eta)} | in-flight={job['active']}")
        if state.get("failed"):
            lines.append(f"{prefix} PAUSED: {state['failed']}")
        return lines
