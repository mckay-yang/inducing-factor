import logging
import time
from typing import Mapping


class ProgressTracker:
    def __init__(self, logger: logging.Logger, label: str, log_every: int = 1):
        self.logger = logger
        self.label = label
        self.log_every = max(1, int(log_every))
        self.stage_index = 0
        self.stage_total = 0
        self.stage_name = ""
        self.total = 0
        self.current = 0
        self.started_at = 0.0

    def start_stage(self, stage_index: int, stage_total: int, stage_name: str, total: int = 0):
        self.stage_index = stage_index
        self.stage_total = stage_total
        self.stage_name = stage_name
        self.total = max(0, int(total))
        self.current = 0
        self.started_at = time.monotonic()
        self.logger.info("[%s][stage %s/%s] %s started", self.label, stage_index, stage_total, stage_name)

    def advance(self, step: int = 1, extra_stats: Mapping[str, object] | None = None):
        self.current += step
        if self.current % self.log_every != 0 and self.current != self.total:
            return
        elapsed = max(0.0, time.monotonic() - self.started_at)
        if self.total > 0 and self.current > 0:
            pct = (self.current / self.total) * 100.0
            eta = elapsed * max(self.total - self.current, 0) / self.current
            progress_text = f"{self.current}/{self.total} ({pct:.1f}%) elapsed={self._fmt(elapsed)} eta={self._fmt(eta)}"
        else:
            progress_text = f"steps={self.current} elapsed={self._fmt(elapsed)}"
        extras = ""
        if extra_stats:
            extras = " " + " ".join(f"{key}={value}" for key, value in extra_stats.items())
        self.logger.info(
            "[%s][stage %s/%s] %s %s%s",
            self.label,
            self.stage_index,
            self.stage_total,
            self.stage_name,
            progress_text,
            extras,
        )

    def finish_stage(self, extra_stats: Mapping[str, object] | None = None):
        elapsed = max(0.0, time.monotonic() - self.started_at)
        extras = ""
        if extra_stats:
            extras = " " + " ".join(f"{key}={value}" for key, value in extra_stats.items())
        self.logger.info(
            "[%s][stage %s/%s] %s finished elapsed=%s%s",
            self.label,
            self.stage_index,
            self.stage_total,
            self.stage_name,
            self._fmt(elapsed),
            extras,
        )

    def _fmt(self, seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
