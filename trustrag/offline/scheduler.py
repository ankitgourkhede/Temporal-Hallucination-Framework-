"""Config-driven offline crawl-and-reindex scheduler, disabled by default."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


class OfflineScheduler:
    """Schedule one offline crawl/reindex job and log each run's summary."""

    def __init__(
        self,
        job_fn: Callable[[], dict | None] | None = None,
        *,
        enabled: bool = False,
        cron: str = "30 2 * * *",
    ):
        self.job_fn = job_fn
        self.enabled = enabled
        self.cron = cron
        self.logger = logging.getLogger("trustrag.offline.scheduler")
        self.scheduler = BackgroundScheduler()
        if self.enabled and self.job_fn is not None:
            self.scheduler.add_job(
                self.run_once,
                CronTrigger.from_crontab(self.cron),
                id="crawl_and_reindex",
                replace_existing=True,
            )

    @classmethod
    def from_config(cls, config_path: str, job_fn: Callable[[], dict | None]) -> "OfflineScheduler":
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        crawler = config.get("crawler", {})
        return cls(job_fn, enabled=bool(crawler.get("enabled", False)), cron=str(crawler.get("cron", "30 2 * * *")))

    def schedule(self, job_fn: Callable[[], dict | None]) -> None:
        self.job_fn = job_fn
        if self.enabled:
            self.scheduler.add_job(
                self.run_once,
                CronTrigger.from_crontab(self.cron),
                id="crawl_and_reindex",
                replace_existing=True,
            )

    def run_once(self) -> None:
        if self.job_fn is not None:
            started = perf_counter()
            result = self.job_fn() or {}
            elapsed = perf_counter() - started
            counts = {key: result.get(key, 0) for key in ("seen", "new", "changed", "unchanged", "inactive")}
            self.logger.info("crawl_and_reindex %s seconds=%.2f", counts, elapsed)

    def start(self) -> None:
        if self.enabled and not self.scheduler.running:
            self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
