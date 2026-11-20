"""Request-correlated stage timing helpers for the research pipeline."""

import logging
import os
import time
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger("research_agent.timing")
logger.setLevel(logging.INFO)


def process_rss_mb():
    """Return process RSS in MiB when the host exposes it, without a dependency."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return None


def log_process_rss(stage: str, request_id: str = "unknown") -> None:
    rss_mb = process_rss_mb()
    if rss_mb is not None:
        logger.info("[MEMORY][%s] stage=%s rss_mb=%.1f", request_id, stage, rss_mb)


@contextmanager
def measure_stage(name: str, request_id: str = "startup"):
    started = time.perf_counter()
    logger.info("[TIMING][%s] stage_start name=%s", request_id, name)
    try:
        yield
    finally:
        logger.info(
            "[TIMING][%s] stage_complete name=%s duration_seconds=%.3f",
            request_id,
            name,
            time.perf_counter() - started,
        )


def timed_stage(name: str):
    """Decorate a function with the same request-aware timing log format."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with measure_stage(name, kwargs.get("request_id", "unknown")):
                return function(*args, **kwargs)
        return wrapped
    return decorate
