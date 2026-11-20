"""Request-correlated stage timing helpers for the research pipeline."""

import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger("research_agent.timing")
logger.setLevel(logging.INFO)


def configure_diagnostic_logging() -> None:
    """Send diagnostic INFO logs to stdout using one idempotent handler."""
    try:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not any(getattr(handler, "_research_agent_diagnostic", False) for handler in logger.handlers):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            handler._research_agent_diagnostic = True
            logger.addHandler(handler)
    except Exception:
        # Diagnostics must never prevent the API from starting.
        pass


def process_rss_mb():
    """Return process RSS in MiB when the host exposes it, without a dependency."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            fields = statm.read().split()
            if len(fields) < 2:
                raise ValueError("/proc/self/statm has no resident-page field")
            resident_pages = int(fields[1])
            if resident_pages < 0:
                raise ValueError("resident page count cannot be negative")
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return None


def _safe_info(message, *args) -> None:
    try:
        logger.info(message, *args)
    except Exception:
        # Diagnostics must never change request behavior.
        pass


def log_process_rss(stage: str, request_id: str = "unknown", **fields) -> None:
    """Log RSS when available; suppress all diagnostic failures."""
    try:
        rss_mb = process_rss_mb()
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        _safe_info(
            "[MEMORY] %s rss_mb=%s request_id=%s%s",
            stage,
            f"{rss_mb:.1f}" if rss_mb is not None else "unavailable",
            request_id,
            f" {details}" if details else "",
        )
    except Exception:
        # A broken logger or host memory API must not break research requests.
        pass


@contextmanager
def measure_stage(name: str, request_id: str = "startup"):
    started = time.perf_counter()
    _safe_info("[TIMING][%s] stage_start name=%s", request_id, name)
    try:
        yield
    finally:
        _safe_info(
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
