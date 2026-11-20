"""Shared Gemini generation call with retries for transient service outages."""

import logging
import time
from services.timing import measure_stage

from google.genai import errors

logger = logging.getLogger(__name__)


def generate_content(client, **kwargs):
    """Retry Gemini 503 UNAVAILABLE responses up to three times."""
    delays = (2, 4, 8)
    for attempt in range(len(delays) + 1):
        try:
            with measure_stage("Gemini generation attempt", kwargs.get("request_id", "unknown")):
                call_kwargs = {key: value for key, value in kwargs.items() if key != "request_id"}
                return client.models.generate_content(**call_kwargs)
        except errors.APIError as exc:
            if getattr(exc, "code", None) != 503 or attempt == len(delays):
                raise
            delay = delays[attempt]
            logger.warning(
                "Gemini returned 503 UNAVAILABLE; retrying in %ss (%s/%s)",
                delay,
                attempt + 1,
                len(delays),
            )
            with measure_stage("Gemini retry backoff", kwargs.get("request_id", "unknown")):
                time.sleep(delay)
