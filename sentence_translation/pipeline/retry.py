"""Generic retry-with-backoff wrapper for transient API failures.

This handles *transport* failures (rate limits, timeouts, 5xx errors) for any
LLM/embedding call. It is unrelated to the pipeline-level "backfill" retries
in orchestrator.py, which regenerate sentences for a word that didn't reach
its passing-sentence quota -- that is a data-quality retry, not a transport one.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger("pipeline.retry")

# Errors where every retry is guaranteed to fail identically. Matched on message
# text rather than status code on purpose: a 429 means either "slow down" (retry)
# or "your credits are gone" (never retry), and a 403 is similarly ambiguous.
#
# This list is deliberately narrow. Anything not positively identified here is
# treated as transient, because abandoning a multi-hour run over a misread error
# is worse than burning a few wasted retries on a genuinely dead one.
_PERMANENT_ERROR_MARKERS = (
    "prepayment credits are depleted",
    "api key not valid",
    "api_key_invalid",
    "permission_denied",
)


def is_permanent(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_ERROR_MARKERS)


def call_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 4,
    base_backoff_seconds: float = 2.0,
    max_backoff_seconds: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: BLE001 - intentionally broad, caller narrows via retry_on
            last_error = exc
            if is_permanent(exc):
                logger.error("permanent failure, not retrying: %s", exc)
                break
            if attempt == max_attempts:
                break
            sleep_for = min(base_backoff_seconds * (2 ** (attempt - 1)), max_backoff_seconds)
            sleep_for *= 1 + random.uniform(-0.2, 0.2)  # jitter
            logger.warning(
                "attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error
