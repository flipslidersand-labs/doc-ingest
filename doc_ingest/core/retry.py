"""Generic retry with exponential backoff (1s, 2s, 4s, ...)."""

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    retries: int,
    is_retryable: Callable[[Exception], bool],
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call *fn* up to *retries* times; sleep 2**attempt seconds between attempts.

    A non-retryable exception, or the last attempt's exception, is re-raised unchanged.
    *on_retry(attempt, exc)* runs before each sleep (attempt is 0-based) — never after the
    final attempt. ``time.sleep`` is looked up at call time so tests can patch it.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1 or not is_retryable(e):
                raise
            if on_retry is not None:
                on_retry(attempt, e)
            time.sleep(2**attempt)
    raise ValueError("retries must be >= 1")
