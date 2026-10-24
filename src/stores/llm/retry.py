import logging
import re
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .LLMInterface import LLMError
from .rate_limit import RateLimiter

T = TypeVar("T")
logger = logging.getLogger(__name__)

MAX_RETRY_AFTER_SECONDS = 60.0
# Gemini puts the delay in the error body, e.g. "retryDelay": "17s".
_RETRY_DELAY_IN_BODY = re.compile(r"retry[_ ]?delay\W+(\d+(?:\.\d+)?)s", re.IGNORECASE)


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """The wait a provider asked for on a rate-limit error, if it said."""
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        value = headers.get("retry-after")
        try:
            if value is not None:
                return float(value)
        except ValueError:
            pass
    match = _RETRY_DELAY_IN_BODY.search(str(exc))
    return float(match.group(1)) if match else None


class wait_retry_after:
    """Honor the provider's Retry-After when given, else jittered exponential backoff."""

    def __init__(self, fallback=None, max_wait: float = MAX_RETRY_AFTER_SECONDS):
        self.fallback = fallback or wait_random_exponential(multiplier=1, max=20)
        self.max_wait = max_wait

    def __call__(self, retry_state: RetryCallState) -> float:
        requested = retry_after_seconds(retry_state.outcome.exception())
        if requested is not None:
            return min(requested, self.max_wait)
        return self.fallback(retry_state)


def call_with_retry(
    fn: Callable[[], T],
    *,
    provider: str,
    is_transient: Callable[[BaseException], bool],
    max_attempts: int = 3,
    limiter: RateLimiter | None = None,
) -> T:
    """Run `fn` under the provider's rate limit, retrying transient errors (429, 5xx, timeouts)."""

    def attempt() -> T:
        if limiter is not None:
            limiter.acquire()
        return fn()

    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_retry_after(),
        retry=retry_if_exception(is_transient),
        reraise=False,
        before_sleep=lambda state: logger.warning(
            "%s call failed (attempt %s), retrying in %.1fs: %s",
            provider, state.attempt_number, state.upcoming_sleep, state.outcome.exception(),
        ),
    )
    try:
        return retrying(attempt)
    except RetryError as exc:
        cause = exc.last_attempt.exception()
        raise LLMError(str(cause), provider=provider, transient=True) from cause
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc), provider=provider, transient=is_transient(exc)) from exc
