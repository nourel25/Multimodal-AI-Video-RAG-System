import logging
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .LLMInterface import LLMError

T = TypeVar("T")
logger = logging.getLogger(__name__)


def call_with_retry(
    fn: Callable[[], T],
    *,
    provider: str,
    is_transient: Callable[[BaseException], bool],
    max_attempts: int = 3,
) -> T:
    """Run `fn`, retrying transient errors (rate limits, 5xx, timeouts) with jittered backoff."""
    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, max=20),
        retry=retry_if_exception(is_transient),
        reraise=False,
        before_sleep=lambda state: logger.warning(
            "%s call failed (attempt %s), retrying: %s",
            provider, state.attempt_number, state.outcome.exception(),
        ),
    )
    try:
        return retrying(fn)
    except RetryError as exc:
        cause = exc.last_attempt.exception()
        raise LLMError(str(cause), provider=provider, transient=True) from cause
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc), provider=provider, transient=is_transient(exc)) from exc
