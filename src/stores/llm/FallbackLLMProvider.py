import logging
from collections.abc import Iterator

from .LLMInterface import ChatMessage, LLMError, LLMInterface

logger = logging.getLogger(__name__)


class FallbackLLMProvider(LLMInterface):
    """Tries the primary provider and switches to the fallback when it fails.

    Streaming only falls back if the primary fails before producing any token,
    so a client never receives two concatenated half-answers.
    """

    def __init__(self, primary: LLMInterface, fallback: LLMInterface):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}->{fallback.name}"
        self.model_id = primary.model_id

    def generate(self, messages: list[ChatMessage], **kwargs) -> str:
        try:
            return self.primary.generate(messages, **kwargs)
        except LLMError as exc:
            logger.warning("Primary LLM failed, using fallback %s: %s", self.fallback.name, exc)
            return self.fallback.generate(messages, **kwargs)

    def stream(self, messages: list[ChatMessage], **kwargs) -> Iterator[str]:
        started = False
        try:
            for token in self.primary.stream(messages, **kwargs):
                started = True
                yield token
        except LLMError as exc:
            if started:
                raise
            logger.warning("Primary LLM stream failed, using fallback %s: %s",
                           self.fallback.name, exc)
            yield from self.fallback.stream(messages, **kwargs)

    def describe_image(self, image: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        try:
            return self.primary.describe_image(image, prompt, mime_type)
        except (LLMError, NotImplementedError) as exc:
            logger.warning("Primary vision call failed, using fallback: %s", exc)
            return self.fallback.describe_image(image, prompt, mime_type)
