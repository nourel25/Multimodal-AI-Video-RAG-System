from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str  # one of ChatRole values
    content: str


class LLMError(RuntimeError):
    """Raised by providers after retries are exhausted or for non-retryable failures."""

    def __init__(self, message: str, *, provider: str, transient: bool = False,
                 quota_exhausted: bool = False):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.transient = transient
        # The provider asked to wait longer than we ever retry (e.g. a daily quota is used
        # up): further calls today will fail too, so callers should stop, not keep trying.
        self.quota_exhausted = quota_exhausted


class ProviderConfigError(ValueError):
    """A backend was selected but is not usable as configured (e.g. missing API key)."""


class LLMInterface(ABC):
    """Chat + vision interface every generation provider implements.

    Calls are synchronous (all three SDKs are); async callers run them in a thread.
    """

    name: str
    model_id: str

    @abstractmethod
    def generate(self, messages: list[ChatMessage], *, temperature: float | None = None,
                 max_tokens: int | None = None) -> str: ...

    @abstractmethod
    def stream(self, messages: list[ChatMessage], *, temperature: float | None = None,
               max_tokens: int | None = None) -> Iterator[str]: ...

    def describe_image(self, image: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        raise NotImplementedError(f"{self.name} provider does not support image input")
