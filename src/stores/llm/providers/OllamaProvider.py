import base64
from collections.abc import Iterator

import httpx
import ollama

from ..LLMInterface import ChatMessage, LLMError, LLMInterface
from ..retry import call_with_retry


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code == 429 or exc.status_code >= 500
    return isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))


class OllamaProvider(LLMInterface):
    name = "ollama"

    def __init__(self, host: str, model_id: str, *, temperature: float = 0.1,
                 max_tokens: int = 1024, max_retries: int = 3):
        self.client = ollama.Client(host=host)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _options(self, temperature, max_tokens) -> dict:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": max_tokens or self.max_tokens,
        }

    def _call(self, fn):
        return call_with_retry(fn, provider=self.name, is_transient=_is_transient,
                               max_attempts=self.max_retries)

    def generate(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        response = self._call(lambda: self.client.chat(
            model=self.model_id, messages=messages, options=self._options(temperature, max_tokens),
        ))
        return response["message"]["content"]

    def stream(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None
               ) -> Iterator[str]:
        chunks = self._call(lambda: self.client.chat(
            model=self.model_id, messages=messages, stream=True,
            options=self._options(temperature, max_tokens),
        ))
        try:
            for chunk in chunks:
                if text := chunk["message"]["content"]:
                    yield text
        except Exception as exc:
            raise LLMError(str(exc), provider=self.name, transient=_is_transient(exc)) from exc

    def describe_image(self, image: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        response = self._call(lambda: self.client.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt,
                       "images": [base64.b64encode(image).decode()]}],
            options=self._options(None, None),
        ))
        return response["message"]["content"]
