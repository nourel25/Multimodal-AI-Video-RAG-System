import base64
from collections.abc import Iterator

import groq

from ..LLMInterface import ChatMessage, LLMError, LLMInterface, ProviderConfigError
from ..retry import call_with_retry


def is_transient_groq_error(exc: BaseException) -> bool:
    if isinstance(exc, (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError)):
        return True
    return isinstance(exc, groq.APIStatusError) and exc.status_code >= 500


class GroqProvider(LLMInterface):
    name = "groq"

    def __init__(self, api_key: str, model_id: str, *, temperature: float = 0.1,
                 max_tokens: int = 1024, max_retries: int = 3):
        if not api_key:
            raise ProviderConfigError("GROQ_API_KEY is required for the GROQ backend")
        # Retries are handled by call_with_retry so all providers behave the same way.
        self.client = groq.Groq(api_key=api_key, max_retries=0)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _call(self, fn):
        return call_with_retry(fn, provider=self.name, is_transient=is_transient_groq_error,
                               max_attempts=self.max_retries)

    def _create(self, messages, temperature, max_tokens, stream: bool):
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=stream,
        )

    def generate(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        response = self._call(lambda: self._create(messages, temperature, max_tokens, False))
        return response.choices[0].message.content or ""

    def stream(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None
               ) -> Iterator[str]:
        chunks = self._call(lambda: self._create(messages, temperature, max_tokens, True))
        try:
            for chunk in chunks:
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    yield text
        except Exception as exc:
            raise LLMError(str(exc), provider=self.name,
                           transient=is_transient_groq_error(exc)) from exc

    def describe_image(self, image: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode()}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
        response = self._call(lambda: self._create(messages, None, None, False))
        return response.choices[0].message.content or ""
