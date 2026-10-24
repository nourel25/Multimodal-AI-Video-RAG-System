from collections.abc import Iterator

from google import genai
from google.genai import errors, types

from ..LLMEnums import ChatRole
from ..LLMInterface import ChatMessage, LLMError, LLMInterface, ProviderConfigError
from ..retry import call_with_retry


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.APIError):
        return exc.code == 429
    return isinstance(exc, (ConnectionError, TimeoutError))


class GeminiProvider(LLMInterface):
    name = "gemini"

    def __init__(self, api_key: str, model_id: str, *, temperature: float = 0.1,
                 max_tokens: int = 1024, max_retries: int = 3):
        if not api_key:
            raise ProviderConfigError("GEMINI_API_KEY is required for the GEMINI backend")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _call(self, fn):
        return call_with_retry(fn, provider=self.name, is_transient=_is_transient,
                               max_attempts=self.max_retries)

    def _build_request(self, messages: list[ChatMessage], temperature, max_tokens):
        # Gemini takes the system prompt as config and names the assistant role "model".
        system = "\n\n".join(m["content"] for m in messages if m["role"] == ChatRole.SYSTEM.value)
        contents = [
            types.Content(
                role="model" if m["role"] == ChatRole.ASSISTANT.value else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
            if m["role"] != ChatRole.SYSTEM.value
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self.max_tokens,
        )
        return contents, config

    def generate(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        contents, config = self._build_request(messages, temperature, max_tokens)
        response = self._call(lambda: self.client.models.generate_content(
            model=self.model_id, contents=contents, config=config,
        ))
        return response.text or ""

    def stream(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None
               ) -> Iterator[str]:
        contents, config = self._build_request(messages, temperature, max_tokens)
        chunks = self._call(lambda: self.client.models.generate_content_stream(
            model=self.model_id, contents=contents, config=config,
        ))
        try:
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise LLMError(str(exc), provider=self.name, transient=_is_transient(exc)) from exc

    def describe_image(self, image: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        contents = [types.Part.from_bytes(data=image, mime_type=mime_type), prompt]
        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=self.max_tokens)
        response = self._call(lambda: self.client.models.generate_content(
            model=self.model_id, contents=contents, config=config,
        ))
        return response.text or ""
