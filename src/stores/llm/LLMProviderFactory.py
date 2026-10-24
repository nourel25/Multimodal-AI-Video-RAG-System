from helpers.config import Settings

from .FallbackLLMProvider import FallbackLLMProvider
from .LLMEnums import LLMBackend
from .LLMInterface import LLMInterface, ProviderConfigError


class LLMProviderFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self, backend: str, *, vision: bool = False) -> LLMInterface:
        """Build a provider; `vision=True` selects the provider's image-capable model."""
        s = self.settings
        common = {
            "temperature": s.GENERATION_TEMPERATURE,
            "max_tokens": s.GENERATION_MAX_TOKENS,
            "max_retries": s.LLM_MAX_RETRIES,
        }
        backend = backend.upper()

        # Provider SDKs are imported lazily so unused backends need no installed SDK or key.
        if backend == LLMBackend.OLLAMA.value:
            from .providers.OllamaProvider import OllamaProvider
            model = s.OLLAMA_VISION_MODEL if vision else s.OLLAMA_MODEL
            return OllamaProvider(host=s.OLLAMA_HOST, model_id=model, **common)

        if backend == LLMBackend.GROQ.value:
            from .providers.GroqProvider import GroqProvider
            model = s.GROQ_VISION_MODEL if vision else s.GROQ_MODEL
            if not model:
                raise ProviderConfigError("GROQ_VISION_MODEL must be set to use Groq for vision")
            return GroqProvider(api_key=s.GROQ_API_KEY, model_id=model, **common)

        if backend == LLMBackend.GEMINI.value:
            from .providers.GeminiProvider import GeminiProvider
            model = s.GEMINI_VISION_MODEL if vision else s.GEMINI_MODEL
            return GeminiProvider(api_key=s.GEMINI_API_KEY, model_id=model, **common)

        raise ProviderConfigError(f"Unsupported LLM backend: {backend!r}")

    def create_generation_client(self) -> LLMInterface:
        primary = self.create(self.settings.GENERATION_BACKEND)
        fallback = self.settings.GENERATION_FALLBACK_BACKEND
        if fallback and fallback.upper() != self.settings.GENERATION_BACKEND.upper():
            return FallbackLLMProvider(primary, self.create(fallback))
        return primary

    def create_vision_client(self) -> LLMInterface | None:
        if not self.settings.VISION_BACKEND:
            return None
        return self.create(self.settings.VISION_BACKEND, vision=True)
