from enum import StrEnum

from helpers.config import Settings
from stores.llm import ProviderConfigError

from .TranscriptionInterface import TranscriptionInterface


class TranscriptionBackend(StrEnum):
    LOCAL = "LOCAL"
    GROQ = "GROQ"


class TranscriptionProviderFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self, backend: str | None = None) -> TranscriptionInterface:
        s = self.settings
        backend = (backend or s.TRANSCRIPTION_BACKEND).upper()

        if backend == TranscriptionBackend.LOCAL.value:
            from .providers.LocalWhisperProvider import LocalWhisperProvider
            return LocalWhisperProvider(
                model_size=s.WHISPER_LOCAL_MODEL,
                device=s.WHISPER_DEVICE,
                compute_type=s.WHISPER_COMPUTE_TYPE,
            )

        if backend == TranscriptionBackend.GROQ.value:
            from .providers.GroqWhisperProvider import GroqWhisperProvider
            return GroqWhisperProvider(
                api_key=s.GROQ_API_KEY, model_id=s.GROQ_WHISPER_MODEL,
                max_retries=s.LLM_MAX_RETRIES,
            )

        raise ProviderConfigError(f"Unsupported transcription backend: {backend!r}")
