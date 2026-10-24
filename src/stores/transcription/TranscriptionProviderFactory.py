import logging
from collections.abc import Callable
from enum import StrEnum

from helpers.config import Settings
from stores.llm import ProviderConfigError

from .TranscriptionInterface import TranscriptionError, TranscriptionInterface, TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionBackend(StrEnum):
    LOCAL = "LOCAL"
    GROQ = "GROQ"


class FallbackTranscriber(TranscriptionInterface):
    """Uses the fallback when the primary fails, e.g. after a free-tier quota is exhausted.

    The fallback is built on first use only, so a local Whisper model is never loaded
    while the hosted API keeps working.
    """

    def __init__(self, primary: TranscriptionInterface,
                 fallback_factory: Callable[[], TranscriptionInterface]):
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: TranscriptionInterface | None = None
        self.name = f"{primary.name}->fallback"

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        try:
            return self.primary.transcribe(audio_path, language)
        except TranscriptionError as exc:
            logger.warning("Primary transcription failed, using fallback: %s", exc)
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            return self._fallback.transcribe(audio_path, language)


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
                requests_per_minute=s.GROQ_WHISPER_REQUESTS_PER_MINUTE,
            )

        raise ProviderConfigError(f"Unsupported transcription backend: {backend!r}")

    def create_with_fallback(self) -> TranscriptionInterface:
        primary = self.create()
        fallback = self.settings.TRANSCRIPTION_FALLBACK_BACKEND
        if fallback and fallback.upper() != self.settings.TRANSCRIPTION_BACKEND.upper():
            return FallbackTranscriber(primary, lambda: self.create(fallback))
        return primary
