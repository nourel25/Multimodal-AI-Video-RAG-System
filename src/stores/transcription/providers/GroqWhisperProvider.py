import os

import groq

from stores.llm import ProviderConfigError
from stores.llm.providers.GroqProvider import is_transient_groq_error
from stores.llm.rate_limit import shared_limiter
from stores.llm.retry import call_with_retry

from ..TranscriptionInterface import (
    TranscriptionError,
    TranscriptionInterface,
    TranscriptionResult,
    TranscriptSegment,
)

# Groq rejects larger uploads on the free tier; the downloader picks low-bitrate audio
# and MAX_VIDEO_DURATION_MINUTES keeps files under this in practice.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# Groq's verbose_json reports the language by name ("English"); store ISO codes like local
# Whisper does, so the prompt language and stored metadata stay consistent.
_LANGUAGE_CODES = {"english": "en", "arabic": "ar", "french": "fr", "german": "de",
                   "spanish": "es", "turkish": "tr", "urdu": "ur", "hindi": "hi"}


def language_code(language: str | None) -> str | None:
    if not language:
        return None
    language = language.strip().lower()
    return _LANGUAGE_CODES.get(language, language)


def _field(item, name):
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


class GroqWhisperProvider(TranscriptionInterface):
    name = "groq-whisper"

    def __init__(self, api_key: str, model_id: str = "whisper-large-v3-turbo",
                 max_retries: int = 3, requests_per_minute: float = 0):
        if not api_key:
            raise ProviderConfigError("GROQ_API_KEY is required for the GROQ transcription backend")
        self.client = groq.Groq(api_key=api_key, max_retries=0)
        self.model_id = model_id
        self.max_retries = max_retries
        self.limiter = shared_limiter(f"groq-whisper:{model_id}", requests_per_minute)

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        size = os.path.getsize(audio_path)
        if size > MAX_UPLOAD_BYTES:
            raise TranscriptionError(
                f"Audio is {size / 1_048_576:.1f} MB; Groq accepts up to "
                f"{MAX_UPLOAD_BYTES / 1_048_576:.0f} MB. Use a shorter video or the LOCAL backend."
            )

        with open(audio_path, "rb") as audio_file:
            payload = (os.path.basename(audio_path), audio_file.read())

        kwargs = {"language": language} if language else {}
        try:
            response = call_with_retry(
                lambda: self.client.audio.transcriptions.create(
                    file=payload,
                    model=self.model_id,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    **kwargs,
                ),
                provider=self.name,
                is_transient=is_transient_groq_error,
                max_attempts=self.max_retries,
                limiter=self.limiter,
            )
        except Exception as exc:
            raise TranscriptionError(f"Groq transcription failed: {exc}") from exc

        # verbose_json fields aren't in the SDK's typed model; they arrive as extras.
        raw_segments = _field(response, "segments") or []
        segments = [
            TranscriptSegment(start=float(_field(s, "start")), end=float(_field(s, "end")),
                              text=str(_field(s, "text")).strip())
            for s in raw_segments
            if str(_field(s, "text") or "").strip()
        ]
        if not segments and (text := (_field(response, "text") or "").strip()):
            segments = [TranscriptSegment(start=0.0, end=0.0, text=text)]
        return TranscriptionResult(segments=segments,
                                   language=language_code(_field(response, "language")))
