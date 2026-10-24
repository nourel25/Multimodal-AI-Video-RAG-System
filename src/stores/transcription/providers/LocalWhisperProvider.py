import logging

from ..TranscriptionInterface import (
    TranscriptionError,
    TranscriptionInterface,
    TranscriptionResult,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class LocalWhisperProvider(TranscriptionInterface):
    """faster-whisper running locally; decodes audio through PyAV, so no ffmpeg binary needed."""

    name = "local-whisper"

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model %s on %s", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        try:
            segments, info = self.model.transcribe(
                audio_path, language=language, vad_filter=True, beam_size=5
            )
            # `segments` is a lazy generator: transcription actually happens here.
            items = [
                TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
                for s in segments
                if s.text.strip()
            ]
        except Exception as exc:
            raise TranscriptionError(f"Local transcription failed: {exc}") from exc
        return TranscriptionResult(segments=items, language=info.language)
