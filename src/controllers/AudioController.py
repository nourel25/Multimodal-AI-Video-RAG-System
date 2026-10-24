import json

from stores.transcription import TranscriptionInterface, TranscriptionResult

from .BaseController import BaseController


class AudioController(BaseController):
    def __init__(self, transcriber: TranscriptionInterface, settings=None):
        super().__init__(settings)
        self.transcriber = transcriber

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        return self.transcriber.transcribe(audio_path, language=language)

    def save_transcript(self, result: TranscriptionResult, user_id: str,
                        youtube_video_id: str) -> str:
        """Persist timestamped segments so a video can be re-chunked without re-transcribing."""
        path = self.user_dir(self.transcripts_dir, user_id) / f"{youtube_video_id}.json"
        path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return str(path)
