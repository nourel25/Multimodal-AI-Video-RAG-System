from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "segments": [vars(segment) for segment in self.segments],
        }


class TranscriptionError(RuntimeError):
    pass


class TranscriptionInterface(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file; `language=None` means auto-detect."""
