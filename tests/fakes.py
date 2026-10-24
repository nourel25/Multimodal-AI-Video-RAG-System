"""Deterministic stand-ins for models and external services."""

import hashlib
import math
from pathlib import Path

from controllers.VideoController import VideoMetadata
from helpers.text import tokenize
from stores.embedding.EmbeddingInterface import EmbeddingInterface, RerankerInterface
from stores.llm import LLMError, LLMInterface
from stores.transcription import TranscriptionInterface, TranscriptionResult, TranscriptSegment


class FakeEmbedder(EmbeddingInterface):
    """Hashed bag-of-words: texts sharing words get similar vectors, like a real model would."""

    size = 64

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class KeywordReranker(RerankerInterface):
    """Scores documents by how many query words they contain."""

    def score(self, query, documents):
        terms = set(tokenize(query))
        return [float(len(terms & set(tokenize(doc)))) for doc in documents]


class FakeLLM(LLMInterface):
    name = "fake"
    model_id = "fake-1"

    def __init__(self, answer: str = "FastAPI is built on Starlette [1].", fail: bool = False):
        self.answer = answer
        self.fail = fail
        self.calls: list[list] = []

    def generate(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if self.fail:
            raise LLMError("boom", provider=self.name, transient=True)
        return self.answer

    def stream(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if self.fail:
            raise LLMError("boom", provider=self.name, transient=True)
        yield from self.answer.split(" ")

    def describe_image(self, image, prompt, mime_type="image/jpeg"):
        return "A slide titled Dependency Injection.\nText: Depends()"


SEGMENTS = [
    TranscriptSegment(0.0, 6.0, "FastAPI is a modern web framework for building APIs with Python."),
    TranscriptSegment(6.0, 12.0, "It is fast thanks to Starlette for the web parts."),
    TranscriptSegment(12.0, 18.0, "Pydantic handles data validation and serialization."),
    TranscriptSegment(18.0, 25.0, "You get automatic interactive documentation with Swagger UI."),
    TranscriptSegment(25.0, 33.0, "Use the dependency injection system to keep code testable."),
]


class FakeTranscriber(TranscriptionInterface):
    name = "fake-whisper"

    def __init__(self, segments=None, fail: bool = False):
        self.segments = SEGMENTS if segments is None else segments
        self.fail = fail
        self.languages: list = []

    def transcribe(self, audio_path, language=None):
        self.languages.append(language)
        if self.fail:
            raise RuntimeError("whisper exploded")
        return TranscriptionResult(segments=list(self.segments), language="en")


class FakeDownloader:
    """Replaces VideoController's network calls (yt-dlp) but keeps its validation logic."""

    def __init__(self, video_controller, durations: dict | None = None):
        self.inner = video_controller
        self.durations = durations or {}
        self.validate_metadata = video_controller.validate_metadata

    def fetch_metadata(self, youtube_video_id):
        if youtube_video_id == "unavailable":
            raise RuntimeError("Video unavailable")
        return VideoMetadata(
            youtube_video_id=youtube_video_id,
            title=f"Video {youtube_video_id}",
            channel="Test Channel",
            duration_seconds=self.durations.get(youtube_video_id, 33.0),
            thumbnail_url="",
            is_live=False,
        )

    def _write(self, base: Path, user_id: str, youtube_video_id: str, ext: str) -> str:
        path = self.inner.user_dir(base, user_id) / f"{youtube_video_id}.{ext}"
        path.write_bytes(b"fake media")
        return str(path)

    def download_audio(self, youtube_video_id, user_id):
        return self._write(self.inner.audios_dir, user_id, youtube_video_id, "webm")

    def download_video(self, youtube_video_id, user_id):
        return self._write(self.inner.videos_dir, user_id, youtube_video_id, "mp4")
