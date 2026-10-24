import logging
from dataclasses import dataclass

from stores.llm import LLMError, LLMInterface
from stores.templates.template_parser import TemplateParser

from .BaseController import BaseController
from .ChunkController import TimedChunk

logger = logging.getLogger(__name__)

MAX_FRAME_WIDTH = 768
JPEG_QUALITY = 80
# Mean absolute pixel difference (0-255) on a 32x32 thumbnail below which two keyframes
# are treated as the same shot, e.g. a static talking head.
DUPLICATE_FRAME_THRESHOLD = 6.0


@dataclass
class Keyframe:
    start: float
    end: float
    timestamp: float
    image: bytes


def _seconds(timecode) -> float:
    if hasattr(timecode, "get_seconds"):
        return float(timecode.get_seconds())
    return float(timecode.seconds)


class VisionController(BaseController):
    """Turns what is *shown* in a video into searchable text chunks.

    Pipeline: scene detection -> one keyframe per scene (merged/deduplicated down to
    VISION_MAX_FRAMES) -> vision-LLM caption + on-screen text for each keyframe.
    """

    def __init__(self, vision_llm: LLMInterface, template_parser: TemplateParser,
                 settings=None):
        super().__init__(settings)
        self.vision_llm = vision_llm
        self.template_parser = template_parser

    # ---------- scene detection ----------

    def detect_scenes(self, video_path: str, duration: float) -> list[tuple[float, float]]:
        from scenedetect import ContentDetector, detect

        s = self.app_settings
        try:
            scenes = detect(video_path, ContentDetector(threshold=s.VISION_SCENE_THRESHOLD,
                                                        min_scene_len=s.VISION_MIN_SCENE_SECONDS))
            spans = [(_seconds(start), _seconds(end)) for start, end in scenes]
        except Exception as exc:
            logger.warning("Scene detection failed, sampling uniformly: %s", exc)
            spans = []

        if len(spans) < 2 and duration > 0:
            # No cuts found (single shot): sample at a fixed interval instead.
            step = max(duration / s.VISION_MAX_FRAMES, s.VISION_MIN_SCENE_SECONDS * 2)
            spans, t = [], 0.0
            while t < duration:
                spans.append((t, min(t + step, duration)))
                t += step
        return self._merge_spans(spans, s.VISION_MAX_FRAMES)

    @staticmethod
    def _merge_spans(spans: list[tuple[float, float]], max_spans: int
                     ) -> list[tuple[float, float]]:
        """Merge adjacent scenes into at most `max_spans` groups, keeping full time coverage."""
        if len(spans) <= max_spans:
            return spans
        merged = []
        group_size = len(spans) / max_spans
        for i in range(max_spans):
            group = spans[int(i * group_size):int((i + 1) * group_size)]
            if group:
                merged.append((group[0][0], group[-1][1]))
        return merged

    # ---------- frame extraction ----------

    def extract_keyframes(self, video_path: str, duration: float) -> list[Keyframe]:
        import cv2
        import numpy as np

        spans = self.detect_scenes(video_path, duration)
        capture = cv2.VideoCapture(video_path)
        keyframes: list[Keyframe] = []
        previous_thumb = None
        try:
            for start, end in spans:
                timestamp = (start + end) / 2
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32))
                thumb = thumb.astype(np.float32)
                if (previous_thumb is not None and keyframes
                        and np.abs(thumb - previous_thumb).mean() < DUPLICATE_FRAME_THRESHOLD):
                    keyframes[-1].end = end  # same shot continues
                    continue
                previous_thumb = thumb

                height, width = frame.shape[:2]
                if width > MAX_FRAME_WIDTH:
                    frame = cv2.resize(frame, (MAX_FRAME_WIDTH,
                                               int(height * MAX_FRAME_WIDTH / width)))
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    keyframes.append(Keyframe(start=start, end=end, timestamp=timestamp,
                                              image=buffer.tobytes()))
        finally:
            capture.release()
        return keyframes

    # ---------- captioning ----------

    def caption_keyframes(self, keyframes: list[Keyframe], video_title: str,
                          language: str | None = None) -> list[TimedChunk]:
        prompt = self.template_parser.get("vision", "frame_prompt",
                                          {"video_title": video_title}, language=language)
        chunks: list[TimedChunk] = []
        for keyframe in keyframes:
            try:
                caption = self.vision_llm.describe_image(keyframe.image, prompt).strip()
            except (LLMError, NotImplementedError) as exc:
                logger.warning("Captioning frame at %.1fs failed: %s", keyframe.timestamp, exc)
                continue
            if not caption or caption.upper().startswith("NONE"):
                continue
            chunks.append(TimedChunk(text=caption, start=keyframe.start, end=keyframe.end))
        return chunks
