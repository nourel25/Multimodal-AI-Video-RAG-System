import pytest

from controllers.VisionController import Keyframe, VisionController
from stores.templates.template_parser import TemplateParser
from tests.fakes import FakeLLM


class CaptionLLM(FakeLLM):
    def __init__(self, captions):
        super().__init__()
        self.captions = iter(captions)

    def describe_image(self, image, prompt, mime_type="image/jpeg"):
        return next(self.captions)


def test_merge_spans_keeps_full_coverage():
    spans = [(float(i), float(i + 1)) for i in range(10)]
    merged = VisionController._merge_spans(spans, 3)
    assert len(merged) == 3
    assert merged[0][0] == 0.0
    assert merged[-1][1] == 10.0


def test_captions_marked_none_are_dropped(settings):
    vision = VisionController(CaptionLLM(["A chart of latency", "NONE", "  "]),
                              TemplateParser(default_language="en"), settings)
    frames = [Keyframe(start=i, end=i + 1, timestamp=i + 0.5, image=b"x") for i in range(3)]
    chunks = vision.caption_keyframes(frames, "Talk")
    assert [(c.text, c.start) for c in chunks] == [("A chart of latency", 0)]


def test_extract_keyframes_from_real_video(settings, tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scenedetect")

    path = str(tmp_path / "scenes.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):  # three 5-second "scenes"
        frame = np.full((120, 160, 3), color, dtype=np.uint8)
        for _ in range(50):
            writer.write(frame)
    writer.release()

    settings.VISION_MIN_SCENE_SECONDS = 1.0
    vision = VisionController(FakeLLM(), TemplateParser(default_language="en"), settings)
    keyframes = vision.extract_keyframes(path, duration=15.0)
    assert len(keyframes) == 3
    assert keyframes[0].image[:2] == b"\xff\xd8"  # JPEG magic bytes
    assert keyframes[0].start == 0.0


class QuotaExhaustedLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def describe_image(self, image, prompt, mime_type="image/jpeg"):
        from stores.llm import LLMError

        self.attempts += 1
        raise LLMError("429 quota exceeded", provider="fake", transient=True)


def test_captioning_stops_after_repeated_failures(settings):
    llm = QuotaExhaustedLLM()
    settings.VISION_MAX_CONSECUTIVE_FAILURES = 3
    vision = VisionController(llm, TemplateParser(default_language="en"), settings)
    frames = [Keyframe(start=i, end=i + 1, timestamp=i, image=b"x") for i in range(10)]
    assert vision.caption_keyframes(frames, "Talk") == []
    assert llm.attempts == 3  # not 10: the circuit breaker saved 7 quota-burning calls
