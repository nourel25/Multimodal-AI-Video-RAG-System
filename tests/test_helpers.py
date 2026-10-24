import pytest

from helpers.text import detect_language, normalize, tokenize
from helpers.youtube import extract_video_id, format_timestamp, watch_url


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL123",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?si=abc",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ",
])
def test_extract_video_id_supports_common_url_shapes(url):
    assert extract_video_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    "https://vimeo.com/123456",
    "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/watch?v=short",
    "https://evil.com/watch?v=dQw4w9WgXcQ",
])
def test_extract_video_id_rejects_non_video_urls(url):
    assert extract_video_id(url) is None


def test_watch_url_adds_timestamp_only_when_positive():
    assert watch_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert watch_url("dQw4w9WgXcQ", 0) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert watch_url("dQw4w9WgXcQ", 222.7).endswith("&t=222s")


@pytest.mark.parametrize(("seconds", "expected"), [(0, "00:00"), (75.9, "01:15"),
                                                   (3725, "1:02:05")])
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_normalize_unifies_arabic_letter_variants_and_strips_diacritics():
    assert normalize("أَحْمَد") == normalize("احمد")
    assert normalize("مدرسة") == "مدرسه"
    assert normalize("مستشفى") == "مستشفي"


def test_tokenize_drops_stopwords_in_both_languages():
    assert tokenize("What is the dependency injection in FastAPI?") == [
        "dependency", "injection", "fastapi"]
    assert "في" not in tokenize("التعلم في الفيديو")


def test_detect_language():
    assert detect_language("ما هي غزوة تبوك؟") == "ar"
    assert detect_language("How does FastAPI validate data?") == "en"
