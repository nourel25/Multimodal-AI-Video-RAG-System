import pytest

from controllers.ChunkController import ChunkController
from stores.transcription import TranscriptSegment


def segments(*texts, length=5.0):
    return [TranscriptSegment(i * length, (i + 1) * length, text) for i, text in enumerate(texts)]


def test_chunks_keep_exact_segment_timestamps():
    chunks = ChunkController().build_chunks(segments("a" * 40, "b" * 40, "c" * 40),
                                            max_chars=85, overlap_segments=0)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 10.0), (10.0, 15.0)]
    assert chunks[0].text == "a" * 40 + " " + "b" * 40


def test_overlap_repeats_trailing_segments():
    chunks = ChunkController().build_chunks(segments("one", "two", "three", "four"),
                                            max_chars=10, overlap_segments=1)
    assert [c.text for c in chunks] == ["one two", "two three", "three four"]


def test_segment_longer_than_limit_becomes_its_own_chunk():
    chunks = ChunkController().build_chunks(segments("x" * 500, "short"), max_chars=100,
                                            overlap_segments=3)
    assert [c.text for c in chunks] == ["x" * 500, "short"]


def test_overlap_larger_than_window_still_terminates():
    chunks = ChunkController().build_chunks(segments(*["word"] * 20), max_chars=4,
                                            overlap_segments=5)
    assert len(chunks) == 20


def test_empty_segments_are_ignored():
    assert ChunkController().build_chunks(segments(" ", ""), max_chars=100) == []


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        ChunkController().build_chunks(segments("a"), max_chars=0)
