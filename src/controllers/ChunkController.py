from dataclasses import dataclass

from stores.transcription import TranscriptSegment


@dataclass
class TimedChunk:
    text: str
    start: float
    end: float


class ChunkController:
    """Groups transcript segments into chunks without ever splitting a segment, so every
    chunk keeps exact start/end timestamps for citations."""

    def build_chunks(self, segments: list[TranscriptSegment], max_chars: int = 800,
                     overlap_segments: int = 1) -> list[TimedChunk]:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        overlap_segments = max(overlap_segments, 0)

        segments = [s for s in segments if s.text.strip()]
        chunks: list[TimedChunk] = []
        start_index = 0

        while start_index < len(segments):
            end_index = start_index
            length = len(segments[start_index].text)
            # Grow the window while the next segment still fits.
            while (end_index + 1 < len(segments)
                   and length + 1 + len(segments[end_index + 1].text) <= max_chars):
                end_index += 1
                length += 1 + len(segments[end_index].text)

            window = segments[start_index:end_index + 1]
            chunks.append(TimedChunk(
                text=" ".join(s.text.strip() for s in window),
                start=window[0].start,
                end=window[-1].end,
            ))

            if end_index + 1 >= len(segments):
                break
            # Overlap by whole segments, but always move forward by at least one.
            start_index = max(end_index + 1 - overlap_segments, start_index + 1)

        return chunks
