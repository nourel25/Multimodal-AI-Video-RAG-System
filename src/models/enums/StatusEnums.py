from enum import StrEnum


class VideoStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"  # some videos failed
    FAILED = "failed"


class JobStage(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    CHUNKING = "chunking"
    EXTRACTING_FRAMES = "extracting_frames"
    CAPTIONING = "captioning"
    INDEXING = "indexing"
    DONE = "done"


class Modality(StrEnum):
    TRANSCRIPT = "transcript"
    VISUAL = "visual"
