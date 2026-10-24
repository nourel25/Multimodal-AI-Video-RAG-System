from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, Field

from models.enums import JobStage, JobStatus, VideoStatus

from .base import MongoDocument, utc_now


class ProcessOptions(BaseModel):
    chunk_max_chars: int = 800
    chunk_overlap_segments: int = 1
    include_visual: bool = True
    language: str | None = None  # None lets the transcriber detect it


class JobVideoResult(BaseModel):
    video_id: str
    youtube_url: str
    status: VideoStatus
    transcript_chunks: int = 0
    visual_chunks: int = 0
    error: str | None = None


class Job(MongoDocument):
    job_user_id: ObjectId
    user_id: str
    video_ids: list[ObjectId]
    options: ProcessOptions = Field(default_factory=ProcessOptions)
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.QUEUED
    progress: float = 0
    current_video_id: str | None = None
    results: list[JobVideoResult] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
