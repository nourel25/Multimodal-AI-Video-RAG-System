from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from models.db_schemas import Video
from models.enums import VideoStatus

from .jobs import JobOut


class AddVideosRequest(BaseModel):
    youtube_urls: list[HttpUrl] = Field(..., min_length=1, max_length=20)
    auto_process: bool = Field(False, description="Queue a processing job for the added videos")
    include_visual: bool = True


class VideoOut(BaseModel):
    video_id: str
    youtube_url: str
    youtube_video_id: str
    title: str
    channel: str
    duration_seconds: float
    thumbnail_url: str
    status: VideoStatus
    error: str | None
    language: str | None
    transcript_chunks: int
    visual_chunks: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_video(cls, video: Video) -> "VideoOut":
        return cls(video_id=str(video.id), **video.model_dump(exclude={"id", "video_user_id",
                                                                        "transcript_path"}))


class SkippedUrl(BaseModel):
    url: str
    signal: str
    detail: str | None = None


class AddVideosResponse(BaseModel):
    signal: str
    user_id: str
    added: list[VideoOut]
    skipped: list[SkippedUrl]
    job: JobOut | None = None


class ListVideosResponse(BaseModel):
    signal: str
    videos: list[VideoOut]


class VideoResponse(BaseModel):
    signal: str
    video: VideoOut


class DeleteVideoResponse(BaseModel):
    signal: str
    video_id: str
    deleted_chunks: int
