from datetime import datetime

from bson import ObjectId
from pydantic import Field

from models.enums import VideoStatus

from .base import MongoDocument, utc_now


class Video(MongoDocument):
    video_user_id: ObjectId
    youtube_url: str
    youtube_video_id: str
    title: str = ""
    channel: str = ""
    duration_seconds: float = 0
    thumbnail_url: str = ""
    status: VideoStatus = VideoStatus.PENDING
    error: str | None = None
    language: str | None = None
    transcript_path: str | None = None
    transcript_chunks: int = 0
    visual_chunks: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
