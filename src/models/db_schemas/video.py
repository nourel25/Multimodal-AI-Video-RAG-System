from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime, timezone
from bson.objectid import ObjectId
import re

class Video(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    video_user_id: ObjectId
    youtube_url: str
    audio_path: str
    transcript_path: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        
    class Config:
        arbitrary_types_allowed = True