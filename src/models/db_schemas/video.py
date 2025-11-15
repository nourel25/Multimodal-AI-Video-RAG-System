from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional
from datetime import datetime, timezone
from bson.objectid import ObjectId
import re

class Video(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    user_id: str = Field(..., min_length=3)
    youtube_url: str
    file_path: str
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        
    @field_validator('user_id')
    def validate_project_id(cls, value):
        if not re.match(r'^[A-Za-z0-9_]+$', value):
            raise ValueError("user_id may contain only letters, numbers, and underscore")
        return value

    class Config:
        arbitrary_types_allowed = True