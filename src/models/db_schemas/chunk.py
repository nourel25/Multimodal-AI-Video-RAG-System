from bson import ObjectId
from pydantic import BaseModel, Field

from models.enums import Modality

from .base import MongoDocument


class Chunk(MongoDocument):
    chunk_user_id: ObjectId
    chunk_video_id: ObjectId
    chunk_text: str = Field(..., min_length=1)
    chunk_order: int = Field(..., gt=0)
    start: float = 0
    end: float = 0
    modality: Modality = Modality.TRANSCRIPT
    chunk_metadata: dict = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    chunk_id: str
    video_id: str
    youtube_video_id: str
    video_title: str = ""
    text: str
    start: float = 0
    end: float = 0
    modality: Modality = Modality.TRANSCRIPT
    score: float
