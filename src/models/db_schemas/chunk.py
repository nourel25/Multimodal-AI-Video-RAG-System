from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId

class Chunk(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    chunk_user_id: ObjectId
    chunk_video_id: ObjectId
    chunk_text: str = Field(..., min_length=1)
    chunk_order: int = Field(..., gt=0)
    chunk_metadata: dict

    class Config:
        arbitrary_types_allowed = True